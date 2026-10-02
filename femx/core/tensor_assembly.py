import numpy as np
import scipy.sparse as sp
import torch
from typing import Optional, Tuple
from femx.backends.numpy_backend import ndarray
from femx.core.mesh import Mesh
from femx.core.dofs import DofMap
from femx.core.state import State
from femx.core.routing import RoutingData, build_routing_matrices
from femx.core.tensor_geometry import (
    BatchedGeometry,
    evaluate_batched_geometry,
    integrate_nodal_source,
    quadrature_coordinates,
)
from femx.core.loads import sample_body_load_batch


def compute_batch_map_unified(
    mesh: Mesh,
    formulation,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    body_load=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified Stage I Batch-Map Engine using true physical continuum mechanics tensor orders.
    Evaluates element stiffness matrices across any PDE formulation (heat, elasticity, coupled multi-physics).
    """
    # 1. Evaluate physics-independent geometry
    geom = evaluate_batched_geometry(mesh, device=device, dtype=dtype)

    # 2. Get physical material tensors from formulation
    tensors = formulation.get_physical_tensors(geom, device=device, dtype=dtype)

    # 3. Unified Contraction delegated to Formulation
    K_local, M_local, F_local = formulation.compute_batch_map(
        geom, tensors, device=device, dtype=dtype, body_load=body_load,
    )

    return K_local, M_local, F_local


def assemble_system_tensor(
    dof_map: DofMap,
    formulation,
    field_name: str = None,
    routing: RoutingData = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    body_load=None,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, ndarray, torch.Tensor, torch.Tensor]:
    """
    Full TensorGalerkin Monolithic Assembly using Unified Batch-Map and SpMM Sparse-Reduce.

    ``body_load`` uses the same forms as ``femx.core.loads.sample_body_load``.
    Coupled thermoelasticity takes ``{"u": ..., "T": ...}``.

    This path is for **state-independent** linear (or linearized-at-zero) systems.
    For finite-strain Neo-Hookean use :func:`assemble_nonlinear_system_tensor`.
    """
    mesh = dof_map.geometry

    if field_name is None:
        if hasattr(formulation, "field_names") and formulation.field_names is not None:
            field_name = list(formulation.field_names) if isinstance(formulation.field_names, (tuple, list)) else formulation.field_names
        else:
            raise ValueError("Must specify field_name or formulation must specify field_names")

    # 1. Precompute or retrieve RoutingData for Stage II
    if routing is None:
        routing = build_routing_matrices(mesh, dof_map, field_name, device=device, dtype=dtype)

    # 2. Stage I: Unified Batch-Map
    K_local, M_local, F_local = compute_batch_map_unified(
        mesh, formulation, device=device, dtype=dtype, body_load=body_load,
    )

    # 3. Stage II: Unified Sparse-Reduce via SpMM
    m_K = K_local.reshape(-1, 1)  # (E * k^2, 1)
    m_M = M_local.reshape(-1, 1)
    m_F = F_local.reshape(-1, 1)  # (E * k, 1)

    v_K = torch.sparse.mm(routing.S_mat, m_K).squeeze(1)  # (N_nnz,)
    v_M = torch.sparse.mm(routing.S_mat, m_M).squeeze(1)
    f_tensor = torch.sparse.mm(routing.S_vec, m_F).squeeze(1)  # (N_dofs,)

    row_ptrs = routing.crow_indices.cpu().numpy()
    col_idxs = routing.col_indices.cpu().numpy()
    values_K = v_K.cpu().detach().numpy()
    values_M = v_M.cpu().detach().numpy()

    K_csr = sp.csr_matrix((values_K, col_idxs, row_ptrs), shape=(routing.n_dofs, routing.n_dofs))
    M_csr = sp.csr_matrix((values_M, col_idxs, row_ptrs), shape=(routing.n_dofs, routing.n_dofs))
    f_np = f_tensor.cpu().detach().numpy()

    return K_csr, M_csr, f_np, K_local, F_local


class NonlinearTensorCache:
    """Cached geometry / routing for repeated Neo-Hookean Newton assemblies."""

    def __init__(
        self,
        geom: BatchedGeometry,
        routing: RoutingData,
        elem_dofs: torch.Tensor,
        lambda_: float,
        mu: float,
        F_e: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.geom = geom
        self.routing = routing
        self.elem_dofs = elem_dofs
        self.lambda_ = lambda_
        self.mu = mu
        self.F_e = F_e
        self.device = device
        self.dtype = dtype


def build_nonlinear_tensor_cache(
    dof_map: DofMap,
    material,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    body_load=None,
    routing: Optional[RoutingData] = None,
) -> NonlinearTensorCache:
    """Precompute geometry, DOF table, and Stage-II routing for Neo-Hookean."""
    from femx.core.torch_kernels import element_dof_indices

    mesh = dof_map.geometry
    if not isinstance(mesh, Mesh):
        raise TypeError("Torch nonlinear assembly currently supports Mesh only.")
    geom = evaluate_batched_geometry(mesh, device=device, dtype=dtype)
    if routing is None:
        routing = build_routing_matrices(mesh, dof_map, "u", device=device, dtype=dtype)
    elem_dofs = element_dof_indices(mesh, dof_map, "u", device=device)
    lambda_, mu = material.get_lame_parameters()

    F_e = None
    if body_load is not None:
        f_gp = torch.tensor(
            sample_body_load_batch(body_load, quadrature_coordinates(geom), n_comp=geom.dim),
            dtype=dtype,
            device=device,
        )
        F_e = integrate_nodal_source(geom, f_gp)

    return NonlinearTensorCache(
        geom=geom,
        routing=routing,
        elem_dofs=elem_dofs,
        lambda_=lambda_,
        mu=mu,
        F_e=F_e,
        device=device,
        dtype=dtype,
    )


def assemble_nonlinear_system_tensor(
    dof_map: DofMap,
    formulation,
    state: State,
    *,
    routing: Optional[RoutingData] = None,
    cache: Optional[NonlinearTensorCache] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    body_load=None,
) -> Tuple[sp.csr_matrix, ndarray]:
    """Assemble Neo-Hookean ``K(U)`` and ``R(U)`` via Torch Stage I/II.

    Returns SciPy CSR tangent and NumPy residual for host BCs / sparse solve.
    Geometry and routing are reused when ``cache`` is supplied.
    """
    from femx.core.torch_kernels import neohookean_system_global

    field_names = list(getattr(formulation, "field_names", ("u",)))
    if field_names != ["u"]:
        raise NotImplementedError(
            "Torch nonlinear assembly currently supports single-field Neo-Hookean only."
        )
    material = formulation.material
    if not hasattr(material, "get_lame_parameters"):
        raise TypeError("formulation.material must provide get_lame_parameters().")

    if cache is None:
        cache = build_nonlinear_tensor_cache(
            dof_map,
            material,
            device=device,
            dtype=dtype,
            body_load=body_load,
            routing=routing,
        )
    elif body_load is not None and cache.F_e is None:
        raise ValueError("body_load was passed but cache has no F_e; rebuild the cache.")

    U_np = state.pack_vector(dof_map)
    U = torch.tensor(U_np, dtype=cache.dtype, device=cache.device)
    R_t, K_values = neohookean_system_global(
        cache.geom,
        U,
        cache.elem_dofs,
        cache.routing,
        cache.lambda_,
        cache.mu,
        F_e=cache.F_e,
    )

    row_ptrs = cache.routing.crow_indices.cpu().numpy()
    col_idxs = cache.routing.col_indices.cpu().numpy()
    values_K = K_values.detach().cpu().numpy()
    K_csr = sp.csr_matrix(
        (values_K, col_idxs, row_ptrs),
        shape=(cache.routing.n_dofs, cache.routing.n_dofs),
    )
    R_np = R_t.detach().cpu().numpy()
    return K_csr, R_np
