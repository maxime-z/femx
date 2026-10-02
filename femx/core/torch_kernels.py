"""Differentiable PyTorch element residuals for heat, elasticity, and Neo-Hookean.

These kernels reuse :class:`~femx.core.tensor_geometry.BatchedGeometry` and gather
or scatter with element DOF indices. Large iterative solvers are out of scope;
``torch.func.jacrev`` is intended for tiny-mesh derivative checks.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from femx.core.dofs import DofMap
from femx.core.mesh import Mesh
from femx.core.tensor_geometry import BatchedGeometry
from femx.materials.hyperelastic import first_piola_torch, neohookean_stress_tangent_torch


def element_dof_indices(
    mesh: Mesh,
    dof_map: DofMap,
    field_names: Union[str, Sequence[str]],
    device: str = "cpu",
) -> torch.Tensor:
    """Return element DOF index table of shape ``(n_elem, n_local_dofs)``."""
    if isinstance(field_names, str):
        names = [field_names]
    else:
        names = list(field_names)
    rows = [dof_map.get_element_dofs_multi(names, cell) for cell in mesh.cells]
    return torch.tensor(np.asarray(rows, dtype=np.int64), device=device)


def gather_element_dofs(U: torch.Tensor, elem_dofs: torch.Tensor) -> torch.Tensor:
    """Gather global unknowns into ``(E, k)`` element blocks."""
    return U[elem_dofs]


def scatter_element_residual(
    R_e: torch.Tensor,
    elem_dofs: torch.Tensor,
    n_dofs: int,
) -> torch.Tensor:
    """Scatter element residuals into a global vector with ``index_add``."""
    R = torch.zeros(n_dofs, dtype=R_e.dtype, device=R_e.device)
    return R.index_add(0, elem_dofs.reshape(-1), R_e.reshape(-1))


def lame_parameters_torch(E: torch.Tensor, nu: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Lame parameters from Young's modulus (torch scalar) and Poisson ratio."""
    lambda_ = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lambda_, mu


def elasticity_tensor_4th_torch(
    E: torch.Tensor,
    nu: float,
    dim: int,
    mode: str = "plane_strain",
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Isotropic 4th-order elasticity tensor ``C4`` of shape ``(dim, dim, dim, dim)``."""
    if device is None:
        device = E.device
    lambda_, mu = lame_parameters_torch(E, nu)
    if mode == "plane_stress":
        lambda_star = (E * nu) / (1.0 - nu**2)
    else:
        lambda_star = lambda_
    delta = torch.eye(dim, dtype=dtype, device=device)
    return (
        lambda_star * torch.einsum("ij,kl->ijkl", delta, delta)
        + mu
        * (
            torch.einsum("ik,jl->ijkl", delta, delta)
            + torch.einsum("il,jk->ijkl", delta, delta)
        )
    )


def heat_stiffness_local(geom: BatchedGeometry, conductivity: torch.Tensor) -> torch.Tensor:
    """Element heat stiffness matrices, shape ``(E, nen, nen)``."""
    eye = torch.eye(geom.dim, dtype=geom.G.dtype, device=geom.G.device)
    K_2nd = conductivity * eye
    return torch.einsum(
        "q,eq,eqai,ij,eqbj->eab", geom.W_hat, geom.detJ, geom.G, K_2nd, geom.G
    )


def heat_residual_local(
    geom: BatchedGeometry,
    conductivity: torch.Tensor,
    T_e: torch.Tensor,
    F_e: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Local heat residual ``K_e T_e - F_e``, shape ``(E, nen)``."""
    K_e = heat_stiffness_local(geom, conductivity)
    R_e = torch.einsum("eab,eb->ea", K_e, T_e)
    if F_e is not None:
        R_e = R_e - F_e
    return R_e


def heat_residual_global(
    geom: BatchedGeometry,
    conductivity: torch.Tensor,
    T: torch.Tensor,
    elem_dofs: torch.Tensor,
    F_e: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Global heat residual from packed nodal temperatures."""
    T_e = gather_element_dofs(T, elem_dofs)
    R_e = heat_residual_local(geom, conductivity, T_e, F_e=F_e)
    return scatter_element_residual(R_e, elem_dofs, T.shape[0])


def elasticity_stiffness_local(
    geom: BatchedGeometry,
    E: torch.Tensor,
    nu: float,
    mode: str = "plane_strain",
) -> torch.Tensor:
    """Element elasticity stiffness matrices, shape ``(E, nen*dim, nen*dim)``."""
    C4 = elasticity_tensor_4th_torch(
        E, nu, geom.dim, mode=mode, dtype=geom.G.dtype, device=geom.G.device
    )
    k_dofs = geom.nen * geom.dim
    K_tensor = torch.einsum(
        "q,eq,eqaj,ijkl,eqbl->eaibk", geom.W_hat, geom.detJ, geom.G, C4, geom.G
    )
    return K_tensor.reshape(geom.E, k_dofs, k_dofs)


def elasticity_residual_local(
    geom: BatchedGeometry,
    E: torch.Tensor,
    nu: float,
    u_e: torch.Tensor,
    F_e: Optional[torch.Tensor] = None,
    mode: str = "plane_strain",
) -> torch.Tensor:
    """Local elasticity residual ``K_e u_e - F_e``, shape ``(E, nen*dim)``."""
    K_e = elasticity_stiffness_local(geom, E, nu, mode=mode)
    R_e = torch.einsum("eab,eb->ea", K_e, u_e)
    if F_e is not None:
        R_e = R_e - F_e
    return R_e


def elasticity_residual_global(
    geom: BatchedGeometry,
    E: torch.Tensor,
    nu: float,
    U: torch.Tensor,
    elem_dofs: torch.Tensor,
    F_e: Optional[torch.Tensor] = None,
    mode: str = "plane_strain",
) -> torch.Tensor:
    """Global elasticity residual from packed nodal displacements."""
    u_e = gather_element_dofs(U, elem_dofs)
    R_e = elasticity_residual_local(geom, E, nu, u_e, F_e=F_e, mode=mode)
    return scatter_element_residual(R_e, elem_dofs, U.shape[0])


def _reshape_u_mat(geom: BatchedGeometry, u_e: torch.Tensor) -> torch.Tensor:
    """Normalize element displacements to ``(E, nen, dim)``."""
    if u_e.ndim == 2:
        return u_e.reshape(geom.E, geom.nen, geom.dim)
    return u_e


def neohookean_residual_local(
    geom: BatchedGeometry,
    u_e: torch.Tensor,
    lambda_: float,
    mu: float,
    F_e: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Local Neo-Hookean residual ``R_int - F_ext``.

    ``u_e`` has shape ``(E, nen, dim)`` or flat ``(E, nen*dim)``.
    """
    u_mat = _reshape_u_mat(geom, u_e)

    # H[e,q,i,j] = sum_a u[e,a,i] * G[e,q,a,j]
    H = torch.einsum("eai,eqaj->eqij", u_mat, geom.G)
    eye = torch.eye(geom.dim, dtype=geom.G.dtype, device=geom.G.device)
    F = eye + H
    P = first_piola_torch(F, lambda_, mu)

    # R_int[e,a,i] = sum_{q,j} P[e,q,i,j] G[e,q,a,j] detJ[e,q] W[q]
    R_int = torch.einsum("q,eq,eqij,eqaj->eai", geom.W_hat, geom.detJ, P, geom.G)
    R_e = R_int.reshape(geom.E, geom.nen * geom.dim)
    if F_e is not None:
        R_e = R_e - F_e
    return R_e


def neohookean_stiffness_local(
    geom: BatchedGeometry,
    u_e: torch.Tensor,
    lambda_: float,
    mu: float,
) -> torch.Tensor:
    """Element Neo-Hookean analytic tangent, shape ``(E, nen*dim, nen*dim)``."""
    u_mat = _reshape_u_mat(geom, u_e)
    H = torch.einsum("eai,eqaj->eqij", u_mat, geom.G)
    eye = torch.eye(geom.dim, dtype=geom.G.dtype, device=geom.G.device)
    F = eye + H
    _, C4 = neohookean_stress_tangent_torch(F, lambda_, mu)
    k_dofs = geom.nen * geom.dim
    K_tensor = torch.einsum(
        "q,eq,eqaj,eqijkl,eqbl->eaibk",
        geom.W_hat,
        geom.detJ,
        geom.G,
        C4,
        geom.G,
    )
    return K_tensor.reshape(geom.E, k_dofs, k_dofs)


def neohookean_system_local(
    geom: BatchedGeometry,
    u_e: torch.Tensor,
    lambda_: float,
    mu: float,
    F_e: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused local Neo-Hookean residual and analytic tangent.

    Returns:
        R_e: shape ``(E, nen*dim)``
        K_e: shape ``(E, nen*dim, nen*dim)``
    """
    u_mat = _reshape_u_mat(geom, u_e)
    H = torch.einsum("eai,eqaj->eqij", u_mat, geom.G)
    eye = torch.eye(geom.dim, dtype=geom.G.dtype, device=geom.G.device)
    F = eye + H
    P, C4 = neohookean_stress_tangent_torch(F, lambda_, mu)

    R_int = torch.einsum("q,eq,eqij,eqaj->eai", geom.W_hat, geom.detJ, P, geom.G)
    R_e = R_int.reshape(geom.E, geom.nen * geom.dim)
    if F_e is not None:
        R_e = R_e - F_e

    k_dofs = geom.nen * geom.dim
    K_tensor = torch.einsum(
        "q,eq,eqaj,eqijkl,eqbl->eaibk",
        geom.W_hat,
        geom.detJ,
        geom.G,
        C4,
        geom.G,
    )
    K_e = K_tensor.reshape(geom.E, k_dofs, k_dofs)
    return R_e, K_e


def neohookean_residual_global(
    geom: BatchedGeometry,
    U: torch.Tensor,
    elem_dofs: torch.Tensor,
    lambda_: float,
    mu: float,
    F_e: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Global Neo-Hookean residual from packed nodal displacements."""
    u_e = gather_element_dofs(U, elem_dofs)
    R_e = neohookean_residual_local(geom, u_e, lambda_, mu, F_e=F_e)
    return scatter_element_residual(R_e, elem_dofs, U.shape[0])


def scatter_stiffness_global(
    K_local: torch.Tensor,
    routing,
) -> torch.Tensor:
    """Reduce local stiffnesses to CSR nonzero values via ``S_mat`` SpMM."""
    m_K = K_local.reshape(-1, 1)
    return torch.sparse.mm(routing.S_mat, m_K).squeeze(1)


def scatter_residual_global(
    R_local: torch.Tensor,
    routing,
) -> torch.Tensor:
    """Reduce local residuals to a global vector via ``S_vec`` SpMM."""
    m_R = R_local.reshape(-1, 1)
    return torch.sparse.mm(routing.S_vec, m_R).squeeze(1)


def neohookean_system_global(
    geom: BatchedGeometry,
    U: torch.Tensor,
    elem_dofs: torch.Tensor,
    routing,
    lambda_: float,
    mu: float,
    F_e: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global Neo-Hookean residual and CSR nonzero tangent values.

    Returns:
        R: shape ``(n_dofs,)``
        K_values: shape ``(N_nnz,)`` aligned with ``routing`` CSR structure
    """
    u_e = gather_element_dofs(U, elem_dofs)
    R_e, K_e = neohookean_system_local(geom, u_e, lambda_, mu, F_e=F_e)
    R = scatter_residual_global(R_e, routing)
    K_values = scatter_stiffness_global(K_e, routing)
    return R, K_values
