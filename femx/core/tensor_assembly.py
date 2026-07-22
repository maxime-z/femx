import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple
from femx.backends.numpy_backend import ndarray
from femx.core.mesh import Mesh
from femx.core.dofs import DofMap
from femx.core.routing import RoutingData, build_routing_matrices
from femx.core.tensor_geometry import BatchedGeometry, evaluate_batched_geometry
from femx.formulations.heat import HeatConductionFormulation
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.formulations.thermoelasticity import LinearThermoelasticityFormulation

def compute_batch_map_unified(
    mesh: Mesh,
    formulation,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified Stage I Batch-Map Engine using true physical continuum mechanics tensor orders.
    Evaluates element stiffness matrices across any PDE formulation (heat, elasticity, coupled multi-physics).
    """
    # 1. Evaluate physics-independent geometry
    geom = evaluate_batched_geometry(mesh, device=device, dtype=dtype)
    
    # 2. Get physical material tensors from formulation
    tensors = formulation.get_physical_tensors(geom, device=device, dtype=dtype)
    
    # 3. Unified Contraction based on tensor rank order
    if isinstance(formulation, HeatConductionFormulation):
        K_2nd, M_0th, f_body = tensors
        # 2nd-order physical tensor contraction (Scalar Heat/Diffusion)
        K_local = torch.einsum('q,eq,eqai,eqij,eqbj->eab', geom.W_hat, geom.detJ, geom.G, K_2nd, geom.G)
        F_local = torch.zeros((geom.E, geom.nen), dtype=dtype, device=device)
    elif isinstance(formulation, LinearElasticityFormulation):
        C_4th, M_0th, f_body = tensors
        k_dofs = geom.nen * geom.dim
        # 4th-order physical tensor contraction (Vector Elasticity/Mechanics)
        K_tensor = torch.einsum('q,eq,eqaj,eqijkl,eqbl->eaibk', geom.W_hat, geom.detJ, geom.G, C_4th, geom.G)
        K_local = K_tensor.reshape(geom.E, k_dofs, k_dofs)
        F_local = torch.zeros((geom.E, k_dofs), dtype=dtype, device=device)
    elif isinstance(formulation, LinearThermoelasticityFormulation):
        C_4th, M_th, K_th, M_0th, T0 = tensors
        u_dofs = geom.nen * geom.dim
        T_dofs = geom.nen
        total_dofs = u_dofs + T_dofs

        # 1. K_uu (4th-order mechanical contraction)
        K_uu_tensor = torch.einsum('q,eq,eqaj,eqijkl,eqbl->eaibk', geom.W_hat, geom.detJ, geom.G, C_4th, geom.G)
        K_uu = K_uu_tensor.reshape(geom.E, u_dofs, u_dofs)

        # 2. K_uT (2nd-order thermal coupling contraction: G_aj * M_th_ij * N_b)
        K_uT_tensor = - torch.einsum('q,eq,eqaj,eqij,qb->eaib', geom.W_hat, geom.detJ, geom.G, M_th, geom.B_hat)
        K_uT = K_uT_tensor.reshape(geom.E, u_dofs, T_dofs)

        # 3. K_TT (2nd-order thermal conductivity contraction)
        K_TT = torch.einsum('q,eq,eqai,eqij,eqbj->eab', geom.W_hat, geom.detJ, geom.G, K_th, geom.G)

        # Assemble block matrix
        K_local = torch.zeros((geom.E, total_dofs, total_dofs), dtype=dtype, device=device)
        K_local[:, 0:u_dofs, 0:u_dofs] = K_uu
        K_local[:, 0:u_dofs, u_dofs:total_dofs] = K_uT
        K_local[:, u_dofs:total_dofs, u_dofs:total_dofs] = K_TT

        F_local = torch.zeros((geom.E, total_dofs), dtype=dtype, device=device)
        if abs(T0) > 1e-12:
            T0_vec = torch.full((geom.E, T_dofs), fill_value=T0, dtype=dtype, device=device)
            F_local[:, 0:u_dofs] -= torch.einsum('eij,ej->ei', K_uT, T0_vec)
    else:
        raise TypeError(f"Unsupported formulation: {type(formulation)}")
        
    return K_local, F_local

# Maintain aliases for backwards compatibility
def compute_batch_map_heat(mesh: Mesh, formulation: HeatConductionFormulation, device: str = "cpu", dtype: torch.dtype = torch.float64):
    return compute_batch_map_unified(mesh, formulation, device=device, dtype=dtype)

def compute_batch_map_elasticity(mesh: Mesh, formulation: LinearElasticityFormulation, device: str = "cpu", dtype: torch.dtype = torch.float64):
    return compute_batch_map_unified(mesh, formulation, device=device, dtype=dtype)

def assemble_system_tensor(
    dof_map: DofMap,
    formulation,
    field_name: str = None,
    routing: RoutingData = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64
) -> Tuple[sp.csr_matrix, sp.csr_matrix, ndarray, torch.Tensor, torch.Tensor]:
    """
    Full TensorGalerkin Monolithic Assembly using Unified Batch-Map and SpMM Sparse-Reduce.
    """
    mesh = dof_map.geometry
    
    if field_name is None:
        if hasattr(formulation, "field_names"):
            field_name = formulation.field_names
        else:
            raise ValueError("Must specify field_name or formulation must specify field_names")
            
    # 1. Precompute or retrieve RoutingData for Stage II
    if routing is None:
        routing = build_routing_matrices(mesh, dof_map, field_name, device=device, dtype=dtype)
        
    # 2. Stage I: Unified Batch-Map
    K_local, F_local = compute_batch_map_unified(mesh, formulation, device=device, dtype=dtype)
        
    # 3. Stage II: Unified Sparse-Reduce via SpMM
    m_K = K_local.reshape(-1, 1) # (E * k^2, 1)
    m_F = F_local.reshape(-1, 1) # (E * k, 1)
    
    v_K = torch.sparse.mm(routing.S_mat, m_K).squeeze(1) # (N_nnz,)
    f_tensor = torch.sparse.mm(routing.S_vec, m_F).squeeze(1) # (N_dofs,)
    
    row_ptrs = routing.crow_indices.cpu().numpy()
    col_idxs = routing.col_indices.cpu().numpy()
    values_K = v_K.cpu().detach().numpy()
    
    K_csr = sp.csr_matrix((values_K, col_idxs, row_ptrs), shape=(routing.n_dofs, routing.n_dofs))
    M_csr = sp.csr_matrix((routing.n_dofs, routing.n_dofs))
    f_np = f_tensor.cpu().detach().numpy()
    
    return K_csr, M_csr, f_np, K_local, F_local
