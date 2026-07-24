import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple
from femx.backends.numpy_backend import ndarray
from femx.core.mesh import Mesh
from femx.core.dofs import DofMap
from femx.core.routing import RoutingData, build_routing_matrices
from femx.core.tensor_geometry import BatchedGeometry, evaluate_batched_geometry
def compute_batch_map_unified(
    mesh: Mesh,
    formulation,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64
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
    K_local, M_local, F_local = formulation.compute_batch_map(geom, tensors, device=device, dtype=dtype)
        
    return K_local, M_local, F_local

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
        if hasattr(formulation, "field_names") and formulation.field_names is not None:
            field_name = list(formulation.field_names) if isinstance(formulation.field_names, (tuple, list)) else formulation.field_names
        else:
            raise ValueError("Must specify field_name or formulation must specify field_names")
            
    # 1. Precompute or retrieve RoutingData for Stage II
    if routing is None:
        routing = build_routing_matrices(mesh, dof_map, field_name, device=device, dtype=dtype)
        
    # 2. Stage I: Unified Batch-Map
    K_local, M_local, F_local = compute_batch_map_unified(mesh, formulation, device=device, dtype=dtype)
        
    # 3. Stage II: Unified Sparse-Reduce via SpMM
    m_K = K_local.reshape(-1, 1) # (E * k^2, 1)
    m_M = M_local.reshape(-1, 1)
    m_F = F_local.reshape(-1, 1) # (E * k, 1)
    
    v_K = torch.sparse.mm(routing.S_mat, m_K).squeeze(1) # (N_nnz,)
    v_M = torch.sparse.mm(routing.S_mat, m_M).squeeze(1)
    f_tensor = torch.sparse.mm(routing.S_vec, m_F).squeeze(1) # (N_dofs,)
    
    row_ptrs = routing.crow_indices.cpu().numpy()
    col_idxs = routing.col_indices.cpu().numpy()
    values_K = v_K.cpu().detach().numpy()
    values_M = v_M.cpu().detach().numpy()
    
    K_csr = sp.csr_matrix((values_K, col_idxs, row_ptrs), shape=(routing.n_dofs, routing.n_dofs))
    M_csr = sp.csr_matrix((values_M, col_idxs, row_ptrs), shape=(routing.n_dofs, routing.n_dofs))
    f_np = f_tensor.cpu().detach().numpy()
    
    return K_csr, M_csr, f_np, K_local, F_local
