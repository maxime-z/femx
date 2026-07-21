import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, Dict, Union
from femx.backends.numpy_backend import ndarray
from femx.core.mesh import Mesh
from femx.core.dofs import DofMap
from femx.core.quadrature import get_quadrature_2d, get_quadrature_triangle
from femx.core.routing import RoutingData, build_routing_matrices
from femx.formulations.heat import HeatConductionFormulation
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.basis.lagrange import evaluate_q1_shape_functions, evaluate_q1_shape_derivatives_ref
from femx.basis.triangle import evaluate_t1_shape_functions, evaluate_t1_shape_derivatives_ref

def compute_batch_map_heat(
    mesh: Mesh,
    formulation: HeatConductionFormulation,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stage I (Batch-Map) for Heat Conduction PDE.
    """
    coords = torch.tensor(mesh.coords, dtype=dtype, device=device) # (N, 2)
    cells = torch.tensor(mesh.cells, dtype=torch.int64, device=device)     # (E, nen)
    E = mesh.n_elements
    nen = cells.shape[1] # 4 for Q1, 3 for T1
    
    # 1. Gather element coordinates X: shape (E, nen, 2)
    X = coords[cells]
    
    # 2. Get quadrature rules and reference shape function data
    if nen == 4:
        pts_np, wts_np = get_quadrature_2d(2, 2)
        Q = len(pts_np)
        B_hat_list = [evaluate_q1_shape_functions(pt[0], pt[1]) for pt in pts_np]
        dB_hat_list = [evaluate_q1_shape_derivatives_ref(pt[0], pt[1]) for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, 4, 2)
    else:
        pts_np, wts_np = get_quadrature_triangle(1)
        Q = len(pts_np)
        B_hat_list = [evaluate_t1_shape_functions(pt[0], pt[1]) for pt in pts_np]
        dB_hat_list = [evaluate_t1_shape_derivatives_ref() for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, 3, 2)
        
    B_hat = torch.tensor(np.array(B_hat_list), dtype=dtype, device=device) # (Q, nen)
    W_hat = torch.tensor(wts_np, dtype=dtype, device=device)               # (Q,)
    
    k_coeff = formulation.material.get_property("K")
    
    J = torch.einsum('eac,qad->eqcd', X, dB_hat_q)
    detJ = torch.linalg.det(J)
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)
    
    G = torch.einsum('eqcd,qad->eqac', J_inv_T, dB_hat_q)
    K_local = k_coeff * torch.einsum('q,eq,eqac,eqbc->eab', W_hat, detJ, G, G)
    F_local = torch.zeros((E, nen), dtype=dtype, device=device)
    
    return K_local, F_local

def compute_batch_map_elasticity(
    mesh: Mesh,
    formulation: LinearElasticityFormulation,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stage I (Batch-Map) for 2D Linear Elasticity PDE.
    """
    coords = torch.tensor(mesh.coords, dtype=dtype, device=device) # (N, 2)
    cells = torch.tensor(mesh.cells, dtype=torch.int64, device=device)     # (E, nen)
    E = mesh.n_elements
    nen = cells.shape[1]
    k_dofs = 2 * nen
    
    X = coords[cells] # (E, nen, 2)
    
    C4_np = formulation.material.get_elasticity_tensor_4th(mode=formulation.mode, dim=2)
    C4 = torch.tensor(C4_np, dtype=dtype, device=device) # (2, 2, 2, 2)
    
    if nen == 4:
        pts_np, wts_np = get_quadrature_2d(2, 2)
        Q = len(pts_np)
        dB_hat_list = [evaluate_q1_shape_derivatives_ref(pt[0], pt[1]) for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, 4, 2)
    else:
        pts_np, wts_np = get_quadrature_triangle(1)
        Q = len(pts_np)
        dB_hat_list = [evaluate_t1_shape_derivatives_ref() for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, 3, 2)
        
    W_hat = torch.tensor(wts_np, dtype=dtype, device=device) # (Q,)
    
    J = torch.einsum('eac,qad->eqcd', X, dB_hat_q)
    detJ = torch.linalg.det(J)
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)
    
    # Physical Shape Gradients G -> (E, Q, nen, 2) where G[e, q, a, j] = dN_a / dx_j
    G = torch.einsum('eqcd,qad->eqac', J_inv_T, dB_hat_q)
    
    # Stage I True 4th-Order Physical Tensor Contraction:
    # K_tensor[e, a, i, b, k] = sum_q ( W[q] * detJ[e, q] * sum_{j, l} ( G[e, q, a, j] * C4[i, j, k, l] * G[e, q, b, l] ) )
    K_tensor = torch.einsum('q,eq,eqaj,ijkl,eqbl->eaibk', W_hat, detJ, G, C4, G)
    
    # Reshape (E, nen, 2, nen, 2) -> (E, 2*nen, 2*nen)
    K_local = K_tensor.reshape(E, k_dofs, k_dofs)
    F_local = torch.zeros((E, k_dofs), dtype=dtype, device=device)
    
    return K_local, F_local

def assemble_system_tensor(
    dof_map: DofMap,
    formulation,
    field_name: str,
    routing: RoutingData = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64
) -> Tuple[sp.csr_matrix, sp.csr_matrix, ndarray, torch.Tensor, torch.Tensor]:
    """
    Full TensorGalerkin Monolithic Assembly (Stage I Batch-Map + Stage II SpMM Sparse-Reduce).
    """
    mesh = dof_map.geometry
    
    # 1. Precompute or retrieve RoutingData for Stage II
    if routing is None:
        routing = build_routing_matrices(mesh, dof_map, field_name, device=device, dtype=dtype)
        
    # 2. Stage I: Batch-Map (Fully Tensorized Physics)
    if isinstance(formulation, HeatConductionFormulation):
        K_local, F_local = compute_batch_map_heat(mesh, formulation, device=device, dtype=dtype)
    elif isinstance(formulation, LinearElasticityFormulation):
        K_local, F_local = compute_batch_map_elasticity(mesh, formulation, device=device, dtype=dtype)
    else:
        raise TypeError(f"TensorGalerkin currently supports HeatConduction and LinearElasticity formulations.")
        
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
