import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, Dict, Union, List
from femx.core.mesh import Mesh
from femx.core.dofs import DofMap

class RoutingData:
    """Precomputed topology-aware binary routing matrices for TensorGalerkin Stage II."""
    def __init__(
        self,
        S_vec: torch.Tensor,
        S_mat: torch.Tensor,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        n_dofs: int,
        n_elements: int,
        k: int
    ):
        self.S_vec = S_vec            # PyTorch CSR tensor of shape (n_dofs, E * k)
        self.S_mat = S_mat            # PyTorch CSR tensor of shape (N_nnz, E * k^2)
        self.crow_indices = crow_indices  # CSR row pointers of shape (n_dofs + 1,)
        self.col_indices = col_indices    # CSR column indices of shape (N_nnz,)
        self.n_dofs = n_dofs
        self.n_elements = n_elements
        self.k = k

def build_routing_matrices(mesh: Mesh, dof_map: DofMap, field_name: Union[str, List[str]], device: str = "cpu", dtype: torch.dtype = torch.float64) -> RoutingData:
    """
    Precompute binary topology routing matrices S_vec and S_mat based on mesh connectivity.
    field_name can be a single field name or a list of coupled field names (e.g. ['u', 'T']).
    """
    n_elements = mesh.n_elements
    n_dofs = dof_map.n_dofs
    cells = mesh.cells
    
    if isinstance(field_name, list):
        field_names = field_name
    else:
        field_names = [field_name]
        
    # 1. Determine local DoFs count k for one element
    sample_elem_dofs = dof_map.get_element_dofs_multi(field_names, cells[0])
    k = len(sample_elem_dofs)
    
    # 2. Build S_vec routing: maps (E * k) flattened local force vector entries to N global DoFs
    vec_rows = []
    vec_cols = []
    
    for e in range(n_elements):
        elem_dofs = dof_map.get_element_dofs_multi(field_names, cells[e])
        for a in range(k):
            g_dof = elem_dofs[a]
            flat_idx = e * k + a
            vec_rows.append(g_dof)
            vec_cols.append(flat_idx)
            
    vec_data = np.ones(len(vec_rows), dtype=np.float64)
    S_vec_coo = sp.coo_matrix((vec_data, (vec_rows, vec_cols)), shape=(n_dofs, n_elements * k)).tocsr()
    
    # 3. Build S_mat routing: maps (E * k^2) flattened local stiffness entries to N_nnz global sparse non-zeros
    mat_elem_rows = []
    mat_elem_cols = []
    flat_k_indices = []
    
    for e in range(n_elements):
        elem_dofs = dof_map.get_element_dofs_multi(field_names, cells[e])
        for a in range(k):
            g_row = elem_dofs[a]
            for b in range(k):
                g_col = elem_dofs[b]
                flat_k_idx = e * k * k + a * k + b
                mat_elem_rows.append(g_row)
                mat_elem_cols.append(g_col)
                flat_k_indices.append(flat_k_idx)
                
    mat_elem_rows = np.array(mat_elem_rows)
    mat_elem_cols = np.array(mat_elem_cols)
    flat_k_indices = np.array(flat_k_indices)
    
    # Identify unique (row, col) non-zero positions in global sparse matrix
    unique_pairs, inverse_indices = np.unique(
        np.vstack([mat_elem_rows, mat_elem_cols]).T,
        axis=0,
        return_inverse=True
    )
    
    N_nnz = len(unique_pairs)
    
    # S_mat entries: row = unique pair ID (0 ... N_nnz-1), col = flat_k_idx
    smat_rows = inverse_indices
    smat_cols = flat_k_indices
    smat_data = np.ones(len(smat_rows), dtype=np.float64)
    
    S_mat_coo = sp.coo_matrix((smat_data, (smat_rows, smat_cols)), shape=(N_nnz, n_elements * k * k)).tocsr()
    
    # Build CSR format for global K
    global_coo = sp.coo_matrix(
        (np.ones(N_nnz), (unique_pairs[:, 0], unique_pairs[:, 1])),
        shape=(n_dofs, n_dofs)
    ).tocsr()
    
    crow_indices = torch.tensor(global_coo.indptr, dtype=torch.int64, device=device)
    col_indices = torch.tensor(global_coo.indices, dtype=torch.int64, device=device)
    
    # Convert S_vec and S_mat to PyTorch CSR sparse tensors
    S_vec_torch = torch.sparse_csr_tensor(
        torch.tensor(S_vec_coo.indptr, dtype=torch.int64),
        torch.tensor(S_vec_coo.indices, dtype=torch.int64),
        torch.tensor(S_vec_coo.data, dtype=dtype),
        size=(n_dofs, n_elements * k),
        device=device
    )
    
    S_mat_torch = torch.sparse_csr_tensor(
        torch.tensor(S_mat_coo.indptr, dtype=torch.int64),
        torch.tensor(S_mat_coo.indices, dtype=torch.int64),
        torch.tensor(S_mat_coo.data, dtype=dtype),
        size=(N_nnz, n_elements * k * k),
        device=device
    )
    
    return RoutingData(
        S_vec=S_vec_torch,
        S_mat=S_mat_torch,
        crow_indices=crow_indices,
        col_indices=col_indices,
        n_dofs=n_dofs,
        n_elements=n_elements,
        k=k
    )
