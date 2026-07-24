import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Tuple, Callable, Optional
from femx.backends.numpy_backend import ndarray, zeros
from femx.core.dofs import DofMap
from femx.solvers.linear import apply_dirichlet_bcs

def solve_transient_thermoelastic(
    K: sp.csr_matrix,
    M: sp.csr_matrix,
    dof_map: DofMap,
    dirichlet_bcs: Dict[int, float],
    force_func: Callable[[float], Tuple[ndarray, ndarray]],
    t_span: Tuple[float, float],
    dt: float,
    T0: float = 293.15,
    alpha_rayleigh: float = 0.0,
    beta_rayleigh: float = 0.0,
    gamma: float = 0.5,
    beta: float = 0.25,
    theta: float = 0.5
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Monolithic Time Integration Solver for Coupled Linear Thermoelasticity.

    Governing Equations:
      M_uu * a + C_uu * v + K_uu * u + K_uT * T = f_u(t)
      M_TT * T_dot + K_TT * T - T0 * K_uT^T * v = f_T(t)

    Time Integration Scheme:
      Newmark-beta scheme for mechanical (u, v, a)
      theta-scheme (Crank-Nicolson / Backward Euler) for thermal (T, T_dot)
    """
    t_start, t_end = t_span
    time_pts = np.arange(t_start, t_end + 1e-12, dt)
    num_steps = len(time_pts)
    
    n_dofs = K.shape[0]
    u_dof_indices = dof_map.get_field_dofs("u")
    T_dof_indices = dof_map.get_field_dofs("T")
    
    n_u_dofs = len(u_dof_indices)
    n_T_dofs = len(T_dof_indices)
    
    # Extract sub-blocks from K and M
    K_csc = K.tocsc()
    M_csc = M.tocsc()
    
    K_uu = K_csc[np.ix_(u_dof_indices, u_dof_indices)].tocsr()
    K_uT = K_csc[np.ix_(u_dof_indices, T_dof_indices)].tocsr()
    K_TT = K_csc[np.ix_(T_dof_indices, T_dof_indices)].tocsr()
    
    M_uu = M_csc[np.ix_(u_dof_indices, u_dof_indices)].tocsr()
    M_TT = M_csc[np.ix_(T_dof_indices, T_dof_indices)].tocsr()
    
    # Rayleigh Damping C_uu = alpha_R * M_uu + beta_R * K_uu
    if alpha_rayleigh != 0.0 or beta_rayleigh != 0.0:
        C_uu = alpha_rayleigh * M_uu + beta_rayleigh * K_uu
    else:
        C_uu = sp.csr_matrix((n_u_dofs, n_u_dofs))
        
    # Thermal coupling dissipation operator C_Tu = - T0 * K_uT^T
    C_Tu = - T0 * K_uT.T.tocsr()
    
    # Newmark integration constants
    c_ma = 1.0 / (beta * dt**2)
    c_mv = 1.0 / (beta * dt)
    c_ma_prev = (1.0 / (2.0 * beta)) - 1.0
    
    c_va = gamma / (beta * dt)
    c_vv = (gamma / beta) - 1.0
    c_va_prev = dt * ((gamma / (2.0 * beta)) - 1.0)
    
    c_T_dot = 1.0 / (theta * dt)
    c_T_prev = (1.0 - theta) / theta

    # Monolithic Effective Stiffness Matrix K_eff
    # K_eff = [ K_uu + c_ma * M_uu + c_va * C_uu ,  K_uT ]
    #         [ c_va * C_Tu                      ,  K_TT + c_T_dot * M_TT ]
    
    K_eff_uu = K_uu + c_ma * M_uu + c_va * C_uu
    K_eff_uT = K_uT
    K_eff_Tu = c_va * C_Tu
    K_eff_TT = K_TT + c_T_dot * M_TT
    
    K_eff = sp.bmat([
        [K_eff_uu, K_eff_uT],
        [K_eff_Tu, K_eff_TT]
    ], format='csr')

    # Apply Dirichlet Boundary Conditions to K_eff
    K_eff_bc, _ = apply_dirichlet_bcs(K_eff, np.zeros(n_dofs), dirichlet_bcs, preserve_symmetry=False)
    
    # Pre-factorize K_eff with SuperLU (splu)
    print("[Transient Solver] Factoring effective global matrix with SuperLU (splu)...")
    solver_lu = spla.splu(K_eff_bc.tocsc())

    # State variables initialization
    u = zeros(n_u_dofs)
    v = zeros(n_u_dofs)
    a = zeros(n_u_dofs)
    
    T = zeros(n_T_dofs)
    T_dot = zeros(n_T_dofs)
    
    u_history = zeros((num_steps, n_u_dofs))
    T_history = zeros((num_steps, n_T_dofs))
    
    # Store initial state (t = 0)
    u_history[0] = u.copy()
    T_history[0] = T.copy()

    # Time integration loop
    for step in range(1, num_steps):
        t = time_pts[step]
        f_u_ext, f_T_ext = force_func(t)
        
        # History vectors for mechanical predictors
        a_hist = c_ma * u + c_mv * v + c_ma_prev * a
        v_hist = c_va * u + c_vv * v + c_va_prev * a
        
        # Right hand side for mechanical block
        R_u = f_u_ext + M_uu @ a_hist + C_uu @ v_hist
        
        # Right hand side for thermal block
        T_hist = c_T_dot * T + c_T_prev * T_dot
        R_T = f_T_ext + M_TT @ T_hist + C_Tu @ v_hist
        
        R_full = np.concatenate([R_u, R_T])
        
        # Apply Dirichlet boundary values to RHS
        for dof, val in dirichlet_bcs.items():
            R_full[dof] = val
            
        # Solve linear system for X_new = [u_new, T_new]
        X_new = solver_lu.solve(R_full)
        
        u_new = X_new[0:n_u_dofs]
        T_new = X_new[n_u_dofs:n_dofs]
        
        # Update state derivatives
        a_new = c_ma * (u_new - u) - c_mv * v - c_ma_prev * a
        v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)
        T_dot_new = c_T_dot * (T_new - T) - c_T_prev * T_dot
        
        # Advance state
        u, v, a = u_new, v_new, a_new
        T, T_dot = T_new, T_dot_new
        
        u_history[step] = u.copy()
        T_history[step] = T.copy()

    return time_pts, u_history, T_history
