from typing import Dict, Tuple, List, Optional
import numpy as np
import scipy.sparse as sp
from femx.backends.numpy_backend import ndarray, solve_linear
from femx.core.dofs import DofMap
from femx.core.state import State
from femx.core.assembly import assemble_nonlinear_system
from femx.formulations.base import Formulation
from femx.solvers.linear import apply_dirichlet_bcs

class NewtonSolver:
    """
    Newton-Raphson solver for nonlinear static finite element systems R(U) = 0.
    """
    def __init__(self, rtol: float = 1e-6, atol: float = 1e-8, max_iter: int = 25):
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter

    def solve(
        self,
        dof_map: DofMap,
        formulation: Formulation,
        state: State,
        dirichlet_bcs: Dict[int, float],
        body_load: Optional[ndarray] = None
    ) -> Tuple[State, List[float]]:
        """
        Perform Newton-Raphson iteration until convergence.
        
        Args:
            dof_map: DofMap for global system
            formulation: Formulation with compute_element_residual_and_tangent
            state: Initial guess State
            dirichlet_bcs: Map of global DOF index -> prescribed value
            body_load: Body force array
            
        Returns:
            state: Converged state
            residual_history: List of free residual norms per iteration
        """
        # Pack current state into vector U
        U = state.pack_vector(dof_map)
        
        # Apply initial Dirichlet boundary conditions to U
        for dof, val in dirichlet_bcs.items():
            U[dof] = val
        state.unpack_vector(U, dof_map)
        
        all_dofs = np.arange(dof_map.n_dofs)
        constrained_dofs = np.fromiter(dirichlet_bcs.keys(), dtype=np.intp) if dirichlet_bcs else np.array([], dtype=np.intp)
        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
        
        residual_history = []
        r0_norm = None
        
        for k in range(self.max_iter):
            # Assemble tangent matrix K and residual vector R
            K, R = assemble_nonlinear_system(dof_map, formulation, state, body_load=body_load)
            
            # Compute residual norm on free DOFs
            R_free = R[free_dofs]
            r_norm = float(np.linalg.norm(R_free))
            residual_history.append(r_norm)
            
            if k == 0:
                r0_norm = r_norm
                
            # Convergence check
            target_tol = self.rtol * r0_norm + self.atol
            if r_norm <= target_tol or r_norm <= self.atol:
                return state, residual_history
                
            # Set up increment solve K * dU = -R
            # For Dirichlet DOFs, we want dU = 0, so RHS for dU is 0.0
            delta_bcs = {dof: 0.0 for dof in constrained_dofs}
            
            # Apply zero-increment BCs to linear system (K, -R)
            K_eff, neg_R_eff = apply_dirichlet_bcs(K, -R, delta_bcs, preserve_symmetry=False)
            
            # Solve for displacement increment dU
            dU = solve_linear(K_eff, neg_R_eff)
            
            # Update solution U and state
            U += dU
            state.unpack_vector(U, dof_map)
            
        raise RuntimeError(
            f"Newton-Raphson failed to converge after {self.max_iter} iterations. "
            f"Final residual norm: {residual_history[-1]:.4e}, target: {target_tol:.4e}"
        )


class LoadStepper:
    """
    Incremental pseudo-time load stepper for static nonlinear problems.
    Applies loads/BCs incrementally over n_steps using NewtonSolver.
    """
    def __init__(self, n_steps: int = 10, newton_solver: Optional[NewtonSolver] = None):
        self.n_steps = n_steps
        self.newton_solver = newton_solver or NewtonSolver()

    def solve(
        self,
        dof_map: DofMap,
        formulation: Formulation,
        initial_state: State,
        full_dirichlet_bcs: Dict[int, float],
        full_body_load: Optional[ndarray] = None
    ) -> Tuple[State, List[State]]:
        """
        Step through pseudo-time steps from lambda = 1/n_steps to 1.0.
        
        Args:
            dof_map: DofMap for global system
            formulation: Hyperelastic formulation
            initial_state: Starting State
            full_dirichlet_bcs: Full magnitude Dirichlet BCs at lambda=1.0
            full_body_load: Full magnitude body force at lambda=1.0
            
        Returns:
            final_state: Converged state at lambda=1.0
            state_history: List of converged states at each step
        """
        state = initial_state
        state_history = [state]
        
        for step in range(1, self.n_steps + 1):
            lam = step / float(self.n_steps)
            
            # Scale boundary conditions and body loads by pseudo-time factor lambda
            step_bcs = {dof: val * lam for dof, val in full_dirichlet_bcs.items()}
            step_load = full_body_load * lam if full_body_load is not None else None
            
            state, _ = self.newton_solver.solve(
                dof_map, formulation, state, step_bcs, body_load=step_load
            )
            state_history.append(state)
            
        return state, state_history
