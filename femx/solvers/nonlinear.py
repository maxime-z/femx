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
    Commits Gauss-point history only after a successful residual check.
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
        body_load: Optional[ndarray] = None,
        neumann_load: Optional[ndarray] = None
    ) -> Tuple[State, List[float]]:
        """
        Perform Newton-Raphson iteration until convergence.
        """
        U = state.pack_vector(dof_map)

        for dof, val in dirichlet_bcs.items():
            U[dof] = val
        state.unpack_vector(U, dof_map)

        all_dofs = np.arange(dof_map.n_dofs)
        constrained_dofs = np.fromiter(dirichlet_bcs.keys(), dtype=np.intp) if dirichlet_bcs else np.array([], dtype=np.intp)
        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)

        residual_history = []
        r0_norm = None

        for k in range(self.max_iter):
            state.copy_committed_to_trial()
            K, R = assemble_nonlinear_system(dof_map, formulation, state, body_load=body_load)
            if neumann_load is not None:
                R -= neumann_load

            R_free = R[free_dofs]
            r_norm = float(np.linalg.norm(R_free))
            residual_history.append(r_norm)

            if k == 0:
                r0_norm = r_norm if r_norm > 0 else 1.0

            target_tol = self.rtol * r0_norm + self.atol
            if r_norm <= target_tol or r_norm <= self.atol:
                state.commit_gauss()
                return state, residual_history

            delta_bcs = {dof: 0.0 for dof in constrained_dofs}
            K_eff, neg_R_eff = apply_dirichlet_bcs(K, -R, delta_bcs, preserve_symmetry=False)

            dU = solve_linear(K_eff, neg_R_eff)

            U_committed = U.copy()
            step_length = 1.0
            accepted = False
            for _ in range(5):
                U_trial = U_committed + step_length * dU
                state.restore_nodal(U_trial, dof_map)
                state.copy_committed_to_trial()
                try:
                    K_test, R_test = assemble_nonlinear_system(dof_map, formulation, state, body_load=body_load)
                    if neumann_load is not None:
                        R_test -= neumann_load
                    r_test_norm = float(np.linalg.norm(R_test[free_dofs]))
                    if r_test_norm < r_norm or step_length <= 0.125:
                        U = U_trial
                        accepted = True
                        break
                except ValueError:
                    step_length *= 0.5
                    state.restore_nodal(U_committed, dof_map)
                    continue
                step_length *= 0.5

            if not accepted:
                U = U_committed + step_length * dU
                state.restore_nodal(U, dof_map)

        raise RuntimeError(
            f"Newton-Raphson failed to converge after {self.max_iter} iterations. "
            f"Final residual norm: {residual_history[-1]:.4e}, target: {target_tol:.4e}"
        )


class LoadStepper:
    """
    Incremental pseudo-time load stepper for static nonlinear problems.
    Applies loads/BCs incrementally over n_steps using NewtonSolver.
    Commits Gauss history only after a successful Newton step.
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
        full_body_load: Optional[ndarray] = None,
        full_neumann_load: Optional[ndarray] = None
    ) -> Tuple[State, List[State]]:
        state = initial_state
        state_history = [state]

        for step in range(1, self.n_steps + 1):
            lam = step / float(self.n_steps)

            step_bcs = {dof: val * lam for dof, val in full_dirichlet_bcs.items()}
            step_body = full_body_load * lam if full_body_load is not None else None
            step_neumann = full_neumann_load * lam if full_neumann_load is not None else None

            state, _ = self.newton_solver.solve(
                dof_map, formulation, state, step_bcs, body_load=step_body, neumann_load=step_neumann
            )
            # NewtonSolver already commits on success; failed Newton raises before commit.
            state_history.append(state)

        return state, state_history
