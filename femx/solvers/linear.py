from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sp
from femx.backends.numpy_backend import ndarray, solve_linear

def apply_dirichlet_bcs(
    K: sp.csr_matrix,
    f: ndarray,
    dirichlet_bcs: Dict[int, float],
    *,
    preserve_symmetry: bool = True,
) -> Tuple[sp.csr_matrix, ndarray]:
    """
    Apply Dirichlet boundary conditions to K u = f.

    Two variants are available:

    1. ``preserve_symmetry=True`` (default, recommended for SPD FEM systems)
       For each constrained DOF ``i`` with value ``g_i``:
         - move known column contribution into the RHS: ``f -= K[:, i] * g_i``
         - zero row ``i`` and column ``i``
         - set ``K[i, i] = 1`` and ``f[i] = g_i``
       This keeps ``K`` symmetric if it was symmetric, so CG / Cholesky / LDLᵀ
       remain valid.

    2. ``preserve_symmetry=False`` (row replacement only)
       Only zero row ``i``, set ``K[i, i] = 1`` and ``f[i] = g_i``.
       Simpler and slightly cheaper to build, but breaks symmetry. Fine for
       general direct solvers such as ``spsolve`` (UMFPACK / SuperLU).

    See also:
        ``docs/dirichlet_bc_reduced_system_proof.md`` — proof that Variant B
        recovers the classical reduced free-DOF system.
        ``docs/choosing_dirichlet_bc_imposition.md`` — when to choose each variant.

    Args:
        K: Global stiffness matrix (csr_matrix)
        f: Global load vector (ndarray)
        dirichlet_bcs: Mapping global DOF index -> prescribed value
        preserve_symmetry: Whether to use the symmetry-preserving variant
    Returns:
        K_constrained, f_constrained
    """
    if not dirichlet_bcs:
        return K.tocsr(), f.copy()

    f_constrained = f.copy()
    constrained = np.fromiter(dirichlet_bcs.keys(), dtype=np.intp)
    values = np.fromiter(dirichlet_bcs.values(), dtype=float)

    if preserve_symmetry:
        # Use original columns before structural edits.
        # f -= sum_i K[:, i] * g_i  (known Dirichlet contribution)
        K_csc = K.tocsc()
        for dof, val in zip(constrained, values):
            if val != 0.0:
                f_constrained -= np.asarray(K_csc[:, dof].todense()).ravel() * val

    # Row (and optionally column) edits are cheapest in LIL.
    K_lil = K.tolil()
    for dof, val in zip(constrained, values):
        K_lil[dof, :] = 0.0
        if preserve_symmetry:
            K_lil[:, dof] = 0.0
        K_lil[dof, dof] = 1.0
        f_constrained[dof] = val

    return K_lil.tocsr(), f_constrained


def solve_system(
    K: sp.csr_matrix,
    f: ndarray,
    dirichlet_bcs: Dict[int, float],
    *,
    preserve_symmetry: bool = True,
) -> ndarray:
    """
    Apply Dirichlet boundary conditions and solve K * u = f.

    Args:
        K: Global stiffness matrix
        f: Global load vector
        dirichlet_bcs: Prescribed DOF constraint dict
        preserve_symmetry: Passed to ``apply_dirichlet_bcs`` (see its docstring)
    Returns:
        u: Global displacement/temperature solution vector (ndarray)
    """
    K_constrained, f_constrained = apply_dirichlet_bcs(
        K, f, dirichlet_bcs, preserve_symmetry=preserve_symmetry
    )
    return solve_linear(K_constrained, f_constrained)
