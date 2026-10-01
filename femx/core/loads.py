"""Sample a body load at one physical Gauss point.

Accepted forms, for a source with ``n_comp`` components:

- ``None``: zero
- scalar: only when ``n_comp == 1`` (a heat source)
- array of shape ``(n_comp,)``: the same value on the whole mesh
- callable ``f(xyz)`` with ``xyz`` of shape ``(dim,)``: returns a scalar or shape ``(n_comp,)``
- array of shape ``(n_elem, n_comp)``: constant on each element
- array of shape ``(n_elem, n_q, n_comp)``: one value per element and quadrature point,
  in the same order as the quadrature points passed to the element routine

Coupled formulations pass one of these per field, typically
``{"u": force, "T": heat_source}``. This function samples a single field.
"""
from typing import Callable, Optional, Union
import numpy as np
from femx.backends.numpy_backend import ndarray, zeros


BodyLoad = Optional[Union[float, ndarray, Callable]]


def sample_body_load(
    body_load: BodyLoad,
    xyz: ndarray,
    *,
    elem_idx: int,
    q_idx: int,
    n_comp: int,
) -> ndarray:
    """Return the source at one Gauss point, shape ``(n_comp,)``."""
    if body_load is None:
        return zeros(n_comp)

    if callable(body_load):
        value = body_load(np.asarray(xyz, dtype=float))
        return _as_components(value, n_comp, "callable body_load")

    arr = np.asarray(body_load, dtype=float)
    if arr.ndim == 0:
        if n_comp != 1:
            raise ValueError(
                f"A scalar body_load is valid only for a 1-component source, got n_comp={n_comp}."
            )
        return np.full(n_comp, float(arr))

    if arr.shape == (n_comp,):
        return arr.copy()

    if arr.ndim == 2 and arr.shape[1] == n_comp:
        if elem_idx < 0 or elem_idx >= arr.shape[0]:
            raise IndexError(
                f"body_load elem_idx={elem_idx} is outside the element axis of shape {arr.shape}."
            )
        return arr[elem_idx].copy()

    if arr.ndim == 3 and arr.shape[2] == n_comp:
        if elem_idx < 0 or elem_idx >= arr.shape[0] or q_idx < 0 or q_idx >= arr.shape[1]:
            raise IndexError(
                f"body_load index (elem={elem_idx}, q={q_idx}) is outside shape {arr.shape}."
            )
        return arr[elem_idx, q_idx].copy()

    raise ValueError(
        f"body_load shape {arr.shape} does not match n_comp={n_comp}. "
        f"Expected {(n_comp,)}, (n_elem, {n_comp}), or (n_elem, n_q, {n_comp})."
    )


def sample_body_load_batch(body_load: BodyLoad, xyz: ndarray, n_comp: int) -> ndarray:
    """Sample one field at every Gauss point.

    ``xyz`` has shape ``(n_elem, n_q, dim)``. The result has shape
    ``(n_elem, n_q, n_comp)`` and follows the same forms as ``sample_body_load``.
    """
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 3:
        raise ValueError(f"xyz must have shape (n_elem, n_q, dim), got {points.shape}.")
    n_elem, n_q, _ = points.shape

    if body_load is None:
        return np.zeros((n_elem, n_q, n_comp))

    if callable(body_load):
        out = np.empty((n_elem, n_q, n_comp))
        for elem_idx in range(n_elem):
            for q_idx in range(n_q):
                out[elem_idx, q_idx] = sample_body_load(
                    body_load, points[elem_idx, q_idx],
                    elem_idx=elem_idx, q_idx=q_idx, n_comp=n_comp,
                )
        return out

    arr = np.asarray(body_load, dtype=float)
    if arr.ndim == 0:
        if n_comp != 1:
            raise ValueError(
                f"A scalar body_load is valid only for a 1-component source, got n_comp={n_comp}."
            )
        return np.full((n_elem, n_q, n_comp), float(arr))

    if arr.shape == (n_comp,):
        return np.broadcast_to(arr, (n_elem, n_q, n_comp)).copy()

    if arr.ndim == 2 and arr.shape == (n_elem, n_comp):
        return np.broadcast_to(arr[:, None, :], (n_elem, n_q, n_comp)).copy()

    if arr.shape == (n_elem, n_q, n_comp):
        return arr.copy()

    raise ValueError(
        f"body_load shape {arr.shape} does not match mesh ({n_elem}, {n_q}) and n_comp={n_comp}."
    )


def split_coupled_body_load(body_load):
    """Split a thermoelastic load into mechanical and thermal sources."""
    if body_load is None:
        return None, None
    if isinstance(body_load, dict):
        return body_load.get("u"), body_load.get("T")
    raise TypeError("Thermoelastic body_load must be None or a dict {'u': ..., 'T': ...}.")


def _as_components(value, n_comp: int, source: str) -> ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        if n_comp != 1:
            raise ValueError(f"{source} returned a scalar, but n_comp={n_comp}.")
        return np.full(n_comp, float(arr))
    if arr.shape != (n_comp,):
        raise ValueError(f"{source} returned shape {arr.shape}, expected {(n_comp,)}.")
    return arr.copy()
