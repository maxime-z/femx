import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Sequence, Union

@dataclass(frozen=True)
class KnotVector:
    knots: np.ndarray

    def __init__(self, knots: Sequence[float]):
        object.__setattr__(self, 'knots', np.array(knots, dtype=np.float64))
        if not np.all(np.diff(self.knots) >= -1e-12):
            raise ValueError("Knot vector must be non-decreasing.")

    def __len__(self):
        return len(self.knots)

    def __getitem__(self, idx):
        return self.knots[idx]
        
    def find_span(self, p: int, u: float) -> int:
        n = len(self.knots) - p - 2
        if u >= self.knots[n + 1]:
            return n
        if u <= self.knots[p]:
            return p
            
        low = p
        high = n + 1
        mid = (low + high) // 2
        while u < self.knots[mid] or u >= self.knots[mid + 1]:
            if u < self.knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        return mid
        
    def find_multiplicity(self, u: float, tol: float = 1e-10) -> int:
        return int(np.sum(np.abs(self.knots - u) <= tol))
        
    def unique_knots(self, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        rounded_knots = np.round(self.knots / tol) * tol
        return np.unique(rounded_knots, return_counts=True)


from dataclasses import dataclass, field
from typing import Tuple, List, Sequence, Union, Dict

@dataclass(frozen=True)
class NurbsPatch:
    degrees: Tuple[int, ...]
    knot_vectors: Tuple[KnotVector, ...]
    control_points: np.ndarray  
    weights: np.ndarray         
    boundaries: Dict[str, np.ndarray] = field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        if 'p_u' in kwargs and 'p_v' in kwargs:
            p_u = kwargs.pop('p_u')
            p_v = kwargs.pop('p_v')
            knots_u = kwargs.pop('knots_u')
            knots_v = kwargs.pop('knots_v')
            
            degrees = (p_u, p_v)
            kv_u = knots_u if isinstance(knots_u, KnotVector) else KnotVector(knots_u)
            kv_v = knots_v if isinstance(knots_v, KnotVector) else KnotVector(knots_v)
            knot_vectors = (kv_u, kv_v)
            
            control_points = kwargs.pop('control_points')
            weights = kwargs.pop('weights')
            boundaries = kwargs.pop('boundaries', {})
        else:
            if len(args) >= 4:
                degrees, knot_vectors, control_points, weights = args[:4]
                boundaries = args[4] if len(args) > 4 else kwargs.pop('boundaries', {})
            else:
                degrees = kwargs.pop('degrees')
                knot_vectors = kwargs.pop('knot_vectors')
                control_points = kwargs.pop('control_points')
                weights = kwargs.pop('weights')
                boundaries = kwargs.pop('boundaries', {})
                
        object.__setattr__(self, 'degrees', degrees)
        object.__setattr__(self, 'knot_vectors', knot_vectors)
        object.__setattr__(self, 'control_points', control_points)
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'boundaries', boundaries)
        
        dim = len(self.degrees)
        if len(self.knot_vectors) != dim:
            raise ValueError("Number of knot vectors must match number of parametric dimensions.")
        if self.weights.ndim != dim:
            raise ValueError("Weights dimension must match parametric dimensions.")
        if self.control_points.ndim != dim + 1:
            raise ValueError("Control points dimension must be (parametric dimensions + 1).")
        
        for i in range(dim):
            n_i = self.control_points.shape[i]
            p_i = self.degrees[i]
            m_i = len(self.knot_vectors[i]) - 1
            if m_i != n_i + p_i:
                raise ValueError(f"Direction {i}: knot vector length ({m_i+1}) must equal control points ({n_i}) + degree ({p_i}) + 1.")
                
    @property
    def parametric_dim(self) -> int:
        return len(self.degrees)
        
    @property
    def physical_dim(self) -> int:
        return self.control_points.shape[-1]

    # Backwards compatibility properties for 2D
    @property
    def p_u(self) -> int:
        return self.degrees[0]

    @property
    def p_v(self) -> int:
        return self.degrees[1]

    @property
    def p_w(self) -> int:
        return self.degrees[2]

    @property
    def knots_u(self) -> np.ndarray:
        return self.knot_vectors[0].knots

    @property
    def knots_v(self) -> np.ndarray:
        return self.knot_vectors[1].knots

    @property
    def knots_w(self) -> np.ndarray:
        return self.knot_vectors[2].knots

    @property
    def n_cp_u(self) -> int:
        return self.control_points.shape[0]

    @property
    def n_cp_v(self) -> int:
        return self.control_points.shape[1]

    @property
    def n_cp_w(self) -> int:
        return self.control_points.shape[2]

    @property
    def n_control_points(self) -> int:
        return int(np.prod(self.control_points.shape[:-1]))

    def get_element_spans(self) -> List[Tuple[int, ...]]:
        if self.parametric_dim == 2:
            spans = []
            for i_u in range(self.p_u, len(self.knots_u) - self.p_u - 1):
                for i_v in range(self.p_v, len(self.knots_v) - self.p_v - 1):
                    if (self.knots_u[i_u + 1] > self.knots_u[i_u] and
                        self.knots_v[i_v + 1] > self.knots_v[i_v]):
                        spans.append((i_u, i_v))
            return spans
        if self.parametric_dim == 3:
            spans = []
            for i_u in range(self.p_u, len(self.knots_u) - self.p_u - 1):
                for i_v in range(self.p_v, len(self.knots_v) - self.p_v - 1):
                    for i_w in range(self.p_w, len(self.knots_w) - self.p_w - 1):
                        if (self.knots_u[i_u + 1] > self.knots_u[i_u] and
                            self.knots_v[i_v + 1] > self.knots_v[i_v] and
                            self.knots_w[i_w + 1] > self.knots_w[i_w]):
                            spans.append((i_u, i_v, i_w))
            return spans
        raise ValueError(f"get_element_spans supports parametric_dim 2 or 3, got {self.parametric_dim}")

    def get_element_control_points(self, span_u: int, span_v: int, span_w: int = None) -> np.ndarray:
        if self.parametric_dim == 2:
            indices = []
            for j in range(self.p_v + 1):
                idx_v = span_v - self.p_v + j
                for i in range(self.p_u + 1):
                    idx_u = span_u - self.p_u + i
                    indices.append(idx_u + idx_v * self.n_cp_u)
            return np.array(indices, dtype=int)
        if self.parametric_dim == 3:
            if span_w is None:
                raise ValueError("span_w is required for a trivariate NURBS patch")
            indices = []
            for k in range(self.p_w + 1):
                idx_w = span_w - self.p_w + k
                for j in range(self.p_v + 1):
                    idx_v = span_v - self.p_v + j
                    for i in range(self.p_u + 1):
                        idx_u = span_u - self.p_u + i
                        indices.append(
                            idx_u + idx_v * self.n_cp_u + idx_w * self.n_cp_u * self.n_cp_v
                        )
            return np.array(indices, dtype=int)
        raise ValueError(f"get_element_control_points supports parametric_dim 2 or 3")

    def flatten_control_points(self) -> np.ndarray:
        """Flatten control points to match ``get_element_control_points`` indices."""
        if self.parametric_dim == 2:
            return self.control_points.transpose(1, 0, 2).reshape((-1, self.physical_dim))
        if self.parametric_dim == 3:
            return self.control_points.transpose(2, 1, 0, 3).reshape((-1, self.physical_dim))
        raise ValueError(f"flatten_control_points supports parametric_dim 2 or 3")
        
    def get_weighted_control_points(self) -> np.ndarray:
        Pw = np.zeros(self.weights.shape + (self.physical_dim + 1,), dtype=np.float64)
        Pw[..., :-1] = self.control_points * self.weights[..., np.newaxis]
        Pw[..., -1] = self.weights
        return Pw
        
    @classmethod
    def from_weighted_control_points(cls, degrees: Tuple[int, ...], knot_vectors: Tuple[KnotVector, ...], Pw: np.ndarray, boundaries: Dict[str, np.ndarray] = None):
        weights = Pw[..., -1]
        safe_weights = np.where(weights == 0, 1.0, weights)
        control_points = Pw[..., :-1] / safe_weights[..., np.newaxis]
        b = boundaries if boundaries is not None else {}
        return cls(degrees=degrees, knot_vectors=knot_vectors, control_points=control_points, weights=weights, boundaries=b)


def insert_knot_curve(p: int, U: KnotVector, Pw: np.ndarray, u: float, r: int = 1) -> Tuple[KnotVector, np.ndarray]:
    n = len(Pw) - 1
    m = n + p + 1
    k = U.find_span(p, u)
    s = U.find_multiplicity(u)
    
    if s + r > p:
        raise ValueError(f"Cannot insert knot {u} {r} times, multiplicity would exceed degree {p}.")
        
    new_m = m + r
    Uq = np.zeros(new_m + 1, dtype=np.float64)
    Uq[:k+1] = U.knots[:k+1]
    Uq[k+1:k+1+r] = u
    Uq[k+1+r:] = U.knots[k+1:]
    
    new_n = n + r
    Qw = np.zeros((new_n + 1, Pw.shape[1]), dtype=np.float64)
    Qw[:k-p+1] = Pw[:k-p+1]
    Qw[k-s+r:] = Pw[k-s:]
    
    Rw = np.zeros((p + 1, Pw.shape[1]), dtype=np.float64)
    Rw[:p-s+1] = Pw[k-p:k-s+1]
    
    for j in range(1, r + 1):
        L = k - p + j
        for i in range(p - j - s + 1):
            alpha = (u - U[L+i]) / (U[i+k+1] - U[L+i])
            Rw[i] = alpha * Rw[i+1] + (1.0 - alpha) * Rw[i]
        Qw[L:L+p-j-s+1] = Rw[:p-j-s+1]
        
    return KnotVector(Uq), Qw

def insert_knot(patch: NurbsPatch, direction: int, u: float, r: int = 1) -> NurbsPatch:
    p = patch.degrees[direction]
    U = patch.knot_vectors[direction]
    Pw = patch.get_weighted_control_points()
    
    Pw_swapped = np.swapaxes(Pw, 0, direction)
    original_shape = list(Pw_swapped.shape)
    num_curves = int(np.prod(original_shape[1:-1]))
    Pw_flat = Pw_swapped.reshape((original_shape[0], num_curves, original_shape[-1]))
    
    Uq, Qw_first = insert_knot_curve(p, U, Pw_flat[:, 0, :], u, r)
    
    Qw_flat = np.zeros((Qw_first.shape[0], num_curves, original_shape[-1]), dtype=np.float64)
    Qw_flat[:, 0, :] = Qw_first
    
    for i in range(1, num_curves):
        _, Qw_i = insert_knot_curve(p, U, Pw_flat[:, i, :], u, r)
        Qw_flat[:, i, :] = Qw_i
        
    new_shape = [Qw_first.shape[0]] + original_shape[1:]
    Qw_swapped = Qw_flat.reshape(new_shape)
    Qw = np.swapaxes(Qw_swapped, 0, direction)
    
    new_knot_vectors = list(patch.knot_vectors)
    new_knot_vectors[direction] = Uq
    
    return NurbsPatch.from_weighted_control_points(patch.degrees, tuple(new_knot_vectors), Qw)


def compute_bezalfs(p: int, t: int) -> np.ndarray:
    ph = p + t
    bezalfs = np.zeros((ph + 1, p + 1))
    for i in range(ph + 1):
        for j in range(max(0, i - t), min(p, i) + 1):
            bezalfs[i, j] = math.comb(p, j) * math.comb(t, i - j) / math.comb(ph, i)
    return bezalfs


def degree_elevate_curve(p: int, U: KnotVector, Pw: np.ndarray, t: int) -> Tuple[KnotVector, np.ndarray]:
    n = len(Pw) - 1
    m = len(U) - 1
    ph = p + t
    
    bezalfs = compute_bezalfs(p, t)
    
    bpts = np.zeros((p + 1, Pw.shape[1]), dtype=np.float64)
    ebpts = np.zeros((ph + 1, Pw.shape[1]), dtype=np.float64)
    Nextbpts = np.zeros((p - 1, Pw.shape[1]), dtype=np.float64) if p > 1 else np.zeros((0, Pw.shape[1]))
    alphas = np.zeros(max(p - 1, 0), dtype=np.float64)
    
    # Each existing knot can gain multiplicity t, and the tail writes ph+1 end knots.
    max_mh = (m + 1) * (t + 1) + ph + 1
    Uh = np.zeros(max_mh, dtype=np.float64)
    Qw = np.zeros((max_mh, Pw.shape[1]), dtype=np.float64)
    
    mh = ph
    kind = ph + 1
    r = -1
    a = p
    b = p + 1
    cind = 1
    ua = U[0]
    Qw[0] = Pw[0]
    for i in range(ph + 1):
        Uh[i] = ua
        
    bpts[:p+1] = Pw[:p+1]
    
    while b < m:
        i = b
        while b < m and U[b] == U[b+1]:
            b += 1
        mul = b - i + 1
        mh += mul + t
        ub = U[b]
        oldr = r
        r = p - mul
        
        if oldr > 0:
            lbz = (oldr + 2) // 2
        else:
            lbz = 1
            
        if r > 0:
            rbz = ph - (r + 1) // 2
        else:
            rbz = ph
            
        if ub == U[-1]:
            rbz = ph - 1
            
        if r > 0:
            numer = ub - ua
            for k in range(p, mul, -1):
                alphas[k-mul-1] = numer / (U[a+k] - ua)
            
            for j in range(1, r + 1):
                save = r - j
                s = mul + j
                for k in range(p, s - 1, -1):
                    bpts[k] = alphas[k-s] * bpts[k] + (1.0 - alphas[k-s]) * bpts[k-1]
                Nextbpts[save] = bpts[p]
                
        for i_idx in range(lbz, ph + 1):
            ebpts[i_idx] = np.zeros_like(Pw[0])
            mpi = min(p, i_idx)
            for j in range(max(0, i_idx - t), mpi + 1):
                ebpts[i_idx] += bezalfs[i_idx, j] * bpts[j]
                
        if oldr > 1:
            first = kind - 2
            last = kind
            den = ub - ua
            bet = (ub - Uh[kind - 1]) / den
            for tr in range(1, oldr):
                i_idx = first
                j = last
                kj = j - kind + 1
                while j - i_idx > tr:
                    if i_idx < cind:
                        alf = (ub - Uh[i_idx]) / (ua - Uh[i_idx])
                        Qw[i_idx] = alf * Qw[i_idx] + (1.0 - alf) * Qw[i_idx - 1]
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - Uh[j - tr]) / den
                        else:
                            gam = bet
                        ebpts[kj] = gam * ebpts[kj] + (1.0 - gam) * ebpts[kj + 1]
                    i_idx += 1
                    j -= 1
                    kj -= 1
                first -= 1
                last += 1
            
        if a != p:
            for i_idx in range(ph - oldr):
                Uh[kind] = ua
                kind += 1
                
        for j in range(lbz, rbz + 1):
            Qw[cind] = ebpts[j]
            cind += 1
            
        if b < m:
            for j in range(r):
                bpts[j] = Nextbpts[j]
            for j in range(r, p + 1):
                bpts[j] = Pw[b - p + j]
            a = b
            b += 1
            ua = ub
            
    for i in range(ph + 1):
        Uh[kind + i] = ub
        
    Qw[cind] = Pw[n]
    
    nh = cind
    Uh = Uh[:kind + ph + 1]
    Qw = Qw[:nh + 1]
    
    return KnotVector(Uh), Qw


def degree_elevate(patch: NurbsPatch, direction: int, t: int) -> NurbsPatch:
    if t == 0:
        return patch
    p = patch.degrees[direction]
    U = patch.knot_vectors[direction]
    Pw = patch.get_weighted_control_points()
    
    Pw_swapped = np.swapaxes(Pw, 0, direction)
    original_shape = list(Pw_swapped.shape)
    num_curves = int(np.prod(original_shape[1:-1]))
    Pw_flat = Pw_swapped.reshape((original_shape[0], num_curves, original_shape[-1]))
    
    Uq, Qw_first = degree_elevate_curve(p, U, Pw_flat[:, 0, :], t)
    
    Qw_flat = np.zeros((Qw_first.shape[0], num_curves, original_shape[-1]), dtype=np.float64)
    Qw_flat[:, 0, :] = Qw_first
    
    for i in range(1, num_curves):
        _, Qw_i = degree_elevate_curve(p, U, Pw_flat[:, i, :], t)
        Qw_flat[:, i, :] = Qw_i
        
    new_shape = [Qw_first.shape[0]] + original_shape[1:]
    Qw_swapped = Qw_flat.reshape(new_shape)
    Qw = np.swapaxes(Qw_swapped, 0, direction)
    
    new_knot_vectors = list(patch.knot_vectors)
    new_knot_vectors[direction] = Uq
    
    new_degrees = list(patch.degrees)
    new_degrees[direction] += t
    
    return NurbsPatch.from_weighted_control_points(tuple(new_degrees), tuple(new_knot_vectors), Qw)


def h_refine(patch: NurbsPatch) -> NurbsPatch:
    """Subdivides all knot spans uniformly."""
    new_patch = patch
    for d in range(patch.parametric_dim):
        U = new_patch.knot_vectors[d]
        unique_knots, _ = U.unique_knots()
        
        knots_to_insert = []
        for i in range(len(unique_knots) - 1):
            mid = 0.5 * (unique_knots[i] + unique_knots[i+1])
            knots_to_insert.append(mid)
            
        for k in knots_to_insert:
            new_patch = insert_knot(new_patch, d, k, r=1)
            
    return new_patch


def p_refine(patch: NurbsPatch, t: int = 1) -> NurbsPatch:
    """Elevates the degree of the patch by t in all parametric directions."""
    new_patch = patch
    for d in range(patch.parametric_dim):
        new_patch = degree_elevate(new_patch, d, t)
    return new_patch


def decompose_to_beziers(patch: NurbsPatch) -> NurbsPatch:
    """Inserts knots until the continuity drops to C^-1, forming Bezier patches."""
    new_patch = patch
    for d in range(patch.parametric_dim):
        p = new_patch.degrees[d]
        unique_knots, _ = new_patch.knot_vectors[d].unique_knots()
        
        inner_knots = unique_knots[1:-1]
        for k in inner_knots:
            s = new_patch.knot_vectors[d].find_multiplicity(k)
            r = p - s
            if r > 0:
                new_patch = insert_knot(new_patch, d, k, r=r)
    return new_patch
