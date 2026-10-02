import heapq
from typing import List, Optional, Set

import numpy as np
from scipy.spatial import cKDTree

from .base import BasisGenerator, normalize_modes

class ZonalBasisGenerator(BasisGenerator):
    """
    Generates a Zonal basis (Identity matrix).
    Each mode corresponds to poking a single actuator.
    """
    
    def generate(self, n_modes: int, normalize: Optional[str] = None) -> np.ndarray:
        """
        Generate Zonal modes: the first ``n_modes`` columns of the identity.

        Args:
            n_modes: Number of modes, at most the number of actuators. Mode
                ``k`` pokes actuator ``k``.
            normalize: ``None``, ``"rms"``, ``"l2"``, ``"peak"`` or ``"pv"``
                (see :func:`aobasis.normalize_modes`). Only ``"rms"`` changes
                the unit pokes.

        Raises:
            ValueError: If ``n_modes`` exceeds the number of actuators.
        """
        self._record_options(n_modes=n_modes, normalize=normalize)
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators)
        self.modes = normalize_modes(np.eye(self.n_actuators)[:, :n_modes], normalize)
        return self.modes


def _build_conflict_graph(positions: np.ndarray, min_distance: float) -> List[Set[int]]:
    n_actuators = positions.shape[0]
    adjacency = [set() for _ in range(n_actuators)]

    if n_actuators == 0 or min_distance <= 0:
        return adjacency

    tree = cKDTree(positions)
    search_radius = np.nextafter(min_distance, 0.0)
    for first, second in tree.query_pairs(r=search_radius, output_type="ndarray"):
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))

    return adjacency


def _dsatur_coloring(adjacency: List[Set[int]]) -> np.ndarray:
    """Greedy DSATUR colouring with a lazy max-heap, O((V + E) log V).

    Picks the uncoloured vertex with the most distinct neighbour colours,
    then the highest degree, then the lowest index, and gives it the
    smallest colour its neighbours do not use.
    """
    n_vertices = len(adjacency)
    colors = np.full(n_vertices, -1, dtype=int)
    neighbor_colors = [set() for _ in range(n_vertices)]
    degrees = [len(neighbors) for neighbors in adjacency]
    heap = [(0, -degrees[v], v) for v in range(n_vertices)]
    heapq.heapify(heap)

    while heap:
        neg_saturation, _, vertex = heapq.heappop(heap)
        if colors[vertex] >= 0 or -neg_saturation != len(neighbor_colors[vertex]):
            continue  # already coloured, or a stale entry
        used = neighbor_colors[vertex]
        color = 0
        while color in used:
            color += 1
        colors[vertex] = color
        for neighbor in adjacency[vertex]:
            if colors[neighbor] < 0 and color not in neighbor_colors[neighbor]:
                neighbor_colors[neighbor].add(color)
                heapq.heappush(heap, (-len(neighbor_colors[neighbor]), -degrees[neighbor], neighbor))

    return colors


def _renumber_colors(colors: np.ndarray) -> np.ndarray:
    mapping = {}
    next_color = 0
    renumbered = np.empty_like(colors)

    for index, color in enumerate(colors):
        color_int = int(color)
        if color_int not in mapping:
            mapping[color_int] = next_color
            next_color += 1
        renumbered[index] = mapping[color_int]

    return renumbered


def _detect_lattice(positions: np.ndarray, rtol: float = 1e-8):
    """Basis ``B`` (2x2, columns) and integer coordinates if ``positions`` lie on a 2-D lattice.

    The basis is the two shortest independent nearest-neighbour vectors,
    Gauss-reduced; square, rectangular, hexagonal and oblique grids are all
    found. Returns ``None`` for other layouts (or fewer than three
    non-collinear actuators).
    """
    if positions.shape[0] < 3:
        return None
    distances, neighbours = cKDTree(positions).query(positions, k=min(9, positions.shape[0]))
    vectors = (positions[neighbours[:, 1:]] - positions[:, None, :]).reshape(-1, 2)
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    order = np.argsort(lengths, kind="stable")
    v1 = vectors[order[0]]
    scale = lengths[order[0]]
    if scale <= 0:
        return None
    cross = np.abs(v1[0] * vectors[order, 1] - v1[1] * vectors[order, 0])
    independent = np.flatnonzero(cross > 1e-6 * scale * lengths[order])
    if independent.size == 0:
        return None
    v2 = vectors[order[independent[0]]]
    # Gauss (Lagrange) reduction of (v1, v2).
    for _ in range(64):
        if v2 @ v2 < v1 @ v1:
            v1, v2 = v2, v1
        mu = np.rint((v1 @ v2) / (v1 @ v1))
        if mu == 0:
            break
        v2 = v2 - mu * v1
    basis = np.column_stack((v1, v2))
    coords = np.linalg.solve(basis, (positions - positions[0]).T).T
    integers = np.rint(coords)
    if np.abs(coords - integers).max() > 1e3 * rtol + 1e-6:
        return None
    return basis, integers.astype(np.int64)


def _short_vectors(gram: np.ndarray, radius: float) -> np.ndarray:
    """Nonzero integer vectors ``n`` with ``n^T gram n < radius^2``."""
    inverse = np.linalg.inv(gram)
    bounds = [int(np.ceil(radius * np.sqrt(inverse[i, i]))) for i in range(2)]
    grid = np.array(
        [(i, j) for i in range(-bounds[0], bounds[0] + 1) for j in range(-bounds[1], bounds[1] + 1)],
        dtype=np.int64,
    )
    norms = np.einsum("ni,ij,nj->n", grid, gram, grid)
    return grid[(norms < radius**2 * (1 - 1e-9)) & np.any(grid != 0, axis=1)]


def _lattice_coloring(positions: np.ndarray, min_distance: float) -> Optional[np.ndarray]:
    """Fewest-colour lattice colouring, or ``None`` if the layout is not a lattice.

    Colours are the cosets of the sublattice ``L`` (Hermite normal form rows
    ``(a, 0)``, ``(b, c)``, ``0 <= b < a``) of smallest index ``a c`` that has
    no nonzero vector shorter than ``min_distance``, so same-colour actuators
    are at least ``min_distance`` apart. A square grid needs at most the
    ``ceil(min_distance / pitch)^2`` colours of a modulo colouring, and often
    fewer (8 instead of 9 at 2.5 pitch).
    """
    found = _detect_lattice(positions)
    if found is None:
        return None
    basis, coords = found
    short = _short_vectors(basis.T @ basis, min_distance)
    if short.size == 0:
        return np.zeros(positions.shape[0], dtype=int)
    index = 1
    while True:
        for a in range(1, index + 1):
            if index % a:
                continue
            c = index // a
            for b in range(a):
                # n = (n0, n1) is in L iff n1 = k c and n0 - k b = 0 (mod a).
                k, rem = np.divmod(short[:, 1], c)
                if not np.any((rem == 0) & ((short[:, 0] - k * b) % a == 0)):
                    k, j = np.divmod(coords[:, 1], c)
                    i = (coords[:, 0] - k * b) % a
                    return _renumber_colors(i + a * j)
        index += 1


def compute_zonal_fast_basis(positions: np.ndarray, min_distance: float) -> np.ndarray:
    """
    Compute a distance-constrained zonal basis.

    Each returned mode is a binary poke pattern. Actuators that are closer than
    ``min_distance`` cannot appear in the same mode. The actuators are
    coloured with a greedy DSATUR colouring of their conflict graph and, when
    they lie on a 2-D lattice (square, hexagonal, ...), with the best
    sublattice colouring as well; the colouring with fewer modes is used
    (the lattice one on ties, for its regular patterns).

    Args:
        positions: ``(n_actuators, 2)`` array of actuator coordinates.
        min_distance: Minimum allowed pairwise distance within a mode.

    Returns:
        ``(n_actuators, n_modes)`` matrix of binary zonal-fast modes.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_actuators, 2).")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative.")
    if positions.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    colors = _renumber_colors(_dsatur_coloring(_build_conflict_graph(positions, min_distance)))
    lattice = _lattice_coloring(positions, min_distance)
    if lattice is not None and lattice.max() <= colors.max():
        colors = lattice
    n_modes = int(colors.max()) + 1

    basis = np.zeros((positions.shape[0], n_modes), dtype=float)
    basis[np.arange(positions.shape[0]), colors] = 1.0
    return basis


class ZonalFastBasisGenerator(BasisGenerator):
    """
    Generate grouped zonal poke patterns separated by a minimum distance.

    A full zonal-fast basis covers every actuator exactly once while using a
    compact coloring of the actuator conflict graph.
    """

    _PARAMETERS = ("min_distance",)

    def __init__(self, positions: np.ndarray, min_distance: float):
        super().__init__(positions)
        if not np.isscalar(min_distance) or not np.isfinite(min_distance) or min_distance < 0:
            raise ValueError("min_distance must be non-negative.")
        self.min_distance = float(min_distance)
        self.full_modes: Optional[np.ndarray] = None

    def generate(
        self,
        n_modes: Optional[int] = None,
        normalize: Optional[str] = None,
        signs: str = "ones",
        seed: Optional[int] = 0,
    ) -> np.ndarray:
        """
        Generate zonal-fast modes.

        Args:
            n_modes: Number of grouped poke modes to return. If omitted, return
                the full distance-constrained basis.
            normalize: ``None`` (unit pokes), ``"rms"``, ``"l2"``,
                ``"peak"`` or ``"pv"`` (see :func:`aobasis.normalize_modes`).
            signs: ``"ones"`` (every poke +1) or ``"random"`` (each poke +1
                or -1 at random, which keeps the patterns zero-mean on
                average and spreads the DM stroke).
            seed: Seed for ``signs="random"``; the default makes the
                patterns reproducible, ``None`` draws fresh ones.

        Returns:
            ``(n_actuators, n_modes)`` matrix of grouped poke patterns.
        """
        self._record_options(n_modes=n_modes, normalize=normalize, signs=signs, seed=seed)
        if signs not in ("ones", "random"):
            raise ValueError(f"signs must be 'ones' or 'random', got {signs!r}.")
        full_basis = compute_zonal_fast_basis(self.positions, self.min_distance)
        if signs == "random":
            full_basis = full_basis * np.random.default_rng(seed).choice([-1.0, 1.0], size=(self.n_actuators, 1))
        self.full_modes = full_basis

        if n_modes is None:
            self.modes = normalize_modes(full_basis, normalize)
            return self.modes

        n_modes = self._validate_n_modes(n_modes)
        if n_modes > full_basis.shape[1]:
            raise ValueError(
                f"Cannot generate {n_modes} zonal-fast modes; full basis only contains {full_basis.shape[1]} modes."
            )

        self.modes = normalize_modes(full_basis[:, :n_modes], normalize)
        return self.modes