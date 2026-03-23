from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .base import BasisGenerator

class ZonalBasisGenerator(BasisGenerator):
    """
    Generates a Zonal basis (Identity matrix).
    Each mode corresponds to poking a single actuator.
    """
    
    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        """
        Generate Zonal modes.
        
        Args:
            n_modes: Number of modes to generate. 
                     If n_modes < n_actuators, returns the first n_modes actuators.
                     If n_modes > n_actuators, raises ValueError (or we could pad with zeros, but that's weird).
        """
        if n_modes > self.n_actuators:
            raise ValueError(f"Cannot generate {n_modes} zonal modes for {self.n_actuators} actuators.")
            
        # Identity matrix
        full_basis = np.eye(self.n_actuators)
        
        self.modes = full_basis[:, :n_modes]
        return self.modes


def _build_conflict_graph(positions: np.ndarray, min_distance: float) -> list[set[int]]:
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


def _dsatur_coloring(adjacency: list[set[int]]) -> np.ndarray:
    n_vertices = len(adjacency)
    colors = np.full(n_vertices, -1, dtype=int)
    neighbor_colors = [set() for _ in range(n_vertices)]
    degrees = np.array([len(neighbors) for neighbors in adjacency], dtype=int)

    for _ in range(n_vertices):
        uncolored = np.flatnonzero(colors < 0)
        if uncolored.size == 0:
            break

        saturation = np.array([len(neighbor_colors[index]) for index in uncolored], dtype=int)
        candidate_order = np.lexsort((-degrees[uncolored], -saturation))
        vertex = int(uncolored[candidate_order[0]])

        used = neighbor_colors[vertex]
        color = 0
        while color in used:
            color += 1

        colors[vertex] = color
        for neighbor in adjacency[vertex]:
            if colors[neighbor] < 0:
                neighbor_colors[neighbor].add(color)

    return colors


def compute_zonal_fast_basis(positions: np.ndarray, min_distance: float) -> np.ndarray:
    """
    Compute a distance-constrained zonal basis.

    Each returned mode is a binary poke pattern. Actuators that are closer than
    ``min_distance`` cannot appear in the same mode, so the basis is built by
    coloring the actuator conflict graph and turning each color into one column
    of the returned matrix.

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

    adjacency = _build_conflict_graph(positions, min_distance)
    colors = _dsatur_coloring(adjacency)
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

    def __init__(self, positions: np.ndarray, min_distance: float):
        super().__init__(positions)
        if min_distance < 0:
            raise ValueError("min_distance must be non-negative.")
        self.min_distance = float(min_distance)
        self.full_modes: Optional[np.ndarray] = None

    def generate(self, n_modes: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Generate zonal-fast modes.

        Args:
            n_modes: Number of grouped poke modes to return. If omitted, return
                the full distance-constrained basis.

        Returns:
            ``(n_actuators, n_modes)`` matrix of binary grouped poke patterns.
        """
        full_basis = compute_zonal_fast_basis(self.positions, self.min_distance)
        self.full_modes = full_basis

        if n_modes is None:
            self.modes = full_basis
            return self.modes

        if n_modes < 0:
            raise ValueError("n_modes must be non-negative.")
        if n_modes > full_basis.shape[1]:
            raise ValueError(
                f"Cannot generate {n_modes} zonal-fast modes; full basis only contains {full_basis.shape[1]} modes."
            )

        self.modes = full_basis[:, :n_modes]
        return self.modes