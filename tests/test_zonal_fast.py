"""Zonal fast: lattice colourings, heap DSATUR, random signs (#34)."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from aobasis import (
    ZonalFastBasisGenerator,
    make_circular_actuator_grid,
    make_hexagonal_actuator_grid,
)
from aobasis.zonal import _build_conflict_graph, _detect_lattice, _dsatur_coloring, compute_zonal_fast_basis


def _check(positions, modes, min_distance):
    support = modes != 0
    assert np.all(support.sum(axis=1) == 1)  # every actuator in exactly one mode
    for k in range(modes.shape[1]):
        members = positions[support[:, k]]
        if members.shape[0] > 1:
            assert pdist(members).min() >= min_distance * (1 - 1e-9)


def _reference_dsatur(adjacency):
    """The original O(V^2) DSATUR, to check the heap version against."""
    n = len(adjacency)
    colors = np.full(n, -1)
    seen = [set() for _ in range(n)]
    degrees = np.array([len(a) for a in adjacency])
    for _ in range(n):
        uncolored = np.flatnonzero(colors < 0)
        saturation = np.array([len(seen[v]) for v in uncolored])
        vertex = int(uncolored[np.lexsort((-degrees[uncolored], -saturation))[0]])
        color = 0
        while color in seen[vertex]:
            color += 1
        colors[vertex] = color
        for w in adjacency[vertex]:
            if colors[w] < 0:
                seen[w].add(color)
    return colors


LAYOUTS = {
    "square": make_circular_actuator_grid(10.0, 32),
    "hexagonal": make_hexagonal_actuator_grid(10.0, 0.4),
    "oblique": np.array([[i + 0.3 * j, 0.8 * j] for i in range(15) for j in range(12)], dtype=float),
}


@pytest.mark.parametrize("name", LAYOUTS)
@pytest.mark.parametrize("factor", [0.9, 1.5, 2.5, 3.5, 5.0])
def test_colourings_respect_min_distance(name, factor):
    positions = LAYOUTS[name]
    min_distance = factor * np.sort(pdist(positions))[0]
    _check(positions, compute_zonal_fast_basis(positions, min_distance), min_distance)


def test_lattice_colouring_beats_square_modulo():
    positions = make_circular_actuator_grid(10.0, 32)
    pitch = 10.0 / 31
    counts = [compute_zonal_fast_basis(positions, f * pitch).shape[1] for f in (2.5, 3.5)]
    assert counts[0] == 8 and counts[1] <= 12  # modulo colouring needs 9 and 16


def test_lattice_detection():
    assert _detect_lattice(LAYOUTS["hexagonal"]) is not None
    assert _detect_lattice(LAYOUTS["oblique"]) is not None
    random = np.random.default_rng(0).uniform(-1, 1, (50, 2))
    assert _detect_lattice(random) is None
    assert _detect_lattice(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])) is None  # collinear


def test_heap_dsatur_matches_reference():
    rng = np.random.default_rng(1)
    for n, radius in ((60, 0.4), (400, 0.15)):
        adjacency = _build_conflict_graph(rng.uniform(-1, 1, (n, 2)), radius)
        assert np.array_equal(_dsatur_coloring(adjacency), _reference_dsatur(adjacency))


def test_random_signs():
    positions = make_circular_actuator_grid(10.0, 20)
    gen = ZonalFastBasisGenerator(positions, min_distance=1.5)
    ones = gen.generate()
    signed = gen.generate(signs="random")
    assert np.array_equal(np.abs(signed), ones)
    assert np.any(signed < 0)
    assert np.array_equal(gen.generate(signs="random"), signed)  # seed 0 by default
    assert not np.array_equal(gen.generate(signs="random", seed=1), signed)
    with pytest.raises(ValueError, match="signs"):
        gen.generate(signs="alternating")
