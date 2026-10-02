"""Hadamard constructions and column selection (#33)."""

import numpy as np
import pytest

from aobasis import HadamardBasisGenerator, make_circular_actuator_grid
from aobasis.hadamard import hadamard_matrix, smallest_hadamard_order


@pytest.mark.parametrize("order", [1, 2, 4, 12, 20, 24, 28, 44, 88, 104, 108, 200, 276])
def test_hadamard_matrix_is_hadamard_with_piston_first(order):
    H = hadamard_matrix(order)
    assert H.shape == (order, order)
    assert np.array_equal(H @ H.T, order * np.eye(order, dtype=int))
    assert np.all(H[:, 0] == 1)


def test_unconstructible_orders():
    with pytest.raises(ValueError):
        hadamard_matrix(52)  # 4 * 13; 25 is a prime power, not a prime
    assert smallest_hadamard_order(97) == 104
    assert smallest_hadamard_order(49) == 56


def test_smallest_construction_is_orthogonal_on_matching_grids():
    positions = make_circular_actuator_grid(10.0, 20)  # 276 actuators, a Paley order
    n = positions.shape[0]
    modes = HadamardBasisGenerator(positions).generate(n, construction="smallest")
    assert np.array_equal(np.abs(modes), np.ones_like(modes))
    assert np.allclose(modes.T @ modes, n * np.eye(n))


def test_smallest_construction_truncates_less():
    rng = np.random.default_rng(0)
    positions = rng.uniform(-1, 1, (97, 2))
    sylvester = HadamardBasisGenerator(positions).generate(96, ignore_piston=True)
    smallest = HadamardBasisGenerator(positions).generate(96, ignore_piston=True, construction="smallest")
    assert np.linalg.cond(smallest) < np.linalg.cond(sylvester)


@pytest.mark.parametrize("n_modes, zero_mean", [(60, True), (149, False)])
def test_balanced_selection_is_piston_poor(n_modes, zero_mean):
    positions = np.random.default_rng(0).uniform(-1, 1, (150, 2))  # rows cut from order 256
    gen = HadamardBasisGenerator(positions)
    first = gen.generate(n_modes)
    balanced = gen.generate(n_modes, selection="balanced")
    assert np.array_equal(np.abs(balanced), np.ones_like(balanced))  # still +/-1
    assert np.all(balanced[:, 0] == 1)  # piston kept first
    worst = np.abs(balanced[:, 1:].mean(axis=0)).max()
    assert worst < np.abs(first[:, 1:].mean(axis=0)).max()
    assert (worst == 0) == zero_mean  # exactly zero-mean columns exist for 60 modes
    assert np.linalg.matrix_rank(balanced) == n_modes
    no_piston = gen.generate(n_modes, selection="balanced", ignore_piston=True)
    assert np.allclose(no_piston.mean(axis=0), 0, atol=1e-12)


def test_bad_options():
    gen = HadamardBasisGenerator(make_circular_actuator_grid(10.0, 8))
    with pytest.raises(ValueError, match="construction"):
        gen.generate(3, construction="paley")
    with pytest.raises(ValueError, match="selection"):
        gen.generate(3, selection="best")
