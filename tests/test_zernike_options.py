"""Zernike orderings, annular Zernikes and the default pupil radius (#31)."""

import warnings

import numpy as np
import pytest

from aobasis import ZernikeBasisGenerator, make_circular_actuator_grid, make_pupil_points


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=12)


ANSI = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3), (4, -4)]
FRINGE = [
    (0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, 3), (3, -3),
    (4, 2), (4, -2), (5, 1), (5, -1), (6, 0), (4, 4), (4, -4), (5, 3), (5, -3), (6, 2), (6, -2),
    (7, 1), (7, -1), (8, 0), (5, 5), (5, -5), (6, 4), (6, -4), (7, 3), (7, -3), (8, 2), (8, -2),
    (9, 1), (9, -1), (10, 0), (12, 0),
]


def test_index_tables():
    assert [ZernikeBasisGenerator._ansi_to_nm(j) for j in range(len(ANSI))] == ANSI
    assert [ZernikeBasisGenerator._fringe_to_nm(j) for j in range(1, 38)] == FRINGE
    with pytest.raises(ValueError):
        ZernikeBasisGenerator._fringe_to_nm(38)


@pytest.mark.parametrize("ordering, table, first", [("ansi", ANSI, 0), ("fringe", FRINGE, 1)])
def test_orderings_reorder_the_noll_modes(ordering, table, first):
    gen = ZernikeBasisGenerator(make_circular_actuator_grid(10.0, 20), pupil_radius=5.0)
    noll = gen.generate(66)
    nm_to_noll = {gen._noll_to_nm(j): j for j in range(1, 67)}
    reordered = gen.generate(11, ordering=ordering)
    for k, nm in enumerate(table[:11]):
        assert np.allclose(reordered[:, k], noll[:, nm_to_noll[nm] - 1])


def test_fringe_limits_and_bad_ordering(grid):
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    assert gen.generate(36, ordering="fringe", ignore_piston=True).shape[1] == 36
    with pytest.raises(ValueError, match="37 terms"):
        gen.generate(38, ordering="fringe")
    with pytest.raises(ValueError, match="ordering"):
        gen.generate(3, ordering="osa")


def test_annular_low_orders_match_mahajan():
    eps = 0.35
    points = make_pupil_points(10.0, 40, obscuration=eps)
    modes = ZernikeBasisGenerator(points, pupil_radius=5.0, obscuration=eps).generate(11)
    x, y = points.T / 5.0
    rho, theta = np.hypot(x, y), np.arctan2(y, x)
    expected = {
        1: 2 * rho * np.cos(theta) / np.sqrt(1 + eps**2),  # tip, j=2
        3: np.sqrt(3) * (2 * rho**2 - 1 - eps**2) / (1 - eps**2),  # defocus, j=4
        5: np.sqrt(6) * rho**2 * np.cos(2 * theta) / np.sqrt(1 + eps**2 + eps**4),  # j=6
        10: np.sqrt(5)
        * (6 * rho**4 - 6 * (1 + eps**2) * rho**2 + 1 + 4 * eps**2 + eps**4)
        / (1 - eps**2) ** 2,  # spherical, j=11
    }
    for k, value in expected.items():
        assert np.allclose(modes[:, k], value, atol=1e-12)


@pytest.mark.parametrize("eps", [0.1, 0.6, 0.9])
def test_annular_radial_functions_are_orthonormal_to_high_order(eps):
    gen = ZernikeBasisGenerator(np.array([[1.0, 0.0]]), pupil_radius=1.0, obscuration=eps)
    x, w = np.polynomial.legendre.leggauss(400)
    rho = eps + (1 - eps) * (x + 1) / 2
    w = w * (1 - eps) / 2 * rho * 2 / (1 - eps**2)
    for m in (0, 5):
        orders = range(m, m + 120, 2)
        radial = np.array([gen._radial_function(n, m, rho) for n in orders])
        assert np.allclose((radial * w) @ radial.T, np.eye(len(radial)), atol=1e-10)
        assert all(gen._radial_function(n, m, np.array([1.0]))[0] > 0 for n in orders)


def test_small_obscuration_tends_to_circular(grid):
    circular = ZernikeBasisGenerator(grid, pupil_radius=5.0).generate(40)
    nearly = ZernikeBasisGenerator(grid, pupil_radius=5.0, obscuration=1e-9).generate(40)
    assert np.allclose(nearly, circular, atol=1e-9)


def test_annular_warns_for_obscured_actuators(grid):
    with pytest.warns(RuntimeWarning, match="obscuration"):
        ZernikeBasisGenerator(grid, pupil_radius=5.0, obscuration=0.3).generate(5)


def test_default_pupil_radius(grid):
    gen = ZernikeBasisGenerator(grid)
    assert gen.pupil_radius == pytest.approx(np.hypot(*grid.T).max())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no actuator outside the default radius
        gen.generate(10)
    with pytest.raises(ValueError):
        ZernikeBasisGenerator(grid, obscuration=1.0)
    with pytest.raises(ValueError):
        ZernikeBasisGenerator(np.zeros((3, 2)))
