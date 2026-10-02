"""Hexagonal grids, central obstruction, spiders, explicit pitch (#32)."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from aobasis import (
    make_circular_actuator_grid,
    make_concentric_actuator_grid,
    make_hexagonal_actuator_grid,
)


def _neighbour_distances(positions, k=7):
    distances, _ = cKDTree(positions).query(positions, k=k)
    return distances[:, 1:]


def test_hexagonal_grid():
    positions = make_hexagonal_actuator_grid(10.0, 0.5)
    assert np.hypot(*positions.T).max() <= 5.0 + 1e-9
    assert np.any(np.all(np.isclose(positions, 0.0), axis=1))  # centre actuator
    interior = np.hypot(*positions.T) < 4.0
    distances = _neighbour_distances(positions)[interior]
    assert np.allclose(distances[:, :6], 0.5)  # six neighbours at one pitch
    expected = np.pi * 25.0 / (np.sqrt(3) / 2 * 0.25)  # pupil area / cell area
    assert abs(positions.shape[0] - expected) / expected < 0.05


def test_circular_grid_with_pitch_and_cell_centres():
    by_pitch = make_circular_actuator_grid(10.0, pitch=0.8)
    assert np.any(np.all(np.isclose(by_pitch, 0.0), axis=1))
    assert np.allclose(np.diff(np.unique(np.round(by_pitch[:, 0], 9))), 0.8)
    cells = make_circular_actuator_grid(10.0, 10, rim=False)
    xs = np.unique(np.round(cells[:, 0], 9))
    assert np.allclose(np.diff(xs), 1.0) and np.isclose(xs.max(), 4.5)
    with pytest.raises(ValueError, match="exactly one"):
        make_circular_actuator_grid(10.0, 10, pitch=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        make_circular_actuator_grid(10.0)


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: make_circular_actuator_grid(10.0, 31, **kw),
        lambda **kw: make_hexagonal_actuator_grid(10.0, 0.35, **kw),
        lambda **kw: make_concentric_actuator_grid(10.0, 12, **kw),
    ],
)
def test_obscuration_and_spiders(make):
    full = make()
    obscured = make(obscuration=0.3)
    radius = np.hypot(*obscured.T)
    assert radius.min() >= 1.5 - 1e-9
    assert obscured.shape[0] < full.shape[0]

    spiders = make(n_spiders=4, spider_width=0.4, spider_angle=np.pi / 4)
    for k in range(4):
        angle = np.pi / 4 + k * np.pi / 2
        along = spiders @ [np.cos(angle), np.sin(angle)]
        across = spiders @ [-np.sin(angle), np.cos(angle)]
        assert not np.any((along >= 0) & (np.abs(across) < 0.2))
    assert spiders.shape[0] < full.shape[0]
    assert np.array_equal(make(n_spiders=4), full)  # zero width removes nothing


def test_mask_validation():
    with pytest.raises(ValueError):
        make_hexagonal_actuator_grid(10.0, 0.5, obscuration=1.0)
    with pytest.raises(ValueError):
        make_hexagonal_actuator_grid(10.0, 0.5, spider_width=-1.0)
    with pytest.raises(ValueError, match="every actuator"):
        make_circular_actuator_grid(10.0, 1, obscuration=0.5)
