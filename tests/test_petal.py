"""Petal (fragment-piston) modes: fragment labelling, modes and the joint projector."""

import warnings

import numpy as np
import pytest

from aobasis import PetalBasis, label_fragments, petal_modes, zernike_modes_on_mask

N = 128
RADIUS = 60.0


def _coords(n=N):
    axis = np.arange(n) - 0.5 * (n - 1)
    return np.meshgrid(axis, axis)  # x (columns), y (rows)


def spider_pupil(n=N, radius=RADIUS, obscuration=15.0, n_spiders=3, width=4.0, angle_deg=10.0):
    """Disc with a central obscuration and straight spiders crossing the centre.

    ``n_spiders`` full-diameter spiders cut the annulus into ``2 n_spiders``
    fragments.
    """
    x, y = _coords(n)
    r = np.hypot(x, y)
    pupil = (r <= radius) & (r >= obscuration)
    for k in range(n_spiders):
        theta = np.deg2rad(angle_deg) + np.pi * k / n_spiders
        across = -x * np.sin(theta) + y * np.cos(theta)
        pupil &= np.abs(across) >= 0.5 * width
    return pupil.astype(float)


@pytest.fixture(scope="module")
def petals():
    return PetalBasis(spider_pupil())


def test_six_fragments(petals):
    assert petals.n_fragments == 6
    assert petals.n_modes == 5
    assert petals.modes.shape == (5, N, N)
    assert petals.indicators.shape == (6, N, N)
    assert np.array_equal(petals.lit, spider_pupil() > 0)
    assert petals.areas.sum() == petals.lit.sum()
    # Six equal petals up to the pixel grid.
    assert petals.areas.min() > 0.9 * petals.areas.max()
    labels, n = label_fragments(spider_pupil())
    assert n == 6 and np.array_equal(labels, petals.labels)
    assert set(np.unique(labels)) == set(range(7))


def test_modes_are_orthonormal_and_piston_free(petals):
    modes = petals.modes[:, petals.lit]
    n_lit = petals.lit.sum()
    assert np.allclose(modes @ modes.T / n_lit, np.eye(5), atol=1e-12)
    assert np.allclose(modes.mean(axis=1), 0.0, atol=1e-12)
    assert np.all(petals.modes[:, ~petals.lit] == 0)
    # Constant on each fragment.
    for k in range(1, 7):
        on = petals.labels == k
        assert np.allclose(np.ptp(petals.modes[:, on], axis=1), 0.0)


def test_modes_span_the_piston_free_indicators(petals):
    indicators = petals.indicators[:, petals.lit]
    piston_free = indicators - indicators.mean(axis=1, keepdims=True)
    modes = petals.modes[:, petals.lit]
    residual = piston_free - (piston_free @ modes.T) @ modes / petals.lit.sum()
    assert np.allclose(residual, 0.0, atol=1e-12)


def test_helmert_structure(petals):
    # Mode k (from 1) is positive on fragment k, zero on the earlier ones.
    for k in range(1, petals.n_fragments):
        mode = petals.modes[k - 1]
        assert mode[petals.labels == k][0] > 0
        for earlier in range(1, k):
            assert np.all(mode[petals.labels == earlier] == 0)
        for later in range(k + 1, petals.n_fragments + 1):
            assert mode[petals.labels == later][0] < 0


def test_petal_modes_shortcut(petals):
    assert np.array_equal(petal_modes(spider_pupil()), petals.modes)
    raw = petal_modes(spider_pupil(), raw=True)
    assert np.array_equal(raw, petals.indicators)
    assert np.array_equal(raw.sum(axis=0), petals.lit.astype(float))


def test_pure_petal_map_is_recovered(petals):
    rng = np.random.default_rng(1)
    pistons = rng.normal(size=6)
    phase = petals.piston_map(pistons)
    weights = petals.areas / petals.areas.sum()
    expected = pistons - weights @ pistons
    assert np.allclose(petals.petal_pistons(phase), expected, atol=1e-12)
    smooth = zernike_modes_on_mask(petals.lit, 6, pupil_radius=RADIUS)
    assert np.allclose(petals.petal_pistons(phase, smooth_modes=smooth), expected, atol=1e-10)
    # Coefficients on the orthonormal modes reproduce the piston-free map.
    coefficients = (petals.modes * petals.piston_map(expected)).sum(axis=(1, 2)) / petals.lit.sum()
    rebuilt = np.tensordot(coefficients, petals.modes, axes=1)
    assert np.allclose(rebuilt, petals.piston_map(expected), atol=1e-12)


def test_tilt_is_not_petal_with_the_joint_fit(petals):
    x, y = _coords()
    tilt = np.where(petals.lit, 0.3 * x - 0.7 * y, 0.0)
    naive = petals.petal_pistons(tilt)
    assert np.abs(naive).max() > 1.0  # fragment means of a tilt are far from zero
    smooth = zernike_modes_on_mask(petals.lit, 3, pupil_radius=RADIUS)
    joint = petals.petal_pistons(tilt, smooth_modes=smooth)
    assert np.abs(joint).max() < 1e-9


def test_low_order_map_is_not_petal_and_petals_survive_it(petals):
    smooth = zernike_modes_on_mask(petals.lit, 10, pupil_radius=RADIUS)
    low_order = np.tensordot([0.0, 2.0, -1.0, 0.5, 0.8, -0.3, 0.2, 0.1, -0.4, 0.6], smooth, axes=1)
    assert np.abs(petals.petal_pistons(low_order)).max() > 0.1
    assert np.abs(petals.petal_pistons(low_order, smooth_modes=smooth)).max() < 1e-9

    pistons = np.array([0.5, -0.2, 0.1, 0.3, -0.6, 0.0])
    pistons -= petals.areas @ pistons / petals.areas.sum()
    measured = petals.petal_pistons(low_order + petals.piston_map(pistons), smooth_modes=smooth)
    assert np.allclose(measured, pistons, atol=1e-9)


def test_reconstructor_and_stacks(petals):
    rng = np.random.default_rng(2)
    smooth = zernike_modes_on_mask(petals.lit, 6, pupil_radius=RADIUS)
    phases = rng.normal(size=(4, N, N))
    weights = petals.petal_reconstructor(smooth)
    assert weights.shape == (6, N, N)
    assert np.all(weights[:, ~petals.lit] == 0)
    stacked = petals.petal_pistons(phases, smooth_modes=smooth)
    assert stacked.shape == (4, 6)
    for k in range(4):
        single = petals.petal_pistons(phases[k], smooth_modes=smooth)
        assert np.allclose(stacked[k], single)
        assert np.allclose(np.tensordot(weights, phases[k], axes=2), single)
    # Pixels off the lit pupil are ignored.
    noisy = phases[0] + np.where(petals.lit, 0.0, 1e3)
    assert np.allclose(petals.petal_pistons(noisy, smooth_modes=smooth), stacked[0])


def test_piston_in_smooth_modes_is_ignored(petals):
    with_piston = zernike_modes_on_mask(petals.lit, 3, pupil_radius=RADIUS)
    without = with_piston[1:]
    rng = np.random.default_rng(3)
    phase = rng.normal(size=(N, N))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = petals.petal_pistons(phase, smooth_modes=with_piston)
    assert np.allclose(a, petals.petal_pistons(phase, smooth_modes=without))


def test_smooth_modes_that_reproduce_a_petal_warn(petals):
    with pytest.warns(RuntimeWarning, match="linearly dependent"):
        petals.petal_reconstructor(petals.modes[:1])


def test_min_pixels_drops_small_fragments():
    pupil = spider_pupil()
    pupil[2:4, 2:4] = 1.0  # a 4-pixel island in the corner
    assert PetalBasis(pupil).n_fragments == 7
    filtered = PetalBasis(pupil, min_pixels=5)
    assert filtered.n_fragments == 6
    assert filtered.labels[2, 2] == 0
    assert np.array_equal(filtered.labels, PetalBasis(spider_pupil()).labels)
    labels, n = label_fragments(pupil, min_pixels=5)
    assert n == 6 and labels.max() == 6


def test_four_connectivity_and_threshold():
    pupil = np.zeros((6, 6))
    pupil[1:3, 1:3] = 1.0
    pupil[3:5, 3:5] = 1.0  # touches the first block at a corner only
    assert label_fragments(pupil)[1] == 2
    bridged = np.zeros((5, 7))
    bridged[1:4, 1:3] = 1.0
    bridged[1:4, 4:6] = 1.0
    bridged[1:4, 3] = 0.6  # a half-lit spider column
    assert label_fragments(bridged)[1] == 2
    assert label_fragments(bridged, threshold=0.5)[1] == 1
    assert label_fragments(bridged.astype(bool), threshold=0.5)[1] == 1


def test_pupil_in_one_piece():
    x, y = _coords(64)
    disc = (np.hypot(x, y) <= 30).astype(float)
    basis = PetalBasis(disc)
    assert basis.n_fragments == 1
    assert basis.modes.shape == (0, 64, 64)
    assert petal_modes(disc).shape == (0, 64, 64)
    assert np.array_equal(petal_modes(disc, raw=True)[0], disc)
    assert np.allclose(basis.petal_pistons(x), 0.0)


def test_invalid_input():
    with pytest.raises(ValueError, match="2-D"):
        PetalBasis(np.ones(5))
    with pytest.raises(ValueError, match="No lit fragment"):
        PetalBasis(np.zeros((8, 8)))
    with pytest.raises(ValueError, match="No lit fragment"):
        PetalBasis(np.full((8, 8), 0.9))
    with pytest.raises(ValueError, match="threshold"):
        label_fragments(np.ones((4, 4)), threshold=0)
    with pytest.raises(ValueError, match="min_pixels"):
        label_fragments(np.ones((4, 4)), min_pixels=0)
    with pytest.raises(ValueError, match="finite"):
        label_fragments(np.full((4, 4), np.nan))
    basis = PetalBasis(spider_pupil(n=32, radius=15, obscuration=4, width=2))
    with pytest.raises(ValueError, match="phase"):
        basis.petal_pistons(np.zeros((16, 16)))
    with pytest.raises(ValueError, match="smooth_modes"):
        basis.petal_pistons(np.zeros((32, 32)), smooth_modes=np.zeros((2, 16, 16)))
    with pytest.raises(ValueError, match="pistons"):
        basis.piston_map(np.zeros(3))
