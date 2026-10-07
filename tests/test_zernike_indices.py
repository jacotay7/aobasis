"""Public Zernike index helpers, the optional rank check and mask images."""

import time
import warnings

import numpy as np
import pytest

import aobasis
from aobasis import (
    FourierBasisGenerator,
    HadamardBasisGenerator,
    ZernikeBasisGenerator,
    ansi_to_nm,
    fringe_to_nm,
    make_circular_actuator_grid,
    nm_to_ansi,
    nm_to_fringe,
    nm_to_noll,
    noll_to_nm,
    positions_from_mask,
    zernike_modes_on_mask,
)

# Noll (1976), Table 1.
NOLL_1_TO_15 = [
    (0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1),
    (3, -3), (3, 3), (4, 0), (4, 2), (4, -2), (4, 4), (4, -4),
]


def test_noll_sequence():
    assert [noll_to_nm(j) for j in range(1, 16)] == NOLL_1_TO_15
    assert [nm_to_noll(n, m) for n, m in NOLL_1_TO_15] == list(range(1, 16))


def test_round_trips():
    for j in range(1, 501):
        assert nm_to_noll(*noll_to_nm(j)) == j
    for j in range(0, 501):
        assert nm_to_ansi(*ansi_to_nm(j)) == j
    for j in range(1, 38):
        assert nm_to_fringe(*fringe_to_nm(j)) == j
    # Every (n, m) of radial order <= 30 appears exactly once.
    pairs = [(n, m) for n in range(31) for m in range(-n, n + 1, 2)]
    assert sorted(nm_to_noll(n, m) for n, m in pairs) == list(range(1, len(pairs) + 1))
    assert sorted(nm_to_ansi(n, m) for n, m in pairs) == list(range(len(pairs)))


def test_arrays_and_scalar_types():
    j = np.arange(1, 1_000_001).reshape(1000, 1000)
    n, m = noll_to_nm(j)
    assert n.shape == m.shape == j.shape
    assert np.array_equal(nm_to_noll(n, m), j)
    # Within order n there are n + 1 indices, ending at (n + 1)(n + 2) / 2.
    assert np.all((n * (n + 1) // 2 < j) & (j <= (n + 1) * (n + 2) // 2))
    assert np.array_equal(nm_to_ansi(*ansi_to_nm(j)), j)
    assert np.array_equal(nm_to_noll(4, [-4, -2, 0, 2, 4]), [15, 13, 11, 12, 14])
    n, m = noll_to_nm([2, 3])
    assert np.array_equal(n, [1, 1]) and np.array_equal(m, [1, -1])
    n, m = noll_to_nm(np.int32(7))
    assert type(n) is int and type(m) is int and type(nm_to_noll(n, m)) is int
    assert noll_to_nm(np.array([], dtype=int))[0].shape == (0,)


def test_private_methods_use_the_public_mapping():
    assert [ZernikeBasisGenerator._noll_to_nm(j) for j in range(1, 100)] == [noll_to_nm(j) for j in range(1, 100)]
    assert ZernikeBasisGenerator._ansi_to_nm(12) == ansi_to_nm(12)
    assert ZernikeBasisGenerator._fringe_to_nm(37) == fringe_to_nm(37) == (12, 0)
    for name in ("noll_to_nm", "nm_to_noll", "ansi_to_nm", "nm_to_ansi", "fringe_to_nm", "nm_to_fringe"):
        assert name in aobasis.__all__


@pytest.mark.parametrize(
    "call",
    [
        lambda: noll_to_nm(0),
        lambda: noll_to_nm(True),
        lambda: noll_to_nm(2.0),
        lambda: noll_to_nm([1, 0]),
        lambda: ansi_to_nm(-1),
        lambda: fringe_to_nm(0),
        lambda: fringe_to_nm(38),
        lambda: nm_to_noll(2, 1),
        lambda: nm_to_noll(1, 3),
        lambda: nm_to_noll(-1, 1),
        lambda: nm_to_ansi(3, 0),
        lambda: nm_to_fringe(6, 6),  # its formula index would clash with Z37
        lambda: nm_to_fringe(11, 1),
        lambda: nm_to_fringe([2, 12], [0, 2]),
    ],
)
def test_invalid_indices(call):
    with pytest.raises(ValueError):
        call()


# Unit-RMS closed forms (Noll 1976) keyed by (n, m): cos for m > 0, sin for m < 0.
CLOSED_FORMS = {
    (0, 0): lambda r, t: np.ones_like(r),
    (1, 1): lambda r, t: 2 * r * np.cos(t),
    (1, -1): lambda r, t: 2 * r * np.sin(t),
    (2, 0): lambda r, t: np.sqrt(3) * (2 * r**2 - 1),
    (2, -2): lambda r, t: np.sqrt(6) * r**2 * np.sin(2 * t),
    (2, 2): lambda r, t: np.sqrt(6) * r**2 * np.cos(2 * t),
    (3, -1): lambda r, t: np.sqrt(8) * (3 * r**3 - 2 * r) * np.sin(t),
    (3, 1): lambda r, t: np.sqrt(8) * (3 * r**3 - 2 * r) * np.cos(t),
    (3, -3): lambda r, t: np.sqrt(8) * r**3 * np.sin(3 * t),
    (3, 3): lambda r, t: np.sqrt(8) * r**3 * np.cos(3 * t),
    (4, 0): lambda r, t: np.sqrt(5) * (6 * r**4 - 6 * r**2 + 1),
    (4, 2): lambda r, t: np.sqrt(10) * (4 * r**4 - 3 * r**2) * np.cos(2 * t),
    (4, -2): lambda r, t: np.sqrt(10) * (4 * r**4 - 3 * r**2) * np.sin(2 * t),
}


@pytest.mark.parametrize("ordering, to_nm, first", [("noll", noll_to_nm, 1), ("ansi", ansi_to_nm, 0), ("fringe", fringe_to_nm, 1)])
def test_index_helpers_describe_the_generated_modes(ordering, to_nm, first):
    positions = make_circular_actuator_grid(2.0, 24)
    modes = ZernikeBasisGenerator(positions, pupil_radius=1.0).generate(13, ordering=ordering)
    rho, theta = np.hypot(*positions.T), np.arctan2(positions[:, 1], positions[:, 0])
    checked = 0
    for k in range(13):
        nm = to_nm(k + first)
        if nm in CLOSED_FORMS:
            assert np.allclose(modes[:, k], CLOSED_FORMS[nm](rho, theta), atol=1e-12), (ordering, k, nm)
            checked += 1
    assert checked >= 10


def test_tip_varies_along_x_and_tilt_along_y():
    positions = make_circular_actuator_grid(2.0, 9)
    modes = ZernikeBasisGenerator(positions, pupil_radius=1.0).generate(nm_to_noll(1, -1))
    tip, tilt = modes[:, nm_to_noll(1, 1) - 1], modes[:, nm_to_noll(1, -1) - 1]
    assert np.allclose(tip, 2 * positions[:, 0]) and np.allclose(tilt, 2 * positions[:, 1])


def _rank_deficient_cases():
    grid = make_circular_actuator_grid(10.0, 12)
    return [
        (ZernikeBasisGenerator(grid, pupil_radius=5.0), {}),
        (ZernikeBasisGenerator(grid, pupil_radius=5.0), {"ordering": "ansi", "ignore_piston": True}),
        (FourierBasisGenerator(grid, pupil_diameter=10.0), {"remove": "tiptilt"}),
        (HadamardBasisGenerator(grid), {"ignore_piston": True}),
    ]


@pytest.mark.parametrize("orthonormalize", [False, True])
@pytest.mark.parametrize("generator, options", _rank_deficient_cases())
def test_check_rank_false_gives_identical_modes(generator, options, orthonormalize):
    n_modes = generator.n_actuators - (1 if options.get("ignore_piston") else 0) - (
        2 if options.get("remove") else 0
    )
    n_modes = min(n_modes, 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        checked = generator.generate(n_modes, orthonormalize=orthonormalize, **options).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        unchecked = generator.generate(n_modes, orthonormalize=orthonormalize, check_rank=False, **options)
    assert np.array_equal(checked, unchecked)
    assert "check_rank" not in generator.generate_options


def test_check_rank_false_silences_the_warning_and_skips_the_qr(monkeypatch):
    grid = make_circular_actuator_grid(10.0, 12)
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    n = grid.shape[0]
    with pytest.warns(RuntimeWarning, match="rank"):
        gen.generate(n)
    with pytest.warns(RuntimeWarning, match="linearly dependent"):
        gen.generate(n, orthonormalize=True)

    def fail(modes):
        raise AssertionError("the rank check ran")

    monkeypatch.setattr("aobasis.base._numerical_rank", fail)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gen.generate(n, check_rank=False)
        gen.generate(n, orthonormalize=True, check_rank=False)
        FourierBasisGenerator(grid, 10.0).generate(20, check_rank=False)
        HadamardBasisGenerator(grid).generate(20, check_rank=False)
    with pytest.raises(AssertionError, match="rank check ran"):
        gen.generate(10)


def test_check_rank_false_is_faster():
    positions = make_circular_actuator_grid(2.0, 120)  # ~11k actuators
    gen = ZernikeBasisGenerator(positions, pupil_radius=1.0)

    def best(**options):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            gen.generate(150, **options)
            times.append(time.perf_counter() - start)
        return min(times)

    assert best(check_rank=False) < best()


def _disc(size, radius):
    y, x = np.mgrid[:size, :size] - (size - 1) / 2.0
    return np.hypot(x, y) <= radius


def test_zernike_modes_on_mask_matches_the_generator():
    mask = _disc(32, 16.0)
    images = zernike_modes_on_mask(mask, 21, pupil_radius=16.0)
    assert images.shape == (21, 32, 32) and images.dtype == float
    assert np.all(images[:, ~mask] == 0)
    modes = ZernikeBasisGenerator(positions_from_mask(mask, 1.0), pupil_radius=16.0).generate(21)
    assert np.array_equal(images[:, mask], modes.T)
    # Tip varies along the columns (x), tilt along the rows (y).
    y, x = np.mgrid[:32, :32] - 15.5
    assert np.allclose(images[1][mask], 2 * x[mask] / 16.0)
    assert np.allclose(images[2][mask], 2 * y[mask] / 16.0)


def test_zernike_modes_on_mask_options():
    mask = _disc(24, 11.5).astype(np.uint8)  # any nonzero pixel counts
    default = zernike_modes_on_mask(mask, 5)
    inside = mask != 0
    radius = np.max(np.hypot(*positions_from_mask(inside, 1.0).T))
    assert np.allclose(default, zernike_modes_on_mask(mask, 5, pupil_radius=radius))
    fringe = zernike_modes_on_mask(mask, 9, ordering="fringe", ignore_piston=True, normalize="rms", check_rank=False)
    assert fringe.shape == (9, 24, 24)
    assert np.allclose(fringe[:, inside].mean(axis=1), 0, atol=1e-12)
    assert np.allclose(np.sqrt(np.mean(fringe[:, inside] ** 2, axis=1)), 1)
    annulus = inside & ~_disc(24, 0.3 * 12.0)
    annular = zernike_modes_on_mask(annulus, 4, pupil_radius=12.0, obscuration=0.3)
    assert annular.shape == (4, 24, 24)
    with pytest.raises(TypeError):
        zernike_modes_on_mask(mask, 3, orthonormalise=True)
    with pytest.raises(ValueError, match="2-D"):
        zernike_modes_on_mask(np.ones(5), 1)
    with pytest.raises(ValueError, match="nonzero"):
        zernike_modes_on_mask(np.zeros((4, 4)), 1)
