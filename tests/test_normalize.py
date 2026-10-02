"""normalize= on every generator and aobasis.normalize_modes (#26)."""

import numpy as np
import pytest

from aobasis import (
    FourierBasisGenerator,
    HadamardBasisGenerator,
    KLBasisGenerator,
    ZernikeBasisGenerator,
    ZonalBasisGenerator,
    ZonalFastBasisGenerator,
    make_circular_actuator_grid,
    normalize_modes,
)


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=12)


GENERATORS = {
    "zernike": (lambda p: ZernikeBasisGenerator(p, pupil_radius=5.0), (12,)),
    "fourier": (lambda p: FourierBasisGenerator(p, pupil_diameter=10.0), (12,)),
    "hadamard": (lambda p: HadamardBasisGenerator(p), (12,)),
    "kl": (lambda p: KLBasisGenerator(p), (12,)),
    "zonal": (lambda p: ZonalBasisGenerator(p), (12,)),
    "zonal_fast": (lambda p: ZonalFastBasisGenerator(p, min_distance=2.0), ()),
}

SIZE = {
    "rms": lambda m: np.sqrt(np.mean(m**2, axis=0)),
    "l2": lambda m: np.linalg.norm(m, axis=0),
    "peak": lambda m: np.abs(m).max(axis=0),
    "pv": lambda m: m.max(axis=0) - m.min(axis=0),
}


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("how", SIZE)
def test_generators_normalize(grid, name, how):
    make, args = GENERATORS[name]
    modes = make(grid).generate(*args, normalize=how)
    size = SIZE[how](modes)
    constant = np.ptp(modes, axis=0) == 0
    if how == "pv":  # piston has no peak-to-valley and is left alone
        assert np.allclose(size[constant], 0.0)
        size = size[~constant]
    assert np.allclose(size, 1.0)


@pytest.mark.parametrize("name", ["zernike", "fourier", "hadamard", "kl"])
def test_normalize_is_applied_after_piston_removal_and_orthonormalization(grid, name):
    make, args = GENERATORS[name]
    plain = make(grid).generate(*args, ignore_piston=True, orthonormalize=True)
    peak = make(grid).generate(*args, ignore_piston=True, orthonormalize=True, normalize="peak")
    assert np.allclose(peak, plain / np.abs(plain).max(axis=0))


def test_kl_eigenvalues_follow_the_normalization(grid):
    plain = KLBasisGenerator(grid)
    plain.generate(10)
    rms = KLBasisGenerator(grid)
    modes = rms.generate(10, normalize="rms")
    # Unit-L2 modes scaled to unit RMS are divided by 1/sqrt(N): variance x 1/N.
    assert np.allclose(rms.eigenvalues, plain.eigenvalues / grid.shape[0])
    cov = rms._von_karman_covariance_cpu()
    coeff_cov = np.linalg.pinv(modes) @ cov @ np.linalg.pinv(modes).T
    assert np.allclose(np.diag(coeff_cov), rms.eigenvalues)


def test_normalize_modes_helper():
    m = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 3.0]])
    assert np.allclose(normalize_modes(m, "peak"), [[1.0, 0.0, -1 / 3], [0.0, 0.0, 1.0]])
    assert np.array_equal(normalize_modes(m, None), m)
    with pytest.raises(ValueError, match="normalize"):
        normalize_modes(m, "max")
    with pytest.raises(ValueError, match="normalize"):
        ZernikeBasisGenerator(np.zeros((1, 2)), 1.0).generate(0, normalize="max")
