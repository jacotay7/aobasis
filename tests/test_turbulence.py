"""von_karman_covariance: the one definition of the von Karman covariance."""

import numpy as np
import pytest
from scipy.special import gamma, kv

import aobasis
from aobasis import KLBasisGenerator, make_circular_actuator_grid, von_karman_covariance, von_karman_covariance_of_distance


def _variance(r0, L0):
    return (5 / 6) * (6.88 / 2) * gamma(5 / 6) / (gamma(1 / 6) * np.pi ** (5 / 3)) * (L0 / r0) ** (5 / 3)


def test_closed_form():
    r = np.array([0.0, 1e-3, 0.1, 1.0, 10.0, 100.0])
    cov = von_karman_covariance_of_distance(r, r0=0.12, L0=25.0)
    sigma2 = _variance(0.12, 25.0)
    assert sigma2 == pytest.approx(0.0863 * (25.0 / 0.12) ** (5 / 3), rel=1e-3)
    u = 2 * np.pi * r[1:] / 25.0
    expected = sigma2 * 2 ** (1 / 6) / gamma(5 / 6) * u ** (5 / 6) * kv(5 / 6, u)
    assert cov[0] == sigma2
    assert np.allclose(cov[1:], expected, rtol=1e-13, atol=0)
    assert np.all(np.diff(cov) < 0)  # decreasing with separation


def test_structure_function_tends_to_kolmogorov():
    # D(r) = 2 [C(0) - C(r)] = 6.88 (r/r0)^(5/3) - 3 C(0) (2 pi r / L0)^2 + ...
    # i.e. 6.88 (r/r0)^(5/3) [1 - 1.485 (r/L0)^(1/3) + ...].
    r0 = 0.15
    r = np.logspace(-2, 0, 9)
    deviation = []
    for L0 in (1e2, 1e3, 1e4):  # larger L0 loses digits to C(0) - C(r) cancellation
        c0 = von_karman_covariance_of_distance(0.0, r0=r0, L0=L0)
        d = 2 * (c0 - von_karman_covariance_of_distance(r, r0=r0, L0=L0))
        kolmogorov = 6.88 * (r / r0) ** (5 / 3)
        assert np.allclose(d, kolmogorov - 3 * c0 * (2 * np.pi * r / L0) ** 2, rtol=1e-3, atol=0)
        assert np.allclose(d / kolmogorov, 1 - 1.485 * (r / L0) ** (1 / 3), rtol=0, atol=1e-3)
        deviation.append(np.max(1 - d / kolmogorov))
    assert deviation[0] > deviation[1] > deviation[2]
    assert deviation[2] < 0.07  # within 7% of 6.88 (r/r0)^(5/3) up to r = L0 / 10^4


def test_points_match_distances_and_are_symmetric():
    points = np.vstack([make_circular_actuator_grid(4.0, 9), np.random.default_rng(0).uniform(-2, 2, (20, 2))])
    cov = von_karman_covariance(points, r0=0.1, L0=20.0)
    r = np.linalg.norm(points[:, None] - points[None], axis=-1)
    assert np.array_equal(cov, cov.T)
    assert np.all(np.diag(cov) == _variance(0.1, 20.0))
    assert np.allclose(cov, von_karman_covariance_of_distance(r, r0=0.1, L0=20.0), rtol=1e-13, atol=0)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-9 * cov[0, 0])  # positive semi-definite


def test_cross_covariance():
    rng = np.random.default_rng(1)
    a, b = rng.uniform(-1, 1, (7, 2)), rng.uniform(-1, 1, (4, 2))
    cross = von_karman_covariance(a, b, r0=0.2, L0=30.0, units="opd")
    full = von_karman_covariance(np.vstack([a, b]), r0=0.2, L0=30.0, units="opd")
    assert cross.shape == (7, 4)
    assert np.allclose(cross, full[:7, 7:], rtol=1e-13, atol=0)
    same = von_karman_covariance(a, a, r0=0.2, L0=30.0)
    assert np.allclose(np.diag(same), _variance(0.2, 30.0), rtol=1e-15)


def test_units():
    points = make_circular_actuator_grid(2.0, 5)
    phase = von_karman_covariance(points, r0=0.15, L0=25.0, wavelength=500e-9)
    opd = von_karman_covariance(points, r0=0.15, L0=25.0, wavelength=500e-9, units="opd")
    assert np.allclose(opd, phase * (500e-9 / (2 * np.pi)) ** 2, rtol=1e-15, atol=0)
    # OPD does not depend on the wavelength r0 is quoted at (r0 ~ lambda^(6/5)).
    r0_h = 0.15 * (1.65e-6 / 500e-9) ** (6 / 5)
    opd_h = von_karman_covariance(points, r0=r0_h, L0=25.0, wavelength=1.65e-6, units="opd")
    assert np.allclose(opd_h, opd, rtol=1e-12, atol=0)
    assert opd[0, 0] == pytest.approx(0.0863 * (25.0 / 0.15) ** (5 / 3) * (500e-9 / (2 * np.pi)) ** 2, rel=1e-3)


def test_kl_uses_the_public_covariance():
    positions = make_circular_actuator_grid(8.0, 14)
    gen = KLBasisGenerator(positions, fried_parameter=0.13, outer_scale=22.0)
    assert np.array_equal(gen._von_karman_covariance_cpu(), von_karman_covariance(positions, r0=0.13, L0=22.0))
    gen.generate(10, ignore_piston=True)  # eigenvalues are in rad^2 at r0_wavelength
    assert gen._sigma2() == aobasis.turbulence._von_karman_variance(0.13, 22.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(r0=0.0, L0=10.0), "r0"),
        (dict(r0=0.1, L0=-1.0), "L0"),
        (dict(r0=0.1, L0=np.inf), "infinite"),
        (dict(r0=0.1, L0=10.0, wavelength=0.0), "wavelength"),
        (dict(r0=0.1, L0=10.0, units="nm"), "units"),
    ],
)
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        von_karman_covariance(np.zeros((2, 2)), **kwargs)


def test_point_and_distance_validation():
    with pytest.raises(ValueError, match="points_a"):
        von_karman_covariance(np.zeros(3), r0=0.1, L0=10.0)
    with pytest.raises(ValueError, match="points_b"):
        von_karman_covariance(np.zeros((2, 2)), np.full((2, 2), np.nan), r0=0.1, L0=10.0)
    with pytest.raises(ValueError, match="non-negative"):
        von_karman_covariance_of_distance([-1.0], r0=0.1, L0=10.0)
