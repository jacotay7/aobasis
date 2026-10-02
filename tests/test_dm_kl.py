"""KL modes of the DM by double diagonalization (#29)."""

import numpy as np
import pytest

from aobasis import (
    DMKLBasisGenerator,
    KLBasisGenerator,
    ZernikeBasisGenerator,
    fit_to_influence_functions,
    gaussian_influence_functions,
    make_pupil_points,
    removal_basis,
)


def _dm(pitch=10.0 / 13, extra_rings=1):
    n = 14 + 2 * extra_rings
    axis = (np.arange(n) - (n - 1) / 2) * pitch
    xx, yy = np.meshgrid(axis, axis)
    act = np.column_stack((xx.ravel(), yy.ravel()))
    return act[np.hypot(act[:, 0], act[:, 1]) <= 5.0 + extra_rings * pitch + 1e-9], pitch


@pytest.fixture(scope="module")
def setup():
    act, pitch = _dm()
    points = make_pupil_points(10.0, 40)
    influence = gaussian_influence_functions(act, points, pitch=pitch)
    return act, points, influence


def _turbulence_cov(points, remove=None, ignore_piston=False, outer_scale=30.0):
    cov = KLBasisGenerator(points, outer_scale=outer_scale)._von_karman_covariance_cpu()
    u = removal_basis(points, remove, ignore_piston=ignore_piston)
    p = np.eye(points.shape[0]) - u @ u.T
    return p @ cov @ p


def test_surfaces_orthonormal_and_coefficients_independent(setup):
    act, points, influence = setup
    gen = DMKLBasisGenerator(act, points, influence)
    modes = gen.generate(40, ignore_piston=True)
    assert modes.shape == (act.shape[0], 40)
    n = points.shape[0]
    surfaces = gen.surfaces
    assert np.allclose(surfaces.T @ surfaces / n, np.eye(40), atol=1e-9)
    coeff_cov = surfaces.T @ _turbulence_cov(points, ignore_piston=True) @ surfaces / n**2
    assert np.allclose(coeff_cov, np.diag(gen.eigenvalues), atol=1e-9 * gen.eigenvalues[0])
    assert np.all(np.diff(gen.eigenvalues) <= 0) and np.all(gen.eigenvalues > 0)


def test_removed_modes_are_orthogonal_to_every_surface(setup):
    act, points, influence = setup
    gen = DMKLBasisGenerator(act, points, influence)
    gen.generate(30, ignore_piston=True, remove="tiptilt")
    blocked = np.column_stack([np.ones(points.shape[0]), points])
    surfaces = gen.surfaces
    assert np.abs(blocked.T @ surfaces).max() <= 1e-9 * np.linalg.norm(blocked) * np.linalg.norm(surfaces[:, 0])


def test_dm_kl_captures_more_turbulence_than_fitted_zernikes(setup):
    act, points, influence = setup
    n, k = points.shape[0], 20
    cov = _turbulence_cov(points, ignore_piston=True)
    gen = DMKLBasisGenerator(act, points, influence)
    gen.generate(k, ignore_piston=True)
    zern = ZernikeBasisGenerator(points, pupil_radius=5.0).generate(k, ignore_piston=True)
    zern_commands = fit_to_influence_functions(zern, influence, orthonormalize=True)
    zern_surfaces = influence @ zern_commands
    zern_captured = np.trace(zern_surfaces.T @ cov @ zern_surfaces) / n**2
    assert gen.eigenvalues.sum() >= zern_captured
    assert gen.eigenvalues.sum() < 1.2 * zern_captured  # same ballpark: both are good low-order bases


def test_permuting_actuators_permutes_commands(setup):
    act, points, influence = setup
    perm = np.random.default_rng(0).permutation(act.shape[0])
    modes = DMKLBasisGenerator(act, points, influence).generate(25, ignore_piston=True)
    permuted = DMKLBasisGenerator(act[perm], points, influence[:, perm]).generate(25, ignore_piston=True)
    assert np.allclose(permuted, modes[perm], atol=1e-8)


def test_unseen_actuator_is_dropped(setup):
    act, points, _ = setup
    far = np.vstack([act, [[100.0, 0.0]]])
    influence = gaussian_influence_functions(far, points, pitch=10.0 / 13)
    gen = DMKLBasisGenerator(far, points, influence)
    modes = gen.generate(10, ignore_piston=True)
    assert np.allclose(modes[-1], 0.0)
    with pytest.raises(ValueError):
        gen.generate(far.shape[0] - 1, ignore_piston=True)  # one direction unseen, one removed


def test_kolmogorov_normalize_and_bad_input(setup):
    act, points, influence = setup
    gen = DMKLBasisGenerator(act, points, influence, outer_scale=np.inf)
    with pytest.raises(ValueError, match="ignore_piston"):
        gen.generate(5)
    plain = gen.generate(5, ignore_piston=True).copy()
    values = gen.eigenvalues.copy()
    peak = gen.generate(5, ignore_piston=True, normalize="peak")
    scale = np.abs(plain).max(axis=0)
    assert np.allclose(peak, plain / scale)
    assert np.allclose(gen.eigenvalues, values * scale**2)
    with pytest.raises(ValueError):
        DMKLBasisGenerator(act, points, influence[:, :-1])
