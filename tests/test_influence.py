"""Fitting modes onto DM influence functions (#11)."""

import numpy as np
import pytest

from aobasis import (
    ZernikeBasisGenerator,
    fit_to_influence_functions,
    gaussian_influence_functions,
    make_circular_actuator_grid,
    make_pupil_points,
)


def _dm(extra_rings=1, pitch=10.0 / 19):
    n = 20 + 2 * extra_rings
    axis = (np.arange(n) - (n - 1) / 2) * pitch
    xx, yy = np.meshgrid(axis, axis)
    act = np.column_stack((xx.ravel(), yy.ravel()))
    return act[np.hypot(act[:, 0], act[:, 1]) <= 5.0 + extra_rings * pitch + 1e-9], pitch


def test_make_pupil_points():
    points = make_pupil_points(10.0, 64)
    assert abs(points.shape[0] - np.pi / 4 * 64**2) < 64
    assert np.hypot(*points.T).max() <= 5.0
    annulus = make_pupil_points(10.0, 64, obscuration=0.3)
    assert np.hypot(*annulus.T).min() >= 1.5
    assert annulus.shape[0] < points.shape[0]
    for bad in ({"obscuration": 1.0}, {"obscuration": -0.1}):
        with pytest.raises(ValueError):
            make_pupil_points(10.0, 64, **bad)


def test_gaussian_influence_functions():
    act = make_circular_actuator_grid(10.0, 11)  # pitch 1
    influence = gaussian_influence_functions(act, act, coupling=0.2)
    assert np.allclose(np.diag(influence), 1.0)
    i = int(np.argmin(np.hypot(*act.T)))  # centre
    neighbour = int(np.argmin(np.hypot(act[:, 0] - 1.0, act[:, 1])))
    assert influence[neighbour, i] == pytest.approx(0.2)
    assert np.allclose(gaussian_influence_functions(act, act, coupling=0.2, pitch=1.0), influence)
    with pytest.raises(ValueError):
        gaussian_influence_functions(act, act, coupling=1.5)


def test_fit_recovers_reachable_surfaces():
    act, pitch = _dm()
    points = make_pupil_points(10.0, 48)
    influence = gaussian_influence_functions(act, points, pitch=pitch)
    true = np.random.default_rng(0).standard_normal((act.shape[0], 5))
    assert np.allclose(fit_to_influence_functions(influence @ true, influence, rcond=0), true, atol=1e-8)


def test_fitted_zernikes_beat_point_sampling_and_report_residuals():
    act, pitch = _dm()
    points = make_pupil_points(10.0, 64)
    influence = gaussian_influence_functions(act, points, pitch=pitch)
    pupil = ZernikeBasisGenerator(points, pupil_radius=5.0).generate(20, ignore_piston=True)
    commands, residual = fit_to_influence_functions(pupil, influence, return_residual=True)
    manual = np.linalg.norm(influence @ commands - pupil, axis=0) / np.linalg.norm(pupil, axis=0)
    assert np.allclose(residual, manual)
    assert residual[0] < 0.02  # tip
    with pytest.warns(RuntimeWarning):  # some actuators lie outside the Zernike pupil
        sampled = ZernikeBasisGenerator(act, pupil_radius=5.0).generate(20, ignore_piston=True)
    sampled_residual = np.linalg.norm(influence @ sampled - pupil, axis=0) / np.linalg.norm(pupil, axis=0)
    assert np.all(residual < sampled_residual)


def test_orthonormalized_surfaces():
    act, pitch = _dm()
    points = make_pupil_points(10.0, 48)
    influence = gaussian_influence_functions(act, points, pitch=pitch)
    pupil = ZernikeBasisGenerator(points, pupil_radius=5.0).generate(30, ignore_piston=True)
    plain = fit_to_influence_functions(pupil, influence)
    ortho = fit_to_influence_functions(pupil, influence, orthonormalize=True)
    surfaces = influence @ ortho
    assert np.allclose(surfaces.T @ surfaces / points.shape[0], np.eye(30), atol=1e-10)
    # Same nested spans as the plain fit, same signs.
    assert np.all(np.sum((influence @ plain) * surfaces, axis=0) > 0)


def test_regularization_and_rcond():
    act, pitch = _dm()
    points = make_pupil_points(10.0, 48)
    far = np.vstack([act, [[100.0, 0.0]]])  # an actuator that touches no point
    influence = gaussian_influence_functions(far, points, pitch=pitch)
    pupil = ZernikeBasisGenerator(points, pupil_radius=5.0).generate(10)
    commands = fit_to_influence_functions(pupil, influence)
    assert np.allclose(commands[-1], 0.0)
    damped = fit_to_influence_functions(pupil, influence, regularization=1.0)
    assert np.linalg.norm(damped) < np.linalg.norm(commands)


def test_fit_rejects_bad_shapes():
    influence = np.ones((10, 3))
    with pytest.raises(ValueError):
        fit_to_influence_functions(np.ones((9, 2)), influence)
    with pytest.raises(ValueError):
        fit_to_influence_functions(np.ones((10, 2)), influence, regularization=-1.0)
