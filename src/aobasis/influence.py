"""Fit modal bases onto deformable-mirror influence functions.

The generators sample each mode at the points they are given. Sampling at
the actuator positions gives commands directly, but a real DM surface is the
superposition of its influence functions, not a set of point values. To get
commands whose DM surface best matches a mode, evaluate the mode on pupil
points instead (every generator accepts arbitrary points) and fit it onto
the influence functions by least squares::

    points = make_pupil_points(diameter=10.0, n_pixels=64)
    actuators = make_circular_actuator_grid(10.0, 20)
    influence = gaussian_influence_functions(actuators, points)
    pupil_modes = ZernikeBasisGenerator(points, pupil_radius=5.0).generate(30, ignore_piston=True)
    commands = fit_to_influence_functions(pupil_modes, influence)  # (n_actuators, 30)

The fit is only as good as the DM's coverage of the pupil: with actuators
that stop at the rim, the surface falls off at the edge (tip fits to ~20%
RMS above); one ring of actuators outside the pupil, as real DMs have,
brings that to ~1%.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree

from .base import _validate_positions_array
from .utils import _validate_positive_finite_scalar, _validate_non_negative_integer


def make_pupil_points(diameter: float, n_pixels: int, obscuration: float = 0.0) -> np.ndarray:
    """Centres of the pixels of an ``n_pixels``-across grid inside an (annular) pupil.

    Args:
        diameter: Pupil diameter.
        n_pixels: Pixels across the diameter; the pixel size is
            ``diameter / n_pixels``.
        obscuration: Central obstruction as a fraction of the diameter
            (``0 <= obscuration < 1``).

    Returns:
        ``(n_points, 2)`` array of ``(x, y)``, row-major.
    """
    diameter = _validate_positive_finite_scalar(diameter, "diameter")
    n_pixels = _validate_non_negative_integer(n_pixels, "n_pixels", minimum=1)
    if not np.isscalar(obscuration) or not np.isfinite(obscuration) or not 0 <= obscuration < 1:
        raise ValueError("obscuration must be in [0, 1).")
    pixel = diameter / n_pixels
    axis = (np.arange(n_pixels) - 0.5 * (n_pixels - 1)) * pixel
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy).ravel()
    inside = (radius <= 0.5 * diameter) & (radius >= 0.5 * diameter * obscuration)
    return np.column_stack((xx.ravel(), yy.ravel()))[inside]


def _nearest_neighbour_distance(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        raise ValueError("pitch cannot be inferred from fewer than two actuators; pass pitch.")
    distances, _ = cKDTree(positions).query(positions, k=2)
    return float(np.median(distances[:, 1]))


def gaussian_influence_functions(
    actuator_positions: np.ndarray,
    points: np.ndarray,
    coupling: float = 0.15,
    pitch: Optional[float] = None,
) -> np.ndarray:
    """Gaussian influence functions sampled at ``points``.

    ``IF[p, a] = coupling ** ((|x_p - x_a| / pitch) ** 2)``: 1 at the
    actuator and ``coupling`` at one pitch from it, the usual model of a
    continuous-facesheet DM.

    Args:
        actuator_positions: ``(n_actuators, 2)`` actuator coordinates.
        points: ``(n_points, 2)`` sample points, e.g. from :func:`make_pupil_points`.
        coupling: Inter-actuator coupling, ``0 < coupling < 1``.
        pitch: Actuator pitch; by default the median nearest-neighbour
            distance between actuators.

    Returns:
        ``(n_points, n_actuators)`` matrix.
    """
    actuators = _validate_positions_array(actuator_positions)
    points = _validate_positions_array(points)
    if not np.isscalar(coupling) or not np.isfinite(coupling) or not 0 < coupling < 1:
        raise ValueError("coupling must be in (0, 1).")
    pitch = _nearest_neighbour_distance(actuators) if pitch is None else _validate_positive_finite_scalar(pitch, "pitch")
    d2 = np.sum((points[:, None, :] - actuators[None, :, :]) ** 2, axis=-1)
    return np.exp(np.log(coupling) * d2 / pitch**2)


def fit_to_influence_functions(
    modes: np.ndarray,
    influence_functions: np.ndarray,
    rcond: float = 1e-6,
    regularization: float = 0.0,
    orthonormalize: bool = False,
    return_residual: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Least-squares DM commands whose surfaces best match ``modes``.

    Solves ``min_c ||IF c - m||^2 + regularization * ||c||^2`` for every mode
    ``m``, through one SVD of the influence functions. Singular values below
    ``rcond`` times the largest are discarded.

    Args:
        modes: ``(n_points, n_modes)`` modes sampled at the points where the
            influence functions are sampled.
        influence_functions: ``(n_points, n_actuators)`` matrix; column ``a``
            is actuator ``a``'s surface for a unit command.
        rcond: Relative cutoff for small singular values of the influence
            functions (unseen or badly seen actuators).
        regularization: Tikhonov weight (in units of ``IF^T IF``'s mean
            diagonal, so it is independent of the number of points).
        orthonormalize: Gram-Schmidt the fitted DM surfaces in order, so
            they are orthonormal with unit RMS over the points
            (``S^T S / n_points = I``); the commands are transformed alike.
        return_residual: Also return each mode's fitting error, the RMS of
            ``IF c - m`` over the RMS of ``m`` (before orthonormalization).

    Returns:
        ``(n_actuators, n_modes)`` commands, and the residuals if requested.
    """
    modes = np.asarray(modes, dtype=float)
    influence = np.asarray(influence_functions, dtype=float)
    if influence.ndim != 2:
        raise ValueError("influence_functions must have shape (n_points, n_actuators).")
    if modes.ndim == 1:
        modes = modes[:, None]
    if modes.ndim != 2 or modes.shape[0] != influence.shape[0]:
        raise ValueError(
            f"modes must have shape ({influence.shape[0]}, n_modes) to match the influence functions."
        )
    if not (np.all(np.isfinite(modes)) and np.all(np.isfinite(influence))):
        raise ValueError("modes and influence_functions must be finite.")
    for value, name in ((rcond, "rcond"), (regularization, "regularization")):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite scalar.")

    u, s, vt = np.linalg.svd(influence, full_matrices=False)
    keep = s > rcond * s[0] if s.size and s[0] > 0 else np.zeros(s.shape, dtype=bool)
    # mean diagonal of IF^T IF = ||IF||_F^2 / n_actuators
    lam = regularization * np.sum(s**2) / max(influence.shape[1], 1)
    gain = np.zeros_like(s)
    gain[keep] = s[keep] / (s[keep] ** 2 + lam)
    commands = vt.T @ (gain[:, None] * (u.T @ modes))

    if return_residual:
        fitted = influence @ commands
        norms = np.linalg.norm(modes, axis=0)
        residual = np.linalg.norm(fitted - modes, axis=0) / np.where(norms > 0, norms, 1.0)

    if orthonormalize and commands.shape[1]:
        surfaces = influence @ commands / np.sqrt(influence.shape[0])
        _, r = np.linalg.qr(surfaces)
        diag = np.abs(np.diag(r))
        if np.any(diag <= 1e-10 * diag.max()):
            raise ValueError("Fitted surfaces are linearly dependent; they cannot be orthonormalized.")
        # Surfaces S = Q R, so commands C R^-1 give surfaces Q (signs kept).
        commands = solve_triangular(r, commands.T, trans="T").T * np.sign(np.diag(r))

    if return_residual:
        return commands, residual
    return commands
