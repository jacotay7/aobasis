"""Von Kármán turbulence statistics at arbitrary points.

This is the one definition of the von Kármán phase covariance in aobasis:
:class:`aobasis.KLBasisGenerator` builds its covariance with it, and other
packages of the AO stack (pyturb, controllers' MMSE priors) can import it
instead of re-deriving it.

For an outer scale ``L0`` and a Fried parameter ``r0`` the phase covariance
at separation ``r`` is

.. math::

    C(r) = \\sigma^2 \\, \\frac{2^{1/6}}{\\Gamma(5/6)} \\, u^{5/6} K_{5/6}(u),
    \\qquad u = 2 \\pi r / L_0,

with the total variance

.. math::

    \\sigma^2 = \\frac{5}{6} \\frac{6.88}{2}
    \\frac{\\Gamma(5/6)}{\\Gamma(1/6) \\, \\pi^{5/3}} \\left(\\frac{L_0}{r_0}\\right)^{5/3}
    \\approx 0.0863 \\left(\\frac{L_0}{r_0}\\right)^{5/3},

so that the structure function ``2 [C(0) - C(r)]`` tends to
``6.88 (r / r0)^(5/3)`` for ``r << L0``. ``6.88`` is the usual rounded
Kolmogorov coefficient ``2 (24/5 Gamma(6/5))^(5/6) = 6.8839``; the variance
is 0.06% below the one computed with the unrounded value.
"""

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv

_UNITS = ("phase", "opd")

# Separations up to this (metres) count as zero and get the variance; the
# Bessel form loses accuracy there.
_ZERO_SEPARATION = 1e-9


def _von_karman_variance(r0: float, L0: float) -> float:
    """Total von Kármán phase variance ``C(0)`` (rad² at r0's wavelength)."""
    A = (5.0 / 6.0) * (6.88 / 2.0) * gamma(5.0 / 6.0) / (gamma(1.0 / 6.0) * np.pi ** (5.0 / 3.0))
    return float(A * (L0 / r0) ** (5.0 / 3.0))


def _phase_covariance_of_distance(r: np.ndarray, r0: float, L0: float) -> np.ndarray:
    """Phase covariance (rad² at r0's wavelength) at separations ``r`` (finite L0)."""
    nu = 5.0 / 6.0
    sigma2 = _von_karman_variance(r0, L0)
    u = 2 * np.pi * r / L0
    out = np.full_like(r, sigma2, dtype=float)
    mask = r > _ZERO_SEPARATION
    out[mask] = sigma2 * 2 ** (1 - nu) / gamma(nu) * u[mask] ** nu * kv(nu, u[mask])
    return out


def _check_parameters(r0, L0, wavelength, units) -> None:
    if not np.isscalar(r0) or not np.isfinite(r0) or r0 <= 0:
        raise ValueError("r0 must be a positive finite scalar.")
    if not np.isscalar(L0) or np.isnan(L0) or L0 <= 0:
        raise ValueError("L0 must be a positive scalar.")
    if np.isinf(L0):
        raise ValueError(
            "The von Karman covariance is infinite for L0=inf (Kolmogorov turbulence); "
            "use a finite outer scale, or the structure function 6.88 (r/r0)^(5/3)."
        )
    if not np.isscalar(wavelength) or not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be a positive finite scalar.")
    if units not in _UNITS:
        raise ValueError(f"units must be one of {_UNITS}, got {units!r}.")


def _to_units(cov: np.ndarray, wavelength: float, units: str) -> np.ndarray:
    if units == "opd":
        cov *= (wavelength / (2 * np.pi)) ** 2
    return cov


def von_karman_covariance_of_distance(
    r,
    *,
    r0: float,
    L0: float,
    wavelength: float = 500e-9,
    units: str = "phase",
) -> np.ndarray:
    """Von Kármán covariance at separations ``r``.

    ``C(0)`` is the total variance ``0.0863 (L0/r0)^(5/3)`` rad² and
    ``2 [C(0) - C(r)]`` tends to ``6.88 (r/r0)^(5/3)`` for ``r << L0`` (see
    :mod:`aobasis.turbulence` for the closed form).

    Args:
        r: Separations in metres (any shape, ``>= 0``).
        r0: Fried parameter in metres, at ``wavelength``.
        L0: Outer scale in metres; must be finite (the Kolmogorov covariance
            is infinite).
        wavelength: Wavelength in metres at which ``r0`` is given.
        units: ``"phase"`` for rad² of phase at ``wavelength`` (phase
            variance scales as ``wavelength^-2``), or ``"opd"`` for m² of
            optical path difference (independent of wavelength).

    Returns:
        Array of the shape of ``r``.
    """
    _check_parameters(r0, L0, wavelength, units)
    r = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(r)) or np.any(r < 0):
        raise ValueError("r must be finite and non-negative.")
    shape = r.shape
    cov = _phase_covariance_of_distance(r.ravel(), r0, L0).reshape(shape)
    return _to_units(cov, wavelength, units)


def _as_points(points, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_points, 2), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite.")
    return points


def von_karman_covariance(
    points_a: np.ndarray,
    points_b: Optional[np.ndarray] = None,
    *,
    r0: float,
    L0: float,
    wavelength: float = 500e-9,
    units: str = "phase",
) -> np.ndarray:
    """Von Kármán turbulence covariance between two sets of points.

    ``cov[i, k]`` is the covariance of the turbulent phase (or OPD) at
    ``points_a[i]`` and ``points_b[k]``; it depends only on their distance
    (:func:`von_karman_covariance_of_distance`). Use it as an MMSE prior,
    e.g. on actuator positions, or to draw correlated samples.

    The function is evaluated once per distinct distance, which is fast on
    regular grids (few distinct distances).

    Args:
        points_a: ``(n_a, 2)`` coordinates ``(x, y)`` in metres.
        points_b: ``(n_b, 2)`` coordinates; by default ``points_a``, and the
            result is then exactly symmetric with the variance on the
            diagonal.
        r0: Fried parameter in metres, at ``wavelength``.
        L0: Outer scale in metres; must be finite.
        wavelength: Wavelength in metres at which ``r0`` is given.
        units: ``"phase"`` for rad² of phase at ``wavelength``, or
            ``"opd"`` for m² of optical path difference.

    Returns:
        ``(n_a, n_b)`` covariance matrix.

    Example:
        >>> import numpy as np, aobasis
        >>> pts = np.array([[0.0, 0.0], [0.1, 0.0]])
        >>> cov = aobasis.von_karman_covariance(pts, r0=0.15, L0=25.0, units="opd")
        >>> cov.shape
        (2, 2)
    """
    _check_parameters(r0, L0, wavelength, units)
    points_a = _as_points(points_a, "points_a")
    if points_b is None:
        distances = pdist(points_a)
    else:
        distances = cdist(points_a, _as_points(points_b, "points_b")).ravel()
    # np.unique(..., return_inverse=True) argsorts all the distances; sorting
    # the values and looking each one up is faster and gives the same indices
    # (every distance is one of the unique values).
    unique = np.unique(distances)
    inverse = np.searchsorted(unique, distances)
    values = _phase_covariance_of_distance(unique, r0, L0)[inverse]
    if points_b is None:
        cov = squareform(values)
        np.fill_diagonal(cov, _von_karman_variance(r0, L0))
    else:
        cov = values.reshape(points_a.shape[0], -1)
    return _to_units(cov, wavelength, units)
