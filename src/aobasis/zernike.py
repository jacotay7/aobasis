import warnings

import numpy as np
from typing import Any, List, Optional, Tuple, Union

from .base import BasisGenerator, RemoveSpec, _check_normalize
from .utils import positions_from_mask

ORDERINGS = ("noll", "ansi", "fringe")

# An index or order: an integer, or an integer array (any shape).
IndexLike = Union[int, np.integer, np.ndarray, List[int]]
IndexPair = Tuple[Any, Any]


def _index_array(value: IndexLike, name: str) -> Tuple[np.ndarray, bool]:
    """``value`` as an int64 array, and whether it was a scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or an array of integers")
    if isinstance(value, (int, np.integer)):
        return np.asarray(value, dtype=np.int64), True
    array = np.asarray(value)
    if array.dtype.kind not in "iu" and array.size:
        raise ValueError(f"{name} must be an integer or an array of integers")
    return array.astype(np.int64), False


def _output(values: Tuple[np.ndarray, ...], scalar: bool):
    """Python ints for scalar input, int64 arrays otherwise."""
    if scalar:
        return tuple(int(v) for v in values) if len(values) > 1 else int(values[0])
    return values if len(values) > 1 else values[0]


def _triangular_row(t: np.ndarray) -> np.ndarray:
    """Largest ``n`` with ``n (n + 1) / 2 <= t``, for integers ``t >= 0``."""
    n = np.floor((np.sqrt(8.0 * t + 1.0) - 1.0) / 2.0).astype(np.int64)
    n += (n + 1) * (n + 2) // 2 <= t  # correct float rounding at very large t
    n -= n * (n + 1) // 2 > t
    return n


def _nm_arrays(n: IndexLike, m: IndexLike) -> Tuple[np.ndarray, np.ndarray, bool]:
    n, n_scalar = _index_array(n, "n")
    m, m_scalar = _index_array(m, "m")
    n, m = np.broadcast_arrays(n, m)
    if np.any((n < 0) | (np.abs(m) > n) | ((n - m) % 2 != 0)):
        raise ValueError("(n, m) must satisfy n >= 0, |m| <= n and n - m even")
    return n, m, n_scalar and m_scalar


def noll_to_nm(j: IndexLike) -> IndexPair:
    """Radial order ``n`` and azimuthal frequency ``m`` of Noll index ``j``.

    Noll, J. Opt. Soc. Am. 66, 207 (1976): ``j = 1`` is piston, and within
    radial order ``n`` the index runs through increasing ``|m|`` (``m = 0``
    first for even ``n``). Each ``|m| > 0`` pair gets the cosine term
    (``m > 0``) on the even ``j`` and the sine term (``m < 0``) on the odd
    ``j``. This is the mapping :class:`ZernikeBasisGenerator` uses for
    ``ordering="noll"``, where ``m > 0`` is ``cos(m theta)``, ``m < 0`` is
    ``sin(|m| theta)`` and ``theta`` is measured from +x toward +y: ``j = 2``
    (1, 1) is tip along +x and ``j = 3`` (1, -1) tilt along +y.

    Args:
        j: Noll index ``>= 1``, or an integer array of them.

    Returns:
        ``(n, m)``: Python ints for an integer ``j``, else int arrays shaped
        like ``j``.

    Example:
        >>> [noll_to_nm(j) for j in range(1, 7)]
        [(0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2)]
    """
    j, scalar = _index_array(j, "Noll index")
    if np.any(j < 1):
        raise ValueError("Noll index must be an integer >= 1")
    n = _triangular_row(j - 1)
    k = j - 1 - n * (n + 1) // 2  # position within radial order n
    abs_m = n % 2 + 2 * ((k + (n + 1) % 2) // 2)
    return _output((n, np.where(j % 2 == 0, abs_m, -abs_m)), scalar)


def nm_to_noll(n: IndexLike, m: IndexLike) -> Any:
    """Noll index of ``(n, m)``, the inverse of :func:`noll_to_nm`.

    ``m > 0`` is the cosine term and ``m < 0`` the sine term. ``n`` and
    ``m`` may be integers or integer arrays (broadcast together); they must
    satisfy ``n >= 0``, ``|m| <= n`` and ``n - m`` even.

    Example:
        >>> nm_to_noll(1, 1), nm_to_noll(1, -1), nm_to_noll(4, 0)
        (2, 3, 11)
    """
    n, m, scalar = _nm_arrays(n, m)
    abs_m = np.abs(m)
    # The |m| pair of order n takes Noll indices n (n + 1) / 2 + |m| (+ 1);
    # m = 0 takes n (n + 1) / 2 + 1.
    j = n * (n + 1) // 2 + abs_m + (m == 0)
    odd = j % 2
    j = np.where(m > 0, j + odd, np.where(m < 0, j + 1 - odd, j))
    return _output((j,), scalar)


def ansi_to_nm(j: IndexLike) -> IndexPair:
    """``(n, m)`` of OSA/ANSI index ``j >= 0`` (``j = (n (n + 2) + m) / 2``).

    ANSI Z80.28 counts from 0 and runs through ``m = -n, -n + 2, ..., n``
    within each radial order: piston, y-tilt (1, -1), x-tilt (1, 1), ...
    ``m > 0`` is the cosine term, as in :func:`noll_to_nm`. Accepts integers
    or integer arrays.
    """
    j, scalar = _index_array(j, "ANSI index")
    if np.any(j < 0):
        raise ValueError("ANSI index must be an integer >= 0")
    n = _triangular_row(j)
    return _output((n, 2 * j - n * (n + 2)), scalar)


def nm_to_ansi(n: IndexLike, m: IndexLike) -> Any:
    """OSA/ANSI index ``(n (n + 2) + m) / 2`` of ``(n, m)``; see :func:`ansi_to_nm`."""
    n, m, scalar = _nm_arrays(n, m)
    return _output(((n * (n + 2) + m) // 2,), scalar)


# The standard 37-term Fringe (University of Arizona) set, Z1 to Z37.
FRINGE_TERMS = 37


def fringe_to_nm(j: IndexLike) -> IndexPair:
    """``(n, m)`` of Fringe index ``1 <= j <= 37``.

    The 37-term University of Arizona set: piston, x-tilt (1, 1), y-tilt
    (1, -1), defocus, 0° and 45° astigmatism, x- and y-coma, primary
    spherical, ..., with Z37 the 12th-order spherical (12, 0). Indices 1-36
    are ``((n + |m|) / 2 + 1)^2 - 2 |m|``, plus 1 for the sine term
    (``m < 0``). Accepts integers or integer arrays.
    """
    j, scalar = _index_array(j, "Fringe index")
    if np.any((j < 1) | (j > FRINGE_TERMS)):
        raise ValueError(f"Fringe index must be an integer in [1, {FRINGE_TERMS}]")
    return _output((_FRINGE_N[j - 1], _FRINGE_M[j - 1]), scalar)


def nm_to_fringe(n: IndexLike, m: IndexLike) -> Any:
    """Fringe index of ``(n, m)``, the inverse of :func:`fringe_to_nm`.

    Raises ``ValueError`` for orders outside the 37-term set.
    """
    n, m, scalar = _nm_arrays(n, m)
    abs_m = np.abs(m)
    z37 = (n == 12) & (m == 0)
    j = np.where(z37, FRINGE_TERMS, ((n + abs_m) // 2 + 1) ** 2 - 2 * abs_m + (m < 0))
    if np.any((j >= FRINGE_TERMS) & ~z37):
        raise ValueError(f"(n, m) is not in the {FRINGE_TERMS}-term Fringe set")
    return _output((j,), scalar)


_FRINGE_PAIRS = (
    (0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, 3), (3, -3),
    (4, 2), (4, -2), (5, 1), (5, -1), (6, 0), (4, 4), (4, -4), (5, 3), (5, -3), (6, 2), (6, -2),
    (7, 1), (7, -1), (8, 0), (5, 5), (5, -5), (6, 4), (6, -4), (7, 3), (7, -3), (8, 2), (8, -2),
    (9, 1), (9, -1), (10, 0), (12, 0),
)
_FRINGE_N = np.array([n for n, _ in _FRINGE_PAIRS], dtype=np.int64)
_FRINGE_M = np.array([m for _, m in _FRINGE_PAIRS], dtype=np.int64)


class ZernikeBasisGenerator(BasisGenerator):
    """
    Generates Zernike polynomials on the actuator grid.

    Args:
        positions: ``(n_actuators, 2)`` actuator coordinates, with the pupil
            centred on the origin.
        pupil_radius: Radius of the unit disk. By default, the largest
            actuator distance from the origin.
        obscuration: Central obstruction as a fraction of the pupil radius
            (``0 <= obscuration < 1``). Above 0 the modes are annular
            Zernike polynomials (Mahajan, J. Opt. Soc. Am. 71, 75, 1981),
            orthonormal over the annulus ``obscuration <= rho <= 1``.
    """

    _PARAMETERS = ("pupil_radius", "obscuration")

    def __init__(self, positions: np.ndarray, pupil_radius: Optional[float] = None, obscuration: float = 0.0):
        super().__init__(positions)
        if pupil_radius is None:
            pupil_radius = float(np.max(np.hypot(self.positions[:, 0], self.positions[:, 1])))
            if pupil_radius <= 0:
                raise ValueError("pupil_radius cannot be inferred when every actuator is at the origin.")
        if not np.isscalar(pupil_radius) or not np.isfinite(pupil_radius) or pupil_radius <= 0:
            raise ValueError("pupil_radius must be a positive finite scalar.")
        if not np.isscalar(obscuration) or not np.isfinite(obscuration) or not 0 <= obscuration < 1:
            raise ValueError("obscuration must be in [0, 1).")
        self.pupil_radius = pupil_radius
        self.obscuration = float(obscuration)
        self._annular_cache = {}

    def _annular_recurrence(self, m: int, count: int):
        """Three-term recurrence of the annular radial functions of order ``m``.

        The annular radial functions are ``A_(m+2k)(rho) = rho^m p_k(rho^2)``
        with ``p_k`` orthonormal on ``[eps^2, 1]`` under
        ``<f, g> = 2 / (1 - eps^2) * integral f g rho drho`` (unit RMS over the
        annulus once the angular factor is included), with positive leading
        coefficients, so ``A(1) > 0`` (Mahajan, J. Opt. Soc. Am. 71, 75,
        1981). The recurrence ``sqrt(b_(k+1)) p_(k+1) = (t - a_k) p_k -
        sqrt(b_k) p_(k-1)`` comes from the Stieltjes procedure on a
        Gauss-Legendre rule that is exact for these polynomials; unlike
        orthogonalizing the circular ``R_n^m``, it stays accurate on narrow
        annuli and at high order.

        Returns ``(p0, alpha, beta)``: ``p_0`` (a constant) and arrays of
        ``a_k`` and ``sqrt(b_(k+1))`` for ``k < count``.
        """
        cached = self._annular_cache.get(m)
        if cached is not None and len(cached[1]) >= count:
            return cached
        count = max(count, 2 * (len(cached[1]) if cached is not None else 0), 8)
        eps = self.obscuration
        nodes, weights = np.polynomial.legendre.leggauss(m + 2 * count + 32)
        rho = eps + (1.0 - eps) * (nodes + 1.0) / 2.0
        weights = weights * (1.0 - eps) / 2.0 * rho ** (2 * m + 1) * 2.0 / (1.0 - eps**2)
        t = rho**2
        p0 = 1.0 / np.sqrt(weights.sum())
        alpha, beta = np.empty(count), np.empty(count)
        p_prev, p = np.zeros_like(t), np.full_like(t, p0)
        b_prev = 0.0
        for k in range(count):
            alpha[k] = np.sum(weights * t * p * p)
            q = (t - alpha[k]) * p - b_prev * p_prev
            beta[k] = np.sqrt(np.sum(weights * q * q))
            p_prev, p, b_prev = p, q / beta[k], beta[k]
        self._annular_cache[m] = (p0, alpha, beta)
        return self._annular_cache[m]

    def _radial_function(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Unit-RMS radial part (Noll factor included), circular or annular."""
        if self.obscuration == 0:
            return np.sqrt(n + 1) * self._zernike_radial(n, m, rho)
        k_max = (n - m) // 2
        p0, alpha, beta = self._annular_recurrence(m, k_max + 1)
        t = np.asarray(rho, dtype=float) ** 2
        p_prev, p = np.zeros_like(t), np.full_like(t, p0)
        for k in range(k_max):
            p_prev, p = p, ((t - alpha[k]) * p - (beta[k - 1] if k else 0.0) * p_prev) / beta[k]
        return rho**m * p

    def _zernike_radial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Compute radial Zernike polynomial R_n^m(rho), m >= 0.

        Uses R_n^m(rho) = (-1)^k rho^m P_k^(m,0)(1 - 2 rho^2), k = (n - m) / 2,
        with the three-term Jacobi recurrence. The explicit factorial sum
        cancels catastrophically in float64 from n ~ 46 on.
        """
        rho = np.asarray(rho, dtype=float)
        k_max = (n - m) // 2
        a = float(m)
        x = 1.0 - 2.0 * rho**2
        p_prev = np.ones_like(x)
        p = p_prev if k_max == 0 else (a + 1.0) + (a + 2.0) * (x - 1.0) / 2.0
        for k in range(2, k_max + 1):
            c = 2 * k + a
            p, p_prev = (
                (c - 1) * (c * (c - 2) * x + a * a) * p
                - 2 * (k + a - 1) * (k - 1) * c * p_prev
            ) / (2 * k * (k + a) * (c - 2)), p
        return (-1.0) ** k_max * rho**m * p

    def generate(
        self,
        n_modes: int,
        ignore_piston: bool = False,
        orthonormalize: bool = False,
        remove: RemoveSpec = None,
        normalize: Optional[str] = None,
        ordering: str = "noll",
        check_rank: bool = True,
    ) -> np.ndarray:
        """
        Generate Noll-normalized Zernike modes, in Noll order by default.

        With ``ordering="noll"``, j=1 is piston, j=2 tip (x), j=3 tilt (y),
        j=4 defocus, j=5/6 the oblique/vertical astigmatism, and so on: even j
        are cosine terms and odd j sine terms (Noll, J. Opt. Soc. Am. 66, 207,
        1976). Each mode carries the Noll factor ``sqrt(n+1)`` (times
        ``sqrt(2)`` for m != 0), so it has unit RMS over the continuous unit
        disk (over the annulus with an ``obscuration``). Cosine terms are
        ``m > 0`` and sine terms ``m < 0`` in every ordering.

        Actuators outside ``pupil_radius`` (or inside the obscuration) get
        the polynomial's value at their true radius (it is not clipped), with
        a warning.

        Args:
            n_modes: Number of modes, at most the number of actuators minus
                the number of removed modes.
            ignore_piston: Remove piston: j=1 is skipped and the mean is
                subtracted from every other mode (sampled Zernikes are not
                zero-mean on a discrete grid).
            orthonormalize: Gram-Schmidt the modes in order so they are
                orthonormal on the actuator grid (the sampled Zernikes are not).
            remove: Further modes to project out of every Zernike, e.g.
                ``"tiptilt"`` or an ``(n_actuators, k)`` array (see
                :func:`aobasis.removal_basis`). Zernikes that lie inside the
                removed modes (tip and tilt for ``"tiptilt"``) are skipped.
            normalize: Scale each mode to unit ``"rms"``, ``"l2"``,
                ``"peak"`` or ``"pv"`` after everything else (see
                :func:`aobasis.normalize_modes`). ``None`` keeps the
                generator's own scale. ``"peak"`` gives the Fringe
                convention's scaling (each polynomial is 1 at the rim).
            ordering: ``"noll"``; ``"ansi"`` (OSA/ANSI Z80.28 single index
                ``j = (n (n + 2) + m) / 2`` from 0: piston, y-tilt, x-tilt,
                oblique astigmatism, defocus, ...); or ``"fringe"``
                (the 37-term University of Arizona set: piston, x-tilt,
                y-tilt, defocus, 0° astigmatism, 45° astigmatism, x-coma,
                y-coma, spherical, ..., Z37 = 12th-order spherical).
                :func:`aobasis.noll_to_nm`, :func:`aobasis.ansi_to_nm` and
                :func:`aobasis.fringe_to_nm` give each index's ``(n, m)``.
            check_rank: Warn if the modes are linearly dependent on the
                actuators. ``False`` skips the check, which is a large part
                of the cost for big bases (a column-pivoted QR of the modes);
                the modes are identical either way.
        """
        self._record_options(
            n_modes=n_modes,
            ignore_piston=ignore_piston,
            orthonormalize=orthonormalize,
            remove=remove,
            normalize=normalize,
            ordering=ordering,
        )
        _check_normalize(normalize)
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}.")
        removed = self._removed_subspace(remove, ignore_piston)
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators - removed.shape[1])
        if n_modes == 0:
            self.modes = np.zeros((self.n_actuators, 0), dtype=float)
            return self.modes

        x = self.positions[:, 0] / self.pupil_radius
        y = self.positions[:, 1] / self.pupil_radius
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        outside = int(np.count_nonzero(rho > 1.0 + 1e-9))
        if outside:
            warnings.warn(
                f"{outside} actuators lie outside pupil_radius={self.pupil_radius}; "
                "their Zernike values are extrapolated.",
                RuntimeWarning,
                stacklevel=2,
            )
        obscured = int(np.count_nonzero(rho < self.obscuration - 1e-9))
        if obscured:
            warnings.warn(
                f"{obscured} actuators lie inside the obscuration ({self.obscuration} of the radius); "
                "their annular Zernike values are extrapolated.",
                RuntimeWarning,
                stacklevel=2,
            )
        index_to_nm = {"noll": self._noll_to_nm, "ansi": self._ansi_to_nm, "fringe": self._fringe_to_nm}[ordering]
        first = 0 if ordering == "ansi" else 1
        n_available = self._FRINGE_TERMS if ordering == "fringe" else None

        def candidates(start: int, count: int) -> np.ndarray:
            columns = []
            if n_available is not None:
                count = min(count, n_available - start)
            for j in range(start + first, start + count + first):
                n, m = index_to_nm(j)
                radial = self._radial_function(n, abs(m), rho)
                if m > 0:
                    columns.append(np.sqrt(2.0) * radial * np.cos(m * theta))
                elif m < 0:
                    columns.append(np.sqrt(2.0) * radial * np.sin(-m * theta))
                else:
                    columns.append(radial)
            return np.column_stack(columns)

        if n_available is not None and n_modes > n_available:
            raise ValueError(f"The Fringe ordering defines {n_available} terms; requested {n_modes}.")
        modes = self._take_outside(candidates, n_modes, removed, n_available=n_available)
        return self._finish(
            modes, orthonormalize=orthonormalize, removed=removed, normalize=normalize, check_rank=check_rank
        )

    # The private names predate the public functions; kept for callers.
    _noll_to_nm = staticmethod(noll_to_nm)
    _ansi_to_nm = staticmethod(ansi_to_nm)
    _fringe_to_nm = staticmethod(fringe_to_nm)
    _FRINGE_TERMS = FRINGE_TERMS


def zernike_modes_on_mask(
    mask: np.ndarray,
    n_modes: int,
    pupil_radius: Optional[float] = None,
    obscuration: float = 0.0,
    **options: Any,
) -> np.ndarray:
    """Zernike modes on the pixels of a 2-D mask, as a stack of images.

    A shortcut for :func:`aobasis.positions_from_mask` (with unit pitch),
    :class:`ZernikeBasisGenerator` and scattering the modes back onto the
    grid. Pixel centres sit at ``i - (size - 1) / 2`` pixels from the array
    centre along each axis; x runs along the columns (axis 1) and y along
    the rows (axis 0), so Noll ``j = 2`` (tip) varies along the columns and
    ``j = 3`` (tilt) along the rows.

    Args:
        mask: 2-D array; the modes are evaluated on its nonzero pixels.
        n_modes: Number of modes, at most the number of mask pixels.
        pupil_radius: Radius of the unit disk in pixels. By default, the
            largest distance of a mask pixel centre from the array centre.
            For a pupil sampled by ``N`` pixels across, ``N / 2`` puts the
            unit circle on the pupil edge.
        obscuration: Central obstruction as a fraction of ``pupil_radius``
            (annular Zernikes above 0).
        **options: Passed to :meth:`ZernikeBasisGenerator.generate`
            (``ordering``, ``ignore_piston``, ``orthonormalize``, ``remove``,
            ``normalize``, ``check_rank``). Arrays given to ``remove`` list
            the mask pixels in ``mask[mask != 0]`` (row-major) order.

    Returns:
        ``(n_modes, ny, nx)`` float array, zero outside the mask.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}.")
    inside = mask != 0
    if not inside.any():
        raise ValueError("mask has no nonzero pixels.")
    positions = positions_from_mask(inside, pitch=1.0)
    generator = ZernikeBasisGenerator(positions, pupil_radius=pupil_radius, obscuration=obscuration)
    modes = generator.generate(n_modes, **options)
    images = np.zeros((modes.shape[1],) + mask.shape)
    images[:, inside] = modes.T
    return images
