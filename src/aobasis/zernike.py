import warnings

import numpy as np
from typing import Tuple, Optional

from .base import BasisGenerator, RemoveSpec, _check_normalize

ORDERINGS = ("noll", "ansi", "fringe")


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
        return self._finish(modes, orthonormalize=orthonormalize, removed=removed, normalize=normalize)

    @staticmethod
    def _ansi_to_nm(j: int) -> Tuple[int, int]:
        """OSA/ANSI index ``j >= 0`` to ``(n, m)``: ``j = (n (n + 2) + m) / 2``."""
        if isinstance(j, bool) or not isinstance(j, (int, np.integer)) or j < 0:
            raise ValueError("ANSI index must be an integer >= 0")
        j = int(j)
        n = 0
        while (n + 1) * (n + 2) // 2 <= j:
            n += 1
        return n, 2 * j - n * (n + 2)

    @staticmethod
    def _fringe_index(n: int, m: int) -> int:
        """Fringe (University of Arizona) index of ``(n, m)``, from 1."""
        return ((n + abs(m)) // 2 + 1) ** 2 - 2 * abs(m) + (1 if m < 0 else 0)

    # The standard 37-term Fringe set: indices 1-36 follow _fringe_index, and
    # Z37 is the 12th-order spherical term.
    _FRINGE_TERMS = 37

    @classmethod
    def _fringe_to_nm(cls, j: int) -> Tuple[int, int]:
        """Fringe index ``1 <= j <= 37`` to ``(n, m)``."""
        if isinstance(j, bool) or not isinstance(j, (int, np.integer)) or not 1 <= j <= cls._FRINGE_TERMS:
            raise ValueError(f"Fringe index must be an integer in [1, {cls._FRINGE_TERMS}]")
        if j == 37:
            return 12, 0
        pairs = [(n, m) for n in range(11) for m in range(-n, n + 1, 2)]
        return next(p for p in pairs if cls._fringe_index(*p) == j)

    @staticmethod
    def _noll_to_nm(j: int) -> Tuple[int, int]:
        """
        Convert Noll index j to radial order n and azimuthal frequency m.

        Within radial order n, |m| increases (m = 0 first for even n), and each
        |m| > 0 pair gets the cosine term (m > 0) on the even j and the sine
        term (m < 0) on the odd j. Noll, J. Opt. Soc. Am. 66, 207 (1976).
        """
        if isinstance(j, bool) or not isinstance(j, (int, np.integer)) or j < 1:
            raise ValueError("Noll index must be an integer >= 1")
        j = int(j)

        n = 0
        while (n + 1) * (n + 2) // 2 < j:
            n += 1

        # Walk the |m| values of order n in Noll order.
        next_j = n * (n + 1) // 2 + 1
        for abs_m in range(n % 2, n + 1, 2):
            if abs_m == 0:
                if j == next_j:
                    return n, 0
                next_j += 1
                continue
            if j in (next_j, next_j + 1):
                return n, abs_m if j % 2 == 0 else -abs_m
            next_j += 2
        raise AssertionError("unreachable")  # pragma: no cover
