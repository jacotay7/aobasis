import warnings

import numpy as np
from typing import Tuple
from .base import BasisGenerator

class ZernikeBasisGenerator(BasisGenerator):
    """
    Generates Zernike polynomials on the actuator grid.
    """
    
    def __init__(self, positions: np.ndarray, pupil_radius: float):
        super().__init__(positions)
        if not np.isscalar(pupil_radius) or not np.isfinite(pupil_radius) or pupil_radius <= 0:
            raise ValueError("pupil_radius must be a positive finite scalar.")
        self.pupil_radius = pupil_radius

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

    def _zernike(self, n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute Zernike polynomial Z_n^m(rho, theta)."""
        R = self._zernike_radial(n, abs(m), rho)
        if m >= 0:
            return R * np.cos(m * theta)
        else:
            return R * np.sin(abs(m) * theta)

    def generate(
        self, n_modes: int, ignore_piston: bool = False, orthonormalize: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Generate Noll-normalized Zernike modes in Noll order.

        j=1 is piston, j=2 tip (x), j=3 tilt (y), j=4 defocus, j=5/6 the
        oblique/vertical astigmatism, and so on: even j are cosine terms and
        odd j sine terms (Noll, J. Opt. Soc. Am. 66, 207, 1976). Each mode
        carries the Noll factor ``sqrt(n+1)`` (times ``sqrt(2)`` for m != 0),
        so it has unit RMS over the continuous unit disk.

        Actuators outside ``pupil_radius`` get the polynomial's value at their
        true radius (it is not clipped), with a warning.

        Args:
            n_modes: Number of modes, at most the number of actuators.
            ignore_piston: Start at j=2 instead of j=1.
            orthonormalize: Gram-Schmidt the modes in order so they are
                orthonormal on the actuator grid (the sampled Zernikes are not).
        """
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators)
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

        first_j = 2 if ignore_piston else 1
        modes = []
        for j in range(first_j, first_j + n_modes):
            n, m = self._noll_to_nm(j)
            norm = np.sqrt(n + 1) * (np.sqrt(2.0) if m != 0 else 1.0)
            modes.append(norm * self._zernike(n, m, rho, theta))

        return self._finish(np.column_stack(modes), orthonormalize=orthonormalize)

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
