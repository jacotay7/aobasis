"""Karhunen-Loève modes of a deformable mirror (double diagonalization)."""

from typing import Optional

import numpy as np
from scipy.linalg import eigh

from .base import BasisGenerator, RemoveSpec, _check_normalize, _validate_positions_array, mode_scales, removal_basis
from .kl import KLBasisGenerator, _canonical_eigenvectors, _cluster_end, _reference_vectors


class DMKLBasisGenerator(BasisGenerator):
    """
    KL modes of a DM from its influence functions (Gendron 1995).

    :class:`KLBasisGenerator` diagonalizes the phase covariance at the
    actuator positions. This generator works with the DM surface instead:

    1. The geometric covariance ``Delta = IF^T IF / n_points`` is whitened,
       ``M = U S^(-1/2)``, so the surfaces ``IF M`` are orthonormal over the
       pupil (unit RMS, mutually orthogonal). Directions with
       ``S < rcond * max(S)`` (actuators the pupil barely sees) are dropped.
    2. The covariance of the turbulence's projection on those surfaces,
       ``M^T IF^T C IF M / n_points^2`` with ``C`` the Von Kármán phase
       covariance between the points, is diagonalized: ``V Lambda V^T``.

    The modes are the commands ``B = M V``. Their DM surfaces are orthonormal
    over the pupil, their coefficients are statistically independent with
    variance ``eigenvalues`` (rad² at ``wavelength``), and the first ``k``
    capture as much turbulence as any ``k`` DM surfaces can.

    Args:
        actuator_positions: ``(n_actuators, 2)`` actuator coordinates (used
            for the sign and degenerate-rotation convention).
        points: ``(n_points, 2)`` pupil points where the influence functions
            are sampled, e.g. from :func:`aobasis.make_pupil_points`.
        influence_functions: ``(n_points, n_actuators)`` matrix.
        fried_parameter, outer_scale, r0_wavelength, wavelength: As for
            :class:`KLBasisGenerator` (``outer_scale=np.inf`` is Kolmogorov).
        rcond: Relative cutoff on the eigenvalues of ``Delta``.
    """

    def __init__(
        self,
        actuator_positions: np.ndarray,
        points: np.ndarray,
        influence_functions: np.ndarray,
        fried_parameter: float = 0.16,
        outer_scale: float = 30.0,
        r0_wavelength: float = 500e-9,
        wavelength: Optional[float] = None,
        rcond: float = 1e-6,
    ):
        super().__init__(actuator_positions)
        self.points = _validate_positions_array(points)
        influence = np.asarray(influence_functions, dtype=float)
        if influence.shape != (self.points.shape[0], self.n_actuators):
            raise ValueError(
                f"influence_functions must have shape ({self.points.shape[0]}, {self.n_actuators}), "
                f"got {influence.shape}."
            )
        if not np.all(np.isfinite(influence)):
            raise ValueError("influence_functions must be finite.")
        if not np.isscalar(rcond) or not np.isfinite(rcond) or rcond < 0:
            raise ValueError("rcond must be a non-negative finite scalar.")
        self.influence_functions = influence
        self.rcond = float(rcond)
        # The turbulence model lives on the pupil points.
        self._turbulence = KLBasisGenerator(
            self.points,
            fried_parameter=fried_parameter,
            outer_scale=outer_scale,
            r0_wavelength=r0_wavelength,
            wavelength=wavelength,
        )
        self.eigenvalues: Optional[np.ndarray] = None

    def _parameters(self):
        params = self._turbulence._parameters()
        params.pop("use_gpu")
        params.update(rcond=self.rcond, n_points=int(self.points.shape[0]))
        return params

    @property
    def surfaces(self) -> np.ndarray:
        """DM surfaces of the generated modes, ``(n_points, n_modes)``."""
        if self.modes is None:
            raise ValueError("No modes generated yet. Call generate() first.")
        return self.influence_functions @ self.modes

    def generate(
        self,
        n_modes: int,
        ignore_piston: bool = False,
        remove: RemoveSpec = None,
        normalize: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate DM KL modes as commands, decreasing variance.

        Args:
            n_modes: Number of modes, at most the number of DM directions the
                pupil sees minus the number of removed modes.
            ignore_piston: Remove piston from the turbulence and keep every
                mode's surface zero-mean over the pupil.
            remove: Further pupil modes (names, or arrays over ``points``)
                to remove likewise (see :func:`aobasis.removal_basis`).
            normalize: ``None`` keeps unit-RMS surfaces; ``"rms"``, ``"l2"``,
                ``"peak"`` or ``"pv"`` scale the commands instead, with
                ``eigenvalues`` rescaled as for :class:`KLBasisGenerator`.
        """
        self._record_options(
            n_modes=n_modes, ignore_piston=ignore_piston, remove=remove, normalize=normalize
        )
        _check_normalize(normalize)
        turbulence = self._turbulence
        n_points = self.points.shape[0]
        removed = removal_basis(self.points, remove, ignore_piston=ignore_piston)
        if turbulence.kolmogorov:
            piston = np.ones(n_points)
            if np.linalg.norm(piston - removed @ (removed.T @ piston)) > 1e-8 * np.sqrt(n_points):
                raise ValueError(
                    "Kolmogorov turbulence (outer_scale=inf) has infinite piston variance; "
                    "pass ignore_piston=True."
                )

        # 1. Whiten the geometric covariance.
        influence = self.influence_functions
        delta = influence.T @ influence / n_points
        s, u = eigh(delta)
        keep = s > self.rcond * s.max() if s.size and s.max() > 0 else np.zeros(s.shape, dtype=bool)
        s, u = s[keep], u[:, keep]
        whiten = u / np.sqrt(s)  # M: surfaces IF M are orthonormal over the pupil
        whitened_if = influence @ whiten  # (n_points, r)

        # Removed pupil modes, as coefficients on the orthonormal surfaces.
        blocked = whitened_if.T @ removed / n_points
        blocked_basis = removal_basis(np.zeros((blocked.shape[0], 2)), blocked) if removed.shape[1] else blocked
        n_modes = self._validate_n_modes(n_modes, max_modes=whiten.shape[1] - blocked_basis.shape[1])
        if n_modes == 0:
            self.eigenvalues = np.array([], dtype=float)
            self.modes = np.zeros((self.n_actuators, 0), dtype=float)
            return self.modes

        # 2. Covariance of the turbulence's projection on the surfaces.
        cov = turbulence._von_karman_covariance_cpu()
        if removed.shape[1]:
            cu = cov @ removed
            cov = cov - removed @ cu.T - cu @ removed.T + removed @ ((removed.T @ cu) @ removed.T)
        coeff_cov = whitened_if.T @ cov @ whitened_if / n_points**2
        if blocked_basis.shape[1]:
            cb = coeff_cov @ blocked_basis
            coeff_cov = (
                coeff_cov
                - blocked_basis @ cb.T
                - cb @ blocked_basis.T
                + blocked_basis @ ((blocked_basis.T @ cb) @ blocked_basis.T)
            )
        coeff_cov = 0.5 * (coeff_cov + coeff_cov.T)

        values, vectors = eigh(coeff_cov)
        values, vectors = values[::-1], vectors[:, ::-1]
        n_vectors = _cluster_end(values, n_modes - 1)
        # Reference vectors of the actuator geometry, in whitened coordinates
        # (M^T Delta R = S^(1/2) U^T R), so the convention depends only on
        # each eigenspace of surfaces.
        def references(d: int) -> np.ndarray:
            return (u.T @ _reference_vectors(self.positions, d)) * np.sqrt(s)[:, None]

        vectors = _canonical_eigenvectors(values, vectors[:, :n_vectors], self.positions, n_modes, references)
        modes = whiten @ vectors
        eigenvalues = values[:n_modes] * (turbulence.r0_wavelength / turbulence.wavelength) ** 2
        if normalize is not None:
            scale = mode_scales(modes, normalize)
            scale = np.where(scale > 0, scale, 1.0)
            modes = modes / scale
            eigenvalues = eigenvalues * scale**2
        self.eigenvalues = eigenvalues
        self.modes = modes
        return self.modes
