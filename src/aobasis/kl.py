import warnings

import numpy as np
from typing import Optional
from scipy.special import kv, gamma
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from .base import BasisGenerator, RemoveSpec, _check_normalize, mode_scales

# CuPy is optional and slow to import, so it is loaded on first GPU use.
_CUPY_BACKEND = None  # (cupy module, K_{5/6} kernel) once loaded, False if unavailable

# K_{5/6}(z) for z > 0, float64, by Temme's series (z < 2) and Steed's
# continued fraction (z >= 2) for K_mu, mu = 5/6 - 1 = -1/6, then one upward
# recurrence step (Numerical Recipes, 2nd ed., section 6.7, bessik). Agrees
# with scipy.special.kv to ~1e-14 relative for all z.
_KV56_MU = 5.0 / 6.0 - 1.0
_KV56_GAMPL = 1.0 / float(gamma(1.0 + _KV56_MU))
_KV56_GAMMI = 1.0 / float(gamma(1.0 - _KV56_MU))
_KV56_KERNEL_PREAMBLE = f'''
const double KV_MU = {_KV56_MU!r};
const double KV_GAMPL = {_KV56_GAMPL!r};
const double KV_GAMMI = {_KV56_GAMMI!r};
const double KV_GAM1 = {(_KV56_GAMMI - _KV56_GAMPL) / (2.0 * _KV56_MU)!r};
const double KV_GAM2 = {(_KV56_GAMMI + _KV56_GAMPL) / 2.0!r};
'''
_KV56_KERNEL_SOURCE = '''
const double EPS = 1e-16;
const double mu2 = KV_MU * KV_MU;
if (!(z > 0.0)) {  // only reached for r = 0, which the caller overwrites
    K = 0.0;
    return;
}
double xi = 1.0 / z;
double rk1;
if (z < 2.0) {
    double x2 = 0.5 * z;
    double pimu = M_PI * KV_MU;
    double fact = pimu / sin(pimu);
    double d = -log(x2);
    double e = KV_MU * d;
    double fact2 = fabs(e) > EPS ? sinh(e) / e : 1.0;
    double ff = fact * (KV_GAM1 * cosh(e) + KV_GAM2 * fact2 * d);
    double sum = ff;
    e = exp(e);
    double p = 0.5 * e / KV_GAMPL;
    double q = 0.5 / (e * KV_GAMMI);
    double c = 1.0;
    d = x2 * x2;
    double sum1 = p;
    for (int i = 1; i < 500; ++i) {
        ff = (i * ff + p + q) / (i * i - mu2);
        c *= d / i;
        p /= (i - KV_MU);
        q /= (i + KV_MU);
        double del = c * ff;
        sum += del;
        sum1 += c * (p - i * ff);
        if (fabs(del) < fabs(sum) * EPS) break;
    }
    rk1 = sum1 * 2.0 * xi;
} else {
    double b = 2.0 * (1.0 + z);
    double d = 1.0 / b;
    double h = d, delh = d;
    double q1 = 0.0, q2 = 1.0;
    double a1 = 0.25 - mu2;
    double q = a1, c = a1, a = -a1;
    double s = 1.0 + q * delh;
    for (int i = 2; i < 500; ++i) {
        a -= 2 * (i - 1);
        c = -a * c / i;
        double qnew = (q1 - b * q2) / a;
        q1 = q2;
        q2 = qnew;
        q += c * qnew;
        b += 2.0;
        d = 1.0 / (b + a * d);
        delh = (b * d - 1.0) * delh;
        h += delh;
        double dels = q * delh;
        s += dels;
        if (fabs(dels / s) < EPS) break;
    }
    h = a1 * h;
    double rkmu = sqrt(M_PI / (2.0 * z)) * exp(-z) / s;
    rk1 = rkmu * (KV_MU + z + 0.5 - h) * xi;
}
K = rk1;  // K_{mu + 1} = K_{5/6}
'''


def _load_cupy():
    """Return ``(cupy, kv56_kernel)``, or ``None`` if CuPy is not installed."""
    global _CUPY_BACKEND
    if _CUPY_BACKEND is None:
        try:
            import cupy
        except ImportError:
            _CUPY_BACKEND = False
        else:
            kernel = cupy.ElementwiseKernel(
                'float64 z',
                'float64 K',
                _KV56_KERNEL_SOURCE,
                name='kv56_kernel_float64',
                preamble=_KV56_KERNEL_PREAMBLE,
            )
            _CUPY_BACKEND = (cupy, kernel)
    return _CUPY_BACKEND or None


# Eigenvalues closer than this (relative) are treated as one degenerate cluster.
_DEGENERATE_RTOL = 1e-8


def _reference_vectors(positions: np.ndarray, count: int) -> np.ndarray:
    """``count`` fixed pseudo-random functions of the actuator geometry.

    They depend on positions (centred and scaled to unit RMS radius), not on
    actuator order, so permuting actuators permutes the result.
    """
    centred = positions - positions.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centred**2, axis=1)))
    x, y = (centred / (scale if scale > 0 else 1.0)).T
    columns = []
    for k in range(count):
        phase = np.sin((12.9898 + 1.618 * k) * x + (78.233 + 2.718 * k) * y + 0.5 + 0.7 * k)
        columns.append(np.mod(phase * 43758.5453, 1.0) - 0.5)
    return np.column_stack(columns)


def _cluster_end(eigenvalues: np.ndarray, start: int) -> int:
    """End (exclusive) of the cluster of eigenvalues equal to ``eigenvalues[start]``."""
    stop = start + 1
    while stop < len(eigenvalues) and abs(eigenvalues[stop] - eigenvalues[start]) <= _DEGENERATE_RTOL * abs(
        eigenvalues[start]
    ):
        stop += 1
    return stop


def _canonical_eigenvectors(
    eigenvalues: np.ndarray, vectors: np.ndarray, positions: np.ndarray, n_keep: int
) -> np.ndarray:
    """First ``n_keep`` eigenvectors, with fixed signs and rotations.

    ``eigenvalues`` are sorted in decreasing order and ``vectors`` holds at
    least every eigenvector of the clusters that the first ``n_keep`` touch,
    so a cluster cut by ``n_keep`` is fixed before it is truncated.

    Within a cluster of (near-)equal eigenvalues spanning ``V``, the modes
    become the in-order orthonormalization of ``V V^T R`` for fixed reference
    vectors ``R``, each with a positive projection on its reference. That
    depends only on the eigenspace, so CPU, GPU and different LAPACKs give
    the same modes. A cluster of one is just a sign choice.
    """
    vectors = np.array(vectors, dtype=float, copy=True)
    start = 0
    while start < n_keep:
        stop = min(_cluster_end(eigenvalues, start), vectors.shape[1])
        block = vectors[:, start:stop]
        refs = _reference_vectors(positions, stop - start)
        q, r = np.linalg.qr(block @ (block.T @ refs))
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        vectors[:, start:stop] = q * signs
        start = stop
    return vectors[:, :n_keep]


# Ask LAPACK for only the leading eigenpairs when fewer than this fraction of
# them are needed; above it the full solver is faster.
_SUBSET_FRACTION = 0.2


def _top_eigenpairs(cov: np.ndarray, n_modes: int):
    """Leading eigenpairs of ``cov`` in decreasing order (CPU).

    Returns at least every eigenpair of the degenerate cluster that eigenpair
    ``n_modes - 1`` belongs to, so it can be canonicalized as a whole.
    """
    n = cov.shape[0]
    count = n_modes + 8
    while count < _SUBSET_FRACTION * n:
        values, vectors = eigh(cov, subset_by_index=[n - count, n - 1])
        values, vectors = values[::-1], vectors[:, ::-1]
        if _cluster_end(values, n_modes - 1) < count:
            return values, vectors
        count *= 2
    values, vectors = eigh(cov)
    return values[::-1], vectors[:, ::-1]


class KLBasisGenerator(BasisGenerator):
    """
    Generates Karhunen-Loève modes based on Von Kármán statistics.

    The modes are the eigenvectors of the Von Kármán phase covariance between
    the actuator positions, sorted by decreasing variance (``eigenvalues``).
    The covariance is sampled at the actuator positions themselves: the modes
    are not fitted to DM influence functions.

    Units: ``positions``, ``fried_parameter`` and ``outer_scale`` share one
    length unit (metres). ``eigenvalues`` are phase variances in rad² at
    ``wavelength`` (by default ``r0_wavelength``, the wavelength at which
    ``fried_parameter`` is given). ``outer_scale`` changes the modes;
    ``fried_parameter`` and the wavelengths only scale the covariance, so
    they change ``eigenvalues`` but not the modes.

    Args:
        positions: ``(n_actuators, 2)`` actuator coordinates.
        fried_parameter: r0 at ``r0_wavelength``.
        outer_scale: L0. ``np.inf`` gives Kolmogorov turbulence, whose piston
            variance is infinite: ``generate`` then needs ``ignore_piston``
            and diagonalizes ``-1/2 P D P`` with the structure function
            ``D(r) = 6.88 (r / r0)^(5/3)``.
        use_gpu: Build and diagonalize the covariance with CuPy (falls back
            to the CPU with a warning when CuPy is missing).
        r0_wavelength: Wavelength of ``fried_parameter`` (default 500 nm).
        wavelength: Wavelength at which ``eigenvalues`` are reported;
            phase variance scales as ``(r0_wavelength / wavelength)^2``.
    """

    def __init__(
        self,
        positions: np.ndarray,
        fried_parameter: float = 0.16,
        outer_scale: float = 30.0,
        use_gpu: bool = False,
        r0_wavelength: float = 500e-9,
        wavelength: Optional[float] = None,
    ):
        super().__init__(positions)
        if not np.isscalar(fried_parameter) or not np.isfinite(fried_parameter) or fried_parameter <= 0:
            raise ValueError("fried_parameter must be a positive finite scalar.")
        if not np.isscalar(outer_scale) or np.isnan(outer_scale) or outer_scale <= 0:
            raise ValueError("outer_scale must be positive (np.inf for Kolmogorov turbulence).")
        for value, name in ((r0_wavelength, "r0_wavelength"), (wavelength, "wavelength")):
            if value is not None and (not np.isscalar(value) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a positive finite scalar.")
        self.fried_parameter = fried_parameter
        self.outer_scale = outer_scale
        self.r0_wavelength = float(r0_wavelength)
        self.wavelength = float(wavelength) if wavelength is not None else self.r0_wavelength
        self.eigenvalues = None
        self.use_gpu = use_gpu

        if self.use_gpu and _load_cupy() is None:
            warnings.warn("CuPy not found; KLBasisGenerator falls back to the CPU.", RuntimeWarning, stacklevel=2)
            self.use_gpu = False

    @property
    def kolmogorov(self) -> bool:
        """True for an infinite outer scale."""
        return bool(np.isinf(self.outer_scale))

    def _sigma2(self) -> float:
        """Von Kármán phase variance (rad² at r0_wavelength)."""
        A = (5.0/6.0) * (6.88/2.0) * gamma(5.0/6.0) / (gamma(1.0/6.0) * np.pi**(5.0/3.0))
        return float(A * (self.outer_scale / self.fried_parameter)**(5.0/3.0))

    def _von_karman_covariance(self) -> np.ndarray:
        """Compute the Von Karman phase covariance matrix."""
        if self.use_gpu:
            return self._von_karman_covariance_gpu()
        else:
            return self._von_karman_covariance_cpu()

    def _covariance_of_distance(self, r: np.ndarray) -> np.ndarray:
        """Phase covariance (rad² at r0_wavelength) at separations ``r`` > 0.

        For Kolmogorov turbulence this is ``-D(r) / 2``, which equals the
        covariance up to a constant that piston removal cancels.
        """
        r0 = self.fried_parameter
        if self.kolmogorov:
            return -0.5 * 6.88 * (r / r0) ** (5.0 / 3.0)
        nu = 5.0 / 6.0
        u = 2 * np.pi * r / self.outer_scale
        out = np.full_like(r, self._sigma2(), dtype=float)
        mask = r > 1e-9
        out[mask] = self._sigma2() * 2 ** (1 - nu) / gamma(nu) * u[mask] ** nu * kv(nu, u[mask])
        return out

    def _von_karman_covariance_cpu(self) -> np.ndarray:
        """Compute the Von Karman phase covariance matrix on CPU.

        The covariance depends only on distance, and actuator grids have few
        distinct distances, so it is evaluated once per distinct distance.
        """
        distances = pdist(self.positions)
        unique, inverse = np.unique(distances, return_inverse=True)
        cov = squareform(self._covariance_of_distance(unique)[inverse])
        np.fill_diagonal(cov, 0.0 if self.kolmogorov else self._sigma2())
        return cov

    def _von_karman_covariance_gpu(self):
        """Compute the Von Karman phase covariance matrix on GPU."""
        cp, kv56 = _load_cupy()
        positions_gpu = cp.asarray(self.positions, dtype=cp.float64)
        diffs = positions_gpu[:, None, :] - positions_gpu[None, :, :]
        r = cp.linalg.norm(diffs, axis=-1)

        if self.kolmogorov:
            return -0.5 * 6.88 * (r / self.fried_parameter) ** (5.0 / 3.0)

        nu = 5.0 / 6.0
        sigma2 = self._sigma2()
        u = 2 * cp.pi * r / self.outer_scale
        kv_values = cp.zeros_like(u, dtype=cp.float64)
        kv56(u, kv_values)
        cov = sigma2 * (2 ** (1 - nu) / gamma(nu)) * (u**nu) * kv_values
        # Zero and very small distances (the diagonal, coincident actuators)
        return cp.where(r <= 1e-9, sigma2, cov)

    def generate(
        self,
        n_modes: int,
        ignore_piston: bool = False,
        orthonormalize: bool = False,
        remove: RemoveSpec = None,
        normalize: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate KL modes (orthonormal columns, decreasing variance).

        Eigenvectors are only defined up to sign, and up to rotation where
        eigenvalues repeat (common on symmetric pupils). The modes follow a
        fixed convention, so the same geometry gives the same modes on CPU and
        GPU and across LAPACK builds: within each group of equal eigenvalues
        (relative spread <= 1e-8) they are the in-order orthonormalization of
        the projections of fixed pseudo-random functions of the actuator
        positions, each with a positive projection on its function.

        Args:
            n_modes: Number of modes, at most the number of actuators minus
                the number of removed modes.
            ignore_piston: Diagonalize the piston-removed covariance
                ``P C P`` (``P = I - 11^T/N``), so every mode has exactly zero
                mean. Piston is not generally an exact KL mode, so this is not
                the same as dropping the first mode.
            orthonormalize: Accepted for a uniform API; KL modes are already
                orthonormal.
            remove: Further modes to keep out of the basis, e.g. ``"tiptilt"``
                (see :func:`aobasis.removal_basis`). As for piston, the
                covariance is diagonalized with ``P = I - U U^T``, ``U`` an
                orthonormal basis of the removed modes, so the KL modes are
                those of the turbulence left after removing them.
            normalize: Scale each mode to unit ``"rms"``, ``"l2"``,
                ``"peak"`` or ``"pv"`` after everything else (see
                :func:`aobasis.normalize_modes`). ``None`` keeps the
                generator's own scale (unit L2 norm). ``eigenvalues`` are
                rescaled to stay the variance of each returned mode's
                coefficient.
        """
        _check_normalize(normalize)
        removed = self._removed_subspace(remove, ignore_piston)
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators - removed.shape[1])
        if self.kolmogorov:
            piston = np.ones(self.n_actuators)
            if np.linalg.norm(piston - removed @ (removed.T @ piston)) > 1e-8 * np.sqrt(self.n_actuators):
                raise ValueError(
                    "Kolmogorov turbulence (outer_scale=inf) has infinite piston variance; "
                    "pass ignore_piston=True."
                )

        if n_modes == 0:
            self.eigenvalues = np.array([], dtype=float)
            self.modes = np.zeros((self.n_actuators, 0), dtype=float)
            return self.modes

        cov = self._von_karman_covariance()
        cp = _load_cupy()[0] if self.use_gpu else None
        xp = cp if self.use_gpu else np
        if removed.shape[1]:
            # P C P with P = I - U U^T, without forming P.
            u = xp.asarray(removed)
            cu = cov @ u
            cov = cov - u @ cu.T - cu @ u.T + u @ ((u.T @ cu) @ u.T)
            cov = 0.5 * (cov + cov.T)

        # Sorted by decreasing variance. The removed modes are eigenvectors of
        # P C P with eigenvalue ~0, so they sort last and are never chosen.
        if self.use_gpu:
            eigenvalues, eigenvectors = cp.linalg.eigh(cov)
            sorter = cp.argsort(eigenvalues)[::-1]
            eigenvalues = cp.asnumpy(eigenvalues[sorter])
            # Keep the whole degenerate cluster that mode n_modes - 1 belongs to.
            n_vectors = _cluster_end(eigenvalues, n_modes - 1)
            eigenvectors = cp.asnumpy(eigenvectors[:, sorter[:n_vectors]])
        else:
            eigenvalues, eigenvectors = _top_eigenpairs(cov, n_modes)
        modes = _canonical_eigenvectors(eigenvalues, eigenvectors, self.positions, n_modes)
        eigenvalues = eigenvalues[:n_modes] * (self.r0_wavelength / self.wavelength) ** 2
        if normalize is not None:
            # phase = a m = (a s)(m / s): dividing a mode by s scales its
            # coefficient variance by s^2.
            scale = mode_scales(modes, normalize)
            scale = np.where(scale > 0, scale, 1.0)
            modes = modes / scale
            eigenvalues = eigenvalues * scale**2
        self.eigenvalues = eigenvalues
        self.modes = modes
        return self.modes
