import warnings

import numpy as np
from scipy.special import kv, gamma
from scipy.linalg import eigh
from .base import BasisGenerator, RemoveSpec

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


class KLBasisGenerator(BasisGenerator):
    """
    Generates Karhunen-Loève modes based on Von Kármán statistics.

    The modes are the eigenvectors of the Von Kármán phase covariance between
    actuators, sorted by decreasing variance (``eigenvalues``). ``outer_scale``
    changes the modes; ``fried_parameter`` only scales the covariance, so it
    changes ``eigenvalues`` but not the modes.
    """
    
    def __init__(self, positions: np.ndarray, fried_parameter: float = 0.16, outer_scale: float = 30.0, use_gpu: bool = False):
        super().__init__(positions)
        if not np.isscalar(fried_parameter) or not np.isfinite(fried_parameter) or fried_parameter <= 0:
            raise ValueError("fried_parameter must be a positive finite scalar.")
        if not np.isscalar(outer_scale) or not np.isfinite(outer_scale) or outer_scale <= 0:
            raise ValueError("outer_scale must be a positive finite scalar.")
        self.fried_parameter = fried_parameter
        self.outer_scale = outer_scale
        self.eigenvalues = None
        self.use_gpu = use_gpu
        
        if self.use_gpu and _load_cupy() is None:
            warnings.warn("CuPy not found; KLBasisGenerator falls back to the CPU.", RuntimeWarning, stacklevel=2)
            self.use_gpu = False

    def _von_karman_covariance(self) -> np.ndarray:
        """Compute the Von Karman phase covariance matrix."""
        if self.use_gpu:
            return self._von_karman_covariance_gpu()
        else:
            return self._von_karman_covariance_cpu()
    
    def _von_karman_covariance_cpu(self) -> np.ndarray:
        """Compute the Von Karman phase covariance matrix on CPU."""
        diffs = self.positions[:, None, :] - self.positions[None, :, :]
        r = np.linalg.norm(diffs, axis=-1)
        
        L0 = self.outer_scale
        r0 = self.fried_parameter
        
        # Variance sigma^2 calculation to match structure function limit
        A = (5.0/6.0) * (6.88/2.0) * gamma(5.0/6.0) / (gamma(1.0/6.0) * np.pi**(5.0/3.0))
        sigma2 = A * (L0 / r0)**(5.0/3.0)
        
        cov = np.zeros_like(r, dtype=float)
        
        # Avoid division by zero
        mask = r > 1e-9
        if np.any(mask):
            u = 2 * np.pi * r[mask] / L0
            nu = 5.0/6.0
            norm_factor = 2**(1 - nu) / gamma(nu)
            cov[mask] = sigma2 * norm_factor * (u**nu) * kv(nu, u)
            
        cov[~mask] = sigma2
        return cov
    
    def _von_karman_covariance_gpu(self):
        """Compute the Von Karman phase covariance matrix on GPU."""
        cp, kv56 = _load_cupy()
        # Transfer positions to GPU
        positions_gpu = cp.asarray(self.positions, dtype=cp.float64)
        
        # Compute pairwise distances on GPU
        diffs = positions_gpu[:, None, :] - positions_gpu[None, :, :]
        r = cp.linalg.norm(diffs, axis=-1)
        
        L0 = self.outer_scale
        r0 = self.fried_parameter
        
        # Compute sigma^2 using GPU operations
        nu = 5.0/6.0
        gamma_5_6 = float(gamma(5.0/6.0))
        gamma_1_6 = float(gamma(1.0/6.0))
        A = (5.0/6.0) * (6.88/2.0) * gamma_5_6 / (gamma_1_6 * cp.pi**(5.0/3.0))
        sigma2 = A * (L0 / r0)**(5.0/3.0)
        
        # Compute covariance for all distances
        u = 2 * cp.pi * r / L0
        norm_factor = 2**(1 - nu) / gamma(nu)
        
        # Use custom GPU kernel for Bessel function K_{5/6}
        kv_values = cp.zeros_like(u, dtype=cp.float64)
        kv56(u, kv_values)
        
        # Compute covariance matrix
        cov = sigma2 * norm_factor * (u**nu) * kv_values
        
        # Handle zero/very small distances (diagonal or very close points)
        mask = r <= 1e-9
        cov = cp.where(mask, sigma2, cov)
        
        return cov

    def generate(
        self,
        n_modes: int,
        ignore_piston: bool = False,
        orthonormalize: bool = False,
        remove: RemoveSpec = None,
    ) -> np.ndarray:
        """
        Generate KL modes (orthonormal columns, decreasing variance).

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
        """
        removed = self._removed_subspace(remove, ignore_piston)
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators - removed.shape[1])

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

        eigenvalues, eigenvectors = cp.linalg.eigh(cov) if self.use_gpu else eigh(cov)
        # Sort by decreasing variance. The removed modes are eigenvectors of
        # P C P with eigenvalue ~0, so they sort last and are never chosen.
        sorter = xp.argsort(eigenvalues)[::-1][:n_modes]
        eigenvalues = eigenvalues[sorter]
        eigenvectors = eigenvectors[:, sorter]
        if self.use_gpu:
            eigenvalues, eigenvectors = cp.asnumpy(eigenvalues), cp.asnumpy(eigenvectors)
        self.eigenvalues = eigenvalues
        self.modes = np.asarray(eigenvectors, dtype=float)
        return self.modes
