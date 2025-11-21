import numpy as np
from scipy.special import kv, gamma
from scipy.linalg import eigh
from .base import BasisGenerator

try:
    import cupy as cp
    from cupy.linalg import eigh as cp_eigh
    HAS_CUPY = True
    
    # Pre-computed gamma values for Bessel function
    GAMMA_1_6 = 5.56631600178
    GAMMA_11_6 = 0.94065585824
    
    # Custom GPU kernel for K_{5/6} Bessel function (optimized for float64)
    _kv56_kernel_float64 = cp.ElementwiseKernel(
        'float64 z',
        'float64 K',
        '''
        double v = 5.0 / 6.0;
        double z_abs = fabs(z);
        if (z_abs < 2.0) {
            // Series approximation for small z
            if (z_abs < 1e-12) {
                K = 1.89718990814 * pow(z_abs, -5.0/6.0);
                return;
            }
            
            double half_z = 0.5 * z;
            double half_z_sq = half_z * half_z;
            double z_pow_v = pow(half_z, v);
            double z_pow_neg_v = pow(half_z, -v);
            
            double sum_a = z_pow_v / gamma_11_6;
            double sum_b = z_pow_neg_v / gamma_1_6;
            double term_a = sum_a;
            double term_b = sum_b;
            
            double prev_sum_a = 0.0;
            double prev_sum_b = 0.0;
            int k = 1;
            double tol = 1e-15;
            
            for (int i = 0; i < 100; ++i) {
                double k_plus_v = k + v;
                double k_minus_v = k - v;
                
                double factor_a = half_z_sq / (k * k_plus_v);
                double factor_b = half_z_sq / (k * k_minus_v);
                
                term_a *= factor_a;
                term_b *= factor_b;
                sum_a += term_a;
                sum_b += term_b;
                
                if ((i & 1) == 1) {
                    double rel_change_a = fabs(sum_a - prev_sum_a) / fabs(sum_a);
                    double rel_change_b = fabs(sum_b - prev_sum_b) / fabs(sum_b);
                    
                    if (rel_change_a < tol && rel_change_b < tol) {
                        break;
                    }
                    prev_sum_a = sum_a;
                    prev_sum_b = sum_b;
                }
                k += 1;
            }
            K = M_PI * (sum_b - sum_a);
        } else {
            // Asymptotic approximation for larger z
            double z_inv = 1.0 / z;
            
            double sum_terms = 1.0 + z_inv * (2.0/9.0 + z_inv * (
                        -7.0/81.0 + z_inv * (175.0/2187.0 + z_inv * (
                            -2275.0/19683.0 + z_inv * 5005.0/177147.0
                        )))); 
            
            double sqrt_term = sqrt(M_PI / (2.0 * z));
            double exp_term = exp(-z);
            K = sqrt_term * exp_term * sum_terms;
        }
        ''',
        name='kv56_kernel_float64',
        preamble=f'''
        const double gamma_1_6 = {GAMMA_1_6};
        const double gamma_11_6 = {GAMMA_11_6};
        '''
    )
    
except ImportError:
    cp = None
    cp_eigh = None
    HAS_CUPY = False

class KLBasisGenerator(BasisGenerator):
    """
    Generates Karhunen-Loève modes based on Von Kármán statistics.
    """
    
    def __init__(self, positions: np.ndarray, fried_parameter: float = 0.16, outer_scale: float = 30.0, use_gpu: bool = False):
        super().__init__(positions)
        self.fried_parameter = fried_parameter
        self.outer_scale = outer_scale
        self.eigenvalues = None
        self.use_gpu = use_gpu
        
        if self.use_gpu and not HAS_CUPY:
            print("Warning: CuPy not found. Falling back to CPU.")
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
        _kv56_kernel_float64(u, kv_values)
        
        # Compute covariance matrix
        cov = sigma2 * norm_factor * (u**nu) * kv_values
        
        # Handle zero/very small distances (diagonal or very close points)
        mask = r <= 1e-9
        cov = cp.where(mask, sigma2, cov)
        
        return cov

    def generate(self, n_modes: int, ignore_piston: bool = False, **kwargs) -> np.ndarray:
        cov = self._von_karman_covariance()
        
        if self.use_gpu:
            # Covariance is already on GPU, compute eigendecomposition
            eigenvalues, eigenvectors = cp_eigh(cov)
            
            # Sort descending on GPU
            sorter = cp.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorter]
            sorted_eigenvectors = eigenvectors[:, sorter]
            
            start_idx = 1 if ignore_piston else 0
            end_idx = start_idx + n_modes
            
            # Extract modes and eigenvalues, then transfer to CPU
            self.eigenvalues = cp.asnumpy(sorted_eigenvalues[start_idx:end_idx])
            self.modes = cp.asnumpy(sorted_eigenvectors[:, start_idx:end_idx])
        else:
            # CPU path - cov is already numpy array
            eigenvalues, eigenvectors = eigh(cov)
            
            # Sort descending
            sorter = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorter]
            sorted_eigenvectors = eigenvectors[:, sorter]
            
            start_idx = 1 if ignore_piston else 0
            end_idx = start_idx + n_modes
            
            self.eigenvalues = sorted_eigenvalues[start_idx:end_idx]
            self.modes = sorted_eigenvectors[:, start_idx:end_idx]
        
        return self.modes
