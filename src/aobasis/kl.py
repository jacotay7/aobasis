import numpy as np
from scipy.special import kv, gamma
from scipy.linalg import eigh
from .base import BasisGenerator

class KLBasisGenerator(BasisGenerator):
    """
    Generates Karhunen-Loève modes based on Von Kármán statistics.
    """
    
    def __init__(self, positions: np.ndarray, fried_parameter: float = 0.16, outer_scale: float = 30.0):
        super().__init__(positions)
        self.fried_parameter = fried_parameter
        self.outer_scale = outer_scale
        self.eigenvalues = None

    def _von_karman_covariance(self) -> np.ndarray:
        """Compute the Von Karman phase covariance matrix."""
        diffs = self.positions[:, None, :] - self.positions[None, :, :]
        r = np.linalg.norm(diffs, axis=-1)
        
        L0 = self.outer_scale
        r0 = self.fried_parameter
        
        # Variance sigma^2 calculation to match structure function limit
        # D(r) = 6.88 * (r/r0)^(5/3) for small r
        # sigma^2 = A * (L0/r0)^(5/3)
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

    def generate(self, n_modes: int, ignore_piston: bool = False, **kwargs) -> np.ndarray:
        cov = self._von_karman_covariance()
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
