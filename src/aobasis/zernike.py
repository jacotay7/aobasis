import numpy as np
import math
from typing import Tuple
from .base import BasisGenerator

class ZernikeBasisGenerator(BasisGenerator):
    """
    Generates Zernike polynomials on the actuator grid.
    """
    
    def __init__(self, positions: np.ndarray, pupil_radius: float):
        super().__init__(positions)
        self.pupil_radius = pupil_radius

    def _zernike_radial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Compute radial Zernike polynomial R_n^m(rho)."""
        R = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            c = ((-1)**k * math.factorial(n - k)) / (
                math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
            )
            R += c * rho**(n - 2 * k)
        return R

    def _zernike(self, n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute Zernike polynomial Z_n^m(rho, theta)."""
        R = self._zernike_radial(n, abs(m), rho)
        if m >= 0:
            return R * np.cos(m * theta)
        else:
            return R * np.sin(abs(m) * theta)

    def generate(self, n_modes: int, ignore_piston: bool = False, **kwargs) -> np.ndarray:
        """
        Generate Zernike modes using Noll indexing.
        j=1: Piston
        j=2: Tip (X-tilt)
        j=3: Tilt (Y-tilt)
        ...
        """
        # Normalize coordinates
        x = self.positions[:, 0] / self.pupil_radius
        y = self.positions[:, 1] / self.pupil_radius
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Clip rho to 1.0 to avoid instability outside unit circle
        rho = np.clip(rho, 0, 1.0)
        
        modes_list = []
        
        current_j = 1
        while len(modes_list) < n_modes:
            if ignore_piston and current_j == 1:
                current_j += 1
                continue
                
            n, m = self._noll_to_nm(current_j)
            mode = self._zernike(n, m, rho, theta)
            modes_list.append(mode)
            current_j += 1
            
        self.modes = np.column_stack(modes_list)
        return self.modes

    def _noll_to_nm(self, j: int) -> Tuple[int, int]:
        """
        Convert Noll index j to radial order n and azimuthal frequency m.
        Based on Noll, J. Opt. Soc. Am. 66, 207 (1976).
        """
        if j < 1:
            raise ValueError("Noll index must be >= 1")
            
        # 1. Find n
        # n is the smallest integer such that j <= (n+1)(n+2)/2
        n = 0
        while True:
            if j <= (n + 1) * (n + 2) // 2:
                break
            n += 1
            
        # 2. Find m
        # j_n_start is the first index for this radial order n
        # The number of modes up to order n-1 is n(n+1)/2
        j_start = n * (n + 1) // 2 + 1
        
        # The sequence of m values for a given n in Noll ordering depends on n mod 4
        # But simpler logic:
        # m values go n, n-2, ..., -(n-2), -n? No, Noll is specific.
        # Noll sorts by m magnitude, then sign.
        
        # Let's implement the standard logic derived from the paper or standard libs
        
        # Order within the block of constant n
        k = j - j_start # 0-based index within the block
        
        # m values are n, n-2, ..., 1 or 0
        # For a given n, there are n+1 modes.
        # If n is even, m \in {0, 2, -2, 4, -4, ... n, -n} ??
        # Actually Noll is:
        # j=1 (n=0): m=0
        # j=2 (n=1): m=1 (odd j -> cos -> m>0? No, Noll j=2 is m=1, j=3 is m=-1)
        # j=3 (n=1): m=-1
        # j=4 (n=2): m=0
        # j=5 (n=2): m=-2 (Wait, Noll j=5 is m=-2? Or m=2?)
        # Let's use a robust algorithm.
        
        # Algorithm from "Noll Indices" standard implementation
        n = int(np.ceil((-3 + np.sqrt(9 + 8*(j-1))) / 2))
        sub_j = j - n*(n+1)//2
        
        if n % 2 == 0:
            # Even n
            # m = 0, 2, -2, 4, -4 ...
            if sub_j == 1:
                m = 0
            elif sub_j % 2 == 0:
                m = 2 * (sub_j // 2)
            else:
                m = -2 * (sub_j // 2)
        else:
            # Odd n
            # m = 1, -1, 3, -3 ...
            if sub_j % 2 == 0:
                m = - (2 * (sub_j // 2) - 1)
            else:
                m = (2 * (sub_j // 2) + 1)
                
        return n, m
