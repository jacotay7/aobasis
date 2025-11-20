import numpy as np
from scipy.linalg import hadamard
from .base import BasisGenerator

class HadamardBasisGenerator(BasisGenerator):
    """
    Generates Hadamard modes.
    Useful for interaction matrix calibration (multiplexing).
    """
    
    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        """
        Generate Hadamard modes.
        
        Since Hadamard matrices exist only for sizes 2^k (or multiples of 4),
        we find the next power of 2 >= n_actuators, generate the Hadamard matrix,
        and truncate it to the number of actuators (rows) and requested modes (columns).
        """
        # Find next power of 2 covering the number of actuators
        # We need at least n_actuators rows to define the pattern on the grid
        # And we need enough columns for n_modes
        
        # Usually for calibration, we want a square matrix that covers all actuators.
        # So size >= n_actuators.
        
        size = 1
        while size < self.n_actuators:
            size *= 2
            
        # If n_modes is larger than this size, we might need a larger matrix?
        # But usually we can't have more orthogonal modes than actuators (degrees of freedom).
        # However, Hadamard patterns are defined by the full matrix.
        
        if n_modes > size:
             # If user asks for more modes than the natural Hadamard block size covering actuators,
             # we might need to go bigger.
             while size < n_modes:
                 size *= 2
        
        H = hadamard(size)
        
        # Truncate to actuators (rows) and modes (cols)
        # Note: Truncated Hadamard is not necessarily orthogonal!
        # But it is the standard way to project Hadamard patterns onto a smaller aperture.
        
        self.modes = H[:self.n_actuators, :n_modes]
        
        # Optional: Normalize? Hadamard entries are 1 and -1.
        # Keeping them as is is standard.
        
        return self.modes
