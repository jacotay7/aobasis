import numpy as np
from .base import BasisGenerator

class ZonalBasisGenerator(BasisGenerator):
    """
    Generates a Zonal basis (Identity matrix).
    Each mode corresponds to poking a single actuator.
    """
    
    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        """
        Generate Zonal modes.
        
        Args:
            n_modes: Number of modes to generate. 
                     If n_modes < n_actuators, returns the first n_modes actuators.
                     If n_modes > n_actuators, raises ValueError (or we could pad with zeros, but that's weird).
        """
        if n_modes > self.n_actuators:
            raise ValueError(f"Cannot generate {n_modes} zonal modes for {self.n_actuators} actuators.")
            
        # Identity matrix
        full_basis = np.eye(self.n_actuators)
        
        self.modes = full_basis[:, :n_modes]
        return self.modes
