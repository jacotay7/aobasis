from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .utils import plot_basis_modes

class BasisGenerator(ABC):
    """
    Abstract base class for AO basis generators.
    """
    
    def __init__(self, positions: np.ndarray):
        """
        Args:
            positions: (N, 2) array of actuator coordinates (x, y) in meters.
        """
        self.positions = np.array(positions)
        self.n_actuators = self.positions.shape[0]
        self.modes: Optional[np.ndarray] = None
        
    @abstractmethod
    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        """
        Generate the basis modes.
        
        Args:
            n_modes: Number of modes to generate.
            
        Returns:
            modes: (n_actuators, n_modes) matrix.
        """
        pass
    
    def save(self, filepath: str | Path) -> None:
        """
        Save the generated basis and actuator positions to a .npz file.
        """
        if self.modes is None:
            raise ValueError("No modes generated yet. Call generate() first.")
            
        np.savez(
            filepath,
            modes=self.modes,
            positions=self.positions,
            basis_type=self.__class__.__name__
        )
        
    @classmethod
    def load(cls, filepath: str | Path) -> 'BasisGenerator':
        """
        Load a basis from a .npz file. 
        Note: This returns a generic container or re-instantiates the specific class if possible.
        For simplicity here, we might just return the data or a generic wrapper.
        """
        data = np.load(filepath)
        positions = data['positions']
        modes = data['modes']
        
        # Create a generic instance to hold the data
        # In a more complex system, we might factory this based on basis_type
        instance = ConcreteBasis(positions)
        instance.modes = modes
        return instance

    def plot(self, count: int = 6, outfile: Optional[str | Path] = None, **kwargs):
        """Plot the generated modes."""
        if self.modes is None:
            raise ValueError("No modes to plot.")
        plot_basis_modes(self.modes, self.positions, count=count, outfile=outfile, **kwargs)

class ConcreteBasis(BasisGenerator):
    """Helper class for loading existing bases."""
    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        if self.modes is None:
            raise NotImplementedError("This is a loaded basis container.")
        return self.modes[:, :n_modes]
