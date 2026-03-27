from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from .utils import plot_basis_modes


def _validate_positions_array(positions: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be a finite numeric array with shape (n_actuators, 2).") from exc

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("positions must have shape (n_actuators, 2).")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite values.")

    return array

class BasisGenerator(ABC):
    """
    Abstract base class for AO basis generators.
    """
    
    def __init__(self, positions: np.ndarray):
        """
        Args:
            positions: (N, 2) array of actuator coordinates (x, y) in meters.
        """
        self.positions = _validate_positions_array(positions)
        self.n_actuators = self.positions.shape[0]
        self.modes: Optional[np.ndarray] = None

    def _validate_n_modes(self, n_modes: int, max_modes: Optional[int] = None) -> int:
        if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)):
            raise ValueError("n_modes must be an integer.")

        n_modes = int(n_modes)
        if n_modes < 0:
            raise ValueError("n_modes must be non-negative.")
        if max_modes is not None and n_modes > max_modes:
            raise ValueError(f"Cannot generate {n_modes} modes; maximum available is {max_modes}.")

        return n_modes
        
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
    
    def save(self, filepath: Union[str, Path]) -> None:
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
    def load(cls, filepath: Union[str, Path]) -> 'BasisGenerator':
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

    def plot(self, count: int = 6, outfile: Optional[Union[str, Path]] = None, **kwargs):
        """Plot the generated modes."""
        if self.modes is None:
            raise ValueError("No modes to plot.")
        plot_basis_modes(self.modes, self.positions, count=count, outfile=outfile, **kwargs)

class ConcreteBasis(BasisGenerator):
    """Helper class for loading existing bases."""
    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        if self.modes is None:
            raise NotImplementedError("This is a loaded basis container.")
        n_modes = self._validate_n_modes(n_modes, max_modes=self.modes.shape[1])
        return self.modes[:, :n_modes]
