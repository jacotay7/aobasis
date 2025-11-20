import numpy as np
from .base import BasisGenerator

class FourierBasisGenerator(BasisGenerator):
    """
    Generates Fourier modes (sine/cosine) on the actuator grid.
    """
    
    def __init__(self, positions: np.ndarray, pupil_diameter: float):
        super().__init__(positions)
        self.pupil_diameter = pupil_diameter

    def generate(self, n_modes: int, ignore_piston: bool = False, **kwargs) -> np.ndarray:
        """
        Generate Fourier modes.
        We generate pairs of sin/cos for increasing spatial frequencies.
        """
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        
        modes_list = []
        
        # Piston
        if not ignore_piston:
            modes_list.append(np.ones_like(x))
        
        # Loop through spatial frequencies
        # kx, ky integers
        # We'll spiral out or just loop kx, ky
        # Simple ordering: by magnitude of k vector
        
        k_pairs = []
        # Generate a pool of k-vectors
        k_max = int(np.sqrt(n_modes)) + 2
        for kx in range(-k_max, k_max + 1):
            for ky in range(-k_max, k_max + 1):
                if kx == 0 and ky == 0:
                    continue
                # We only need half the plane for real Fourier basis (sin/cos)
                # But simpler to just generate sin/cos pairs for positive k's?
                # Let's stick to standard real Fourier series expansion logic
                # cos(2pi(ux + vy)), sin(2pi(ux + vy))
                k_pairs.append((kx, ky))
                
        # Sort by spatial frequency magnitude
        k_pairs.sort(key=lambda k: k[0]**2 + k[1]**2)
        
        # Filter duplicates/redundancies for real basis
        # We want unique spatial frequencies. 
        # For each (kx, ky), we can have cos and sin.
        # But (kx, ky) and (-kx, -ky) are redundant.
        # So we keep only "positive" half plane.
        
        unique_ks = []
        seen = set()
        for kx, ky in k_pairs:
            if (kx, ky) in seen or (-kx, -ky) in seen:
                continue
            seen.add((kx, ky))
            unique_ks.append((kx, ky))
            
        # Generate modes
        # Fundamental frequency base: 1 cycle per pupil diameter
        f0 = 1.0 / self.pupil_diameter
        
        for kx, ky in unique_ks:
            if len(modes_list) >= n_modes:
                break
                
            arg = 2 * np.pi * f0 * (kx * x + ky * y)
            
            # Cosine component
            modes_list.append(np.cos(arg))
            
            if len(modes_list) >= n_modes:
                break
                
            # Sine component
            modes_list.append(np.sin(arg))
            
        self.modes = np.column_stack(modes_list)
        return self.modes
