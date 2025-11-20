import time
import numpy as np
from aobasis import (
    KLBasisGenerator, 
    ZernikeBasisGenerator, 
    FourierBasisGenerator,
    ZonalBasisGenerator,
    HadamardBasisGenerator,
    make_circular_actuator_grid
)

def benchmark():
    # Grid sizes to test
    grid_sizes = [16, 32, 64]
    
    generators = [
        ("KL", KLBasisGenerator, {"fried_parameter": 0.16, "outer_scale": 30.0}),
        ("Zernike", ZernikeBasisGenerator, {"pupil_radius": 5.0}),
        ("Fourier", FourierBasisGenerator, {"pupil_diameter": 10.0}),
        ("Zonal", ZonalBasisGenerator, {}),
        ("Hadamard", HadamardBasisGenerator, {})
    ]
    
    print(f"Benchmarking Basis Generation (n_modes=n_acts)")
    print("=" * 80)
    print(f"{'Basis':<10} | {'Grid Size':<10} | {'Actuators':<10} | {'Time (s)':<10}")
    print("-" * 80)
    
    results = {}
    
    for name, GenClass, kwargs in generators:
        results[name] = []
        for size in grid_sizes:
            # Setup
            positions = make_circular_actuator_grid(10.0, size)
            n_act = positions.shape[0]
            
            # Update kwargs based on size if needed (e.g. pupil radius is constant)
            gen = GenClass(positions, **kwargs)
            
            # Timing
            start_time = time.perf_counter()
            gen.generate(n_modes=n_act)
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            print(f"{name:<10} | {size:<10} | {n_act:<10} | {duration:<10.4f}")
            results[name].append((size, n_act, duration))
        print("-" * 80)

if __name__ == "__main__":
    benchmark()
