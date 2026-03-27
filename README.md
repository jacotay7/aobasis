# AO Basis (aobasis)

A Python package for generating various modal basis sets for Adaptive Optics (AO) systems. This tool allows you to easily create, visualize, and save basis sets for any deformable mirror geometry.

## Features

- **Karhunen-Loève (KL) Modes**: Optimized for atmospheric turbulence (Von Kármán spectrum).
  - Optional GPU acceleration available for large systems (requires CuPy).
- **Zernike Polynomials**: Standard optical aberration modes (Noll indexing).
- **Fourier Modes**: Sinusoidal basis sets.
- **Zonal Basis**: Single actuator pokes (Identity).
- **Zonal Fast Basis**: Distance-constrained grouped actuator pokes for faster calibration sweeps.
- **Hadamard Basis**: Orthogonal binary patterns for calibration.
- **Flexible Geometry**: Works with arbitrary actuator positions (defaulting to circular grids).
- **Piston Removal**: Option to exclude piston/DC modes from generation.
- **Visualization**: Built-in plotting tools for quick inspection.
- **Serialization**: Save and load basis sets to/from `.npz` files.

## Installation

### Prerequisites
- Python 3.8 or higher
- (Optional) For GPU-accelerated KL generation: CUDA-compatible GPU and CuPy

### Install from Source
Clone the repository and install using pip:

```bash
git clone https://github.com/jacotay7/aobasis.git
cd aobasis
pip install .
```

For development (editable install with test dependencies):
```bash
pip install -e ".[dev]"
```

### GPU Acceleration (Optional)
To enable GPU acceleration for KL basis generation, you need to install CuPy and ensure you have a CUDA-compatible GPU.

#### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.x or 12.x)

#### Installation via Conda (Recommended)
This method automatically handles CUDA dependencies:

```bash
# Create a new conda environment (optional but recommended)
conda create -n aobasis python=3.12
conda activate aobasis

# Install CuPy from conda-forge (auto-detects CUDA version)
conda install -c conda-forge cupy

# Install CUDA toolkit if not already present
conda install -c nvidia cuda-toolkit
```

#### Installation via Pip
If you prefer pip and already have CUDA installed on your system:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

#### Verify Installation
Test that CuPy is working correctly:

```python
import cupy as cp
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")

# Simple test
a = cp.array([1, 2, 3])
b = cp.array([4, 5, 6])
print(f"Sum: {cp.asnumpy(a + b)}")  # Should print [5, 7, 9]
```

If you encounter any issues, consult the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Quick Start

Here is a simple example of generating and plotting KL modes for a 10-meter telescope:

```python
from aobasis import KLBasisGenerator, make_circular_actuator_grid

# 1. Define the actuator geometry
positions = make_circular_actuator_grid(telescope_diameter=10.0, grid_size=20)

# 2. Initialize the generator (use_gpu=True for GPU acceleration if available)
kl_gen = KLBasisGenerator(positions, fried_parameter=0.16, outer_scale=30.0, use_gpu=False)

# 3. Generate modes (excluding piston)
modes = kl_gen.generate(n_modes=50, ignore_piston=True)

# 4. Plot the first 6 modes
kl_gen.plot(count=6, title_prefix="KL Mode")

# 5. Save to disk
kl_gen.save("my_kl_basis.npz")
```

## Zonal Fast Basis

`ZonalFastBasisGenerator` groups actuators into binary poke patterns such that no two actuators in the same mode are closer than a user-defined distance `D`. This is useful when you want a compact calibration basis that reduces the number of measurements compared with pure zonal pokes. For square-grid actuator layouts it uses a modulo lattice grouping directly, and for exotic layouts it falls back to a greedy graph-coloring approach.

```python
import numpy as np

from aobasis import ZonalFastBasisGenerator, make_circular_actuator_grid, make_concentric_actuator_grid

# Example 1: grid-like actuator positions clipped by a circular pupil.
positions = make_circular_actuator_grid(telescope_diameter=10.0, grid_size=20)
grid_gen = ZonalFastBasisGenerator(positions, min_distance=0.8)
grid_modes = grid_gen.generate()
print("Grid layout:", grid_modes.shape)
grid_gen.plot(count=min(12, grid_modes.shape[1]), title_prefix="Zonal Fast Grid")

# Example 2: non-grid actuator positions.
exotic_positions = make_concentric_actuator_grid(telescope_diameter=10.0, n_rings=5)
exotic_positions = exotic_positions + 0.03 * np.sin(exotic_positions)
exotic_gen = ZonalFastBasisGenerator(exotic_positions, min_distance=1.0)
exotic_modes = exotic_gen.generate()
print("Exotic layout:", exotic_modes.shape)
exotic_gen.plot(count=min(12, exotic_modes.shape[1]), title_prefix="Zonal Fast Exotic")
```

The returned matrix still has the standard `(n_actuators, n_modes)` layout, but each column is now a sparse binary pattern rather than a single-actuator poke. Every actuator appears in exactly one column of the full basis.

## Performance

Generation times for 100 modes benchmarked on the following system:
- **CPU**: AMD Ryzen 9 9950X3D (16-core, 32-thread)
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB)
- **OS**: Linux (Ubuntu)

| Basis | 16x16 Grid (~170 acts) | 32x32 Grid (~740 acts) | 64x64 Grid (~3100 acts) |
|-------|------------------------|------------------------|-------------------------|
| **KL (CPU)** | 0.010s | 0.170s | 3.008s |
| **KL (GPU)** | 0.005s | 0.019s | 0.202s |
| **Zernike** | 0.001s | 0.002s | 0.005s |
| **Fourier** | <0.001s | 0.001s | 0.003s |
| **Zonal** | <0.001s | <0.001s | 0.003s |
| **Zonal Fast** | depends on spacing threshold | depends on spacing threshold | depends on spacing threshold |
| **Hadamard** | <0.001s | 0.001s | 0.031s |

*Note: KL basis generation is computationally intensive ($O(N^3)$) due to the dense covariance matrix diagonalization. GPU acceleration provides significant speedup (8-15x) for larger grids.*

## Tutorials

We provide Jupyter notebooks to help you get started.

1.  **Getting Started**: `tutorials/getting_started.ipynb` covers all supported basis types, including zonal fast grouped pokes.

To run the tutorials:
```bash
# Install Jupyter if you haven't already
pip install jupyter

# Launch the notebook server
jupyter notebook tutorials/getting_started.ipynb
```

## Development & Testing

This project uses `pytest` for testing. To run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## Issues

If you encounter any bugs or have feature requests, please file an issue on the [GitHub Issues](https://github.com/jacotay7/aobasis/issues) page.

## Contact

For questions or support, please contact:

**User Name**  
Email: jacobataylor7@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
