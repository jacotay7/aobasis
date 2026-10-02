# AO Basis (aobasis)

A Python package for generating various modal basis sets for Adaptive Optics (AO) systems. This tool allows you to easily create, visualize, and save basis sets for any deformable mirror geometry.

## Features

- **Karhunen-Loève (KL) Modes**: Optimized for atmospheric turbulence (Von Kármán spectrum).
  - Optional GPU acceleration available for large systems (requires CuPy).
  - `DMKLBasisGenerator`: KL modes of the DM itself, from its influence functions (double diagonalization): orthonormal surfaces over the pupil, statistically independent coefficients.
- **Zernike Polynomials**: Standard optical aberration modes with Noll normalization, in Noll, OSA/ANSI or Fringe order (`ordering=`), and annular Zernikes for a central obstruction (`obscuration=`).
- **Fourier Modes**: Sinusoidal basis sets; aliased frequencies are skipped, so each mode is independent of the ones before it. Near full size the raw matrix is ill-conditioned; `orthonormalize=True` gives an accurate orthonormal basis.
- **Zonal Basis**: Single actuator pokes (Identity).
- **Zonal Fast Basis**: Distance-constrained grouped actuator pokes for faster calibration sweeps.
- **Hadamard Basis**: +/-1 patterns for calibration: a truncated Sylvester matrix, or (`construction="smallest"`) the smallest Sylvester-doubled Paley matrix, which needs no truncation on many grids; `selection="balanced"` picks the most piston-free columns.
- **Flexible Geometry**: Works with arbitrary actuator positions (defaulting to circular grids).
- **Mode Removal**: `ignore_piston=True` makes every mode zero-mean, and `remove=` keeps other modes out of the basis (`"tiptilt"`, or any `(n_actuators, k)` array). Zernike, Fourier and Hadamard modes have them projected out; KL diagonalizes the covariance with them removed. `aobasis.project_out` and `aobasis.removal_basis` do the same for any matrix.
- **Orthonormalization**: Zernike, Fourier and Hadamard modes sampled on a discrete grid are not orthogonal; `generate(..., orthonormalize=True)` Gram-Schmidts them in order (`aobasis.orthonormalize_modes` does the same for any matrix). A `RuntimeWarning` flags a rank-deficient basis.
- **Normalization**: `generate(..., normalize="rms" | "l2" | "peak" | "pv")` scales every mode to unit size after piston removal and orthonormalization (`aobasis.normalize_modes` does the same for any matrix). KL `eigenvalues` follow the normalization.
- **Visualization**: Built-in plotting tools for quick inspection (`pip install aobasis[plot]` for matplotlib).
- **Serialization**: `save`/`load` use `.npz` files holding the modes, positions, generator parameters, `generate()` options, KL eigenvalues and aobasis version (`load` returns a `ConcreteBasis` with all of them). `save_fits`/`load_fits` do the same in FITS (`pip install aobasis[fits]`).
- **Influence-function fitting**: `fit_to_influence_functions` turns modes sampled on the pupil into least-squares DM commands, with `gaussian_influence_functions` and `make_pupil_points` to build the inputs.
- **Geometry helpers**: `make_circular_actuator_grid` (by `grid_size` or `pitch`, actuators on the rim or in cell centres), `make_hexagonal_actuator_grid`, `make_concentric_actuator_grid`, all with `obscuration=` and spider arms (`n_spiders`, `spider_width`, `spider_angle`), and `positions_from_mask` for a boolean actuator map.

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

`ZonalFastBasisGenerator` groups actuators into binary poke patterns such that no two actuators in the same mode are closer than a user-defined distance `D`. This is useful when you want a compact calibration basis that reduces the number of measurements compared with pure zonal pokes. It colours the actuators' conflict graph greedily (DSATUR) and, when the actuators lie on a lattice (square, hexagonal, ...), also with the best sublattice colouring, keeping whichever needs fewer modes. `generate(signs="random")` gives each poke a random sign.

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

## Fitting Modes onto Influence Functions

Sampling a mode at the actuator positions treats the DM as a set of point values. To get commands whose DM *surface* matches a mode, evaluate the mode on pupil points (every generator accepts arbitrary points) and fit it onto the DM influence functions by least squares:

```python
from aobasis import (
    ZernikeBasisGenerator, fit_to_influence_functions, gaussian_influence_functions,
    make_circular_actuator_grid, make_pupil_points,
)

points = make_pupil_points(diameter=10.0, n_pixels=64)          # pupil pixel centres
actuators = make_circular_actuator_grid(11.0, 22)               # one ring beyond the pupil
influence = gaussian_influence_functions(actuators, points)     # or your measured IFs, (n_points, n_actuators)
pupil_modes = ZernikeBasisGenerator(points, pupil_radius=5.0).generate(50, ignore_piston=True)
commands, residual = fit_to_influence_functions(
    pupil_modes, influence, orthonormalize=True, return_residual=True
)
```

`commands` is the `(n_actuators, n_modes)` modal-to-command matrix, `residual` each mode's relative fitting error, and `orthonormalize=True` makes the DM surfaces orthonormal over the pupil. `rcond` and `regularization` control unseen or badly seen actuators.

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

CI cannot run the GPU tests: they need CuPy and a CUDA device, and skip without them. Run them locally before changing GPU code:

```bash
pip install cupy-cuda12x
pytest -k gpu -rs   # -rs shows a skip reason if no device is found
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
