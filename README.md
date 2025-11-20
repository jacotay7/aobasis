# AO Basis (aobasis)

A Python package for generating various modal basis sets for Adaptive Optics (AO) systems. This tool allows you to easily create, visualize, and save basis sets for any deformable mirror geometry.

## Features

- **Karhunen-Loève (KL) Modes**: Optimized for atmospheric turbulence (Von Kármán spectrum).
- **Zernike Polynomials**: Standard optical aberration modes (Noll indexing).
- **Fourier Modes**: Sinusoidal basis sets.
- **Zonal Basis**: Single actuator pokes (Identity).
- **Hadamard Basis**: Orthogonal binary patterns for calibration.
- **Flexible Geometry**: Works with arbitrary actuator positions (defaulting to circular grids).
- **Piston Removal**: Option to exclude piston/DC modes from generation.
- **Visualization**: Built-in plotting tools for quick inspection.
- **Serialization**: Save and load basis sets to/from `.npz` files.

## Installation

### Prerequisites
- Python 3.8 or higher

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

## Quick Start

Here is a simple example of generating and plotting KL modes for a 10-meter telescope:

```python
from aobasis import KLBasisGenerator, make_circular_actuator_grid

# 1. Define the actuator geometry
positions = make_circular_actuator_grid(telescope_diameter=10.0, grid_size=20)

# 2. Initialize the generator
kl_gen = KLBasisGenerator(positions, fried_parameter=0.16, outer_scale=30.0)

# 3. Generate modes (excluding piston)
modes = kl_gen.generate(n_modes=50, ignore_piston=True)

# 4. Plot the first 6 modes
kl_gen.plot(count=6, title_prefix="KL Mode")

# 5. Save to disk
kl_gen.save("my_kl_basis.npz")
```

## Performance

Generation times for 100 modes on a standard laptop (M1/M2 class):

| Basis | 16x16 Grid (~170 acts) | 32x32 Grid (~740 acts) | 64x64 Grid (~3100 acts) |
|-------|------------------------|------------------------|-------------------------|
| **KL** | 0.01s | 0.29s | 5.60s |
| **Zernike** | 0.001s | 0.002s | 0.02s |
| **Fourier** | 0.001s | 0.001s | 0.003s |
| **Zonal** | <0.001s | <0.001s | 0.005s |
| **Hadamard** | <0.001s | 0.004s | 0.09s |

*Note: KL basis generation is computationally intensive ($O(N^3)$) due to the dense covariance matrix diagonalization.*

## Tutorials

We provide Jupyter notebooks to help you get started.

1.  **Getting Started**: `tutorials/getting_started.ipynb` covers all supported basis types and features.

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
Email: jtaylor@keck.hawaii.edu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
