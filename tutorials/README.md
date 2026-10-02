# aobasis tutorials

Jupyter walkthroughs, from a five-minute start to calibrating an AO system. Each notebook stands on its own; the "start here" column says which ones you need first.

```bash
pip install "aobasis[tutorials]"   # aobasis + matplotlib + astropy + Jupyter
jupyter notebook tutorials/
```

## Getting started: "I just need a basis"

| Notebook | What you'll learn | Start here |
|---|---|---|
| [01 · Quickstart](01_quickstart.ipynb) | actuator positions → M2C → plot → save, in five minutes | — |
| [02 · A tour of the bases](02_tour_of_bases.ipynb) | what KL, Zernike, Fourier, zonal, zonal-fast and Hadamard modes are for, and how to choose | 01 |

## Everyday use: "I need it to fit my system"

| Notebook | What you'll learn | Start here |
|---|---|---|
| [03 · Shaping a basis](03_shaping_a_basis.ipynb) | piston, tip/tilt and waffle removal; orthonormalization; normalization; Zernike orderings; what the warnings mean | 01 |
| [04 · Actuator geometry](04_actuator_geometry.ipynb) | square and hexagonal grids, pitch conventions, obstructions and spiders, your own DM layout, annular Zernikes | 01 |

## In depth: "I build AO systems with it"

| Notebook | What you'll learn | Start here |
|---|---|---|
| [05 · KL modes and turbulence](05_kl_and_turbulence.ipynb) | the covariance behind KL, eigenvalues as priors, r0/L0/wavelength, Kolmogorov, phase screens, why KL beats Zernike, GPU | 02 |
| [06 · Influence functions and DM KL](06_influence_functions.ipynb) | fitting modes onto DM influence functions, coverage and regularization, orthonormal surfaces, `DMKLBasisGenerator`, measured IFs | 03, 05 |
| [07 · Calibration patterns](07_calibration.ipynb) | measuring interaction matrices with zonal, Hadamard and zonal-fast patterns; noise, frame budgets and modal reconstructors | 02, 06 |
| [08 · Saving and sharing](08_saving_and_sharing.ipynb) | `.npz` and FITS files, metadata, rebuilding a basis, handing an M2C to a controller, C2M | 01 |

Ready-to-run scripts for common jobs are in [`examples/`](../examples).

Every notebook is executed in CI, so the code and outputs match the current release.
