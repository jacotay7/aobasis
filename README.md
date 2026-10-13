# aobasis

Modal basis sets for adaptive-optics deformable mirrors: KL, Zernike, Fourier, Hadamard, zonal and zonal-fast modes, for any actuator geometry.

Every generator takes your actuator positions and returns a **modal-to-command matrix** (M2C), an `(n_actuators, n_modes)` NumPy array whose column `k` is the DM command for mode `k`. It's ready for a reconstructor, a calibration sequence or a real-time controller such as [pyRTC](https://github.com/jacotay7/pyRTC).

## Installation

```bash
pip install aobasis                 # numpy + scipy only
pip install "aobasis[plot]"         # + matplotlib, for .plot()
pip install "aobasis[fits]"         # + astropy, for save_fits / load_fits
pip install "aobasis[tutorials]"    # + matplotlib, astropy and Jupyter, for the tutorials
```

aobasis is pure Python (3.8–3.14) and is tested on Linux, x86-64 and aarch64.

**GPU (optional).** `KLBasisGenerator(..., use_gpu=True)` builds and diagonalizes the covariance with [CuPy](https://docs.cupy.dev/en/stable/install.html). Install the CuPy wheel for your CUDA version, e.g. `pip install cupy-cuda12x`, or `conda install -c conda-forge cupy`. Without CuPy it falls back to the CPU with a warning, and the modes are the same either way.

## Quick start

```python
import numpy as np
import aobasis

# 1. Actuator positions: (N, 2) in metres, centred, in your DM's actuator order
positions = aobasis.make_circular_actuator_grid(telescope_diameter=10.0, grid_size=20)

# 2. A generator and its physical parameters
kl = aobasis.KLBasisGenerator(positions, fried_parameter=0.16, outer_scale=30.0)

# 3. The M2C: (276, 50), piston-free
m2c = kl.generate(n_modes=50, ignore_piston=True)

# 4. Use, inspect, keep
coefficients = np.zeros(50); coefficients[3] = 1.0  # some of mode 3
commands = m2c @ coefficients                       # put modal coefficients on the DM
c2m = aobasis.command_to_mode_matrix(m2c)           # and back
kl.plot(count=6)                                    # needs aobasis[plot]
kl.save("kl_m2c.npz")                               # modes + parameters + options + eigenvalues
```

## What's in it

**Bases**

| Generator | Modes | Typical use |
|---|---|---|
| `KLBasisGenerator` | Karhunen-Loève modes of Von Kármán (or Kolmogorov, `outer_scale=np.inf`) turbulence at the actuators, with `eigenvalues` | closed-loop control; turbulence statistics and priors |
| `DMKLBasisGenerator` | KL modes of the DM surface from its influence functions (double diagonalization) | control with a real DM |
| `ZernikeBasisGenerator` | Noll-normalized Zernikes in Noll, OSA/ANSI or Fringe order; annular Zernikes with `obscuration=` | aberration commands, optical testing, NCPA |
| `FourierBasisGenerator` | sines and cosines, aliased frequencies skipped | frequency response, spatial filtering |
| `HadamardBasisGenerator` | ±1 patterns, Sylvester or Paley matrices (`construction="smallest"`), `selection="balanced"` | low-noise interaction-matrix calibration |
| `ZonalBasisGenerator` | single-actuator pokes | simplest calibration |
| `ZonalFastBasisGenerator` | groups of pokes at least `min_distance` apart (optimal lattice colourings) | calibration in a few frames |

**Shaping a basis.** These options are the same on every generator:
- `ignore_piston=True` makes every mode exactly zero-mean;
- `remove="tiptilt"` (or any `(n_actuators, k)` array, e.g. waffle) keeps other modes out;
- `orthonormalize=True` Gram-Schmidts the modes in order;
- `normalize="rms" | "l2" | "peak" | "pv"` sets the scale.

Unknown options raise `TypeError`. Zernike, Fourier and Hadamard warn when the modes are linearly dependent; `check_rank=False` skips that check, which is most of the cost of a large Zernike basis.

**Zernike helpers.**
- `noll_to_nm(j)` and `nm_to_noll(n, m)` convert between Noll indices and `(n, m)`, with `ansi_to_nm`/`nm_to_ansi` and `fringe_to_nm`/`nm_to_fringe` for the other orderings. `m > 0` is the cosine term, so Noll 2 = (1, 1) is tip along +x and Noll 3 = (1, -1) tilt along +y. They take integers or integer arrays.
- `zernike_modes_on_mask(mask, n_modes, pupil_radius=...)` evaluates Zernikes on the pixels of a 2-D mask and returns `(n_modes, ny, nx)` images, zero outside the mask.

**Petal modes.** Thick spiders (ELT petals) or segment gaps (GMT) cut the pupil into islands whose relative pistons a gradient sensor cannot see (the low-wind effect, segment phasing). `PetalBasis(pupil, threshold=1.0, min_pixels=1)` labels the fragments of a pupil image (4-connected pixels with `pupil >= threshold`; fragments under `min_pixels` pixels dropped) and gives:
- `labels`, the `[y, x]` fragment map, and `indicators`, the `(n, ny, nx)` raw piston of each fragment;
- `modes`, `(n - 1, ny, nx)` piston-free petal modes, orthonormal (unit RMS) over the lit pupil;
- `petal_pistons(phase, smooth_modes=None)`, the fragment pistons of a phase map minus their (area-weighted) mean. Fragment means count tip/tilt as petal, since a tilt has a different mean on every petal; pass a few low-order modes (e.g. `zernike_modes_on_mask(basis.lit, 6, ...)`) and they are fitted jointly, so only the steps are reported. `petal_reconstructor(smooth_modes)` returns the same map as weight images to reuse.

`petal_modes(pupil, ..., raw=False)` returns the mode images directly and `label_fragments(pupil, ...)` only the labels.

```python
petals = aobasis.PetalBasis(pupil, min_pixels=20)        # pupil: (ny, nx) amplitude in [0, 1]
smooth = aobasis.zernike_modes_on_mask(petals.lit, 6, pupil_radius=pupil.shape[1] / 2)
pistons = petals.petal_pistons(phase, smooth_modes=smooth)  # (n_fragments,), tilt-insensitive
```

**Geometry.**
- `make_circular_actuator_grid` builds a square grid by `grid_size` or `pitch`, with actuators on the rim or in cell centres.
- `make_hexagonal_actuator_grid` and `make_concentric_actuator_grid` cover other layouts.
- All three take a central obstruction and spider arms.
- `positions_from_mask` reads a boolean DM map, and any `(N, 2)` array works too.

**Influence functions.** `fit_to_influence_functions` gives least-squares commands whose DM surface matches modes sampled on the pupil. `gaussian_influence_functions` and `make_pupil_points` build the inputs.

**A DM model.** `DeformableMirror` turns commands into a surface on a pixel grid (indexed `[y, x]`) or at arbitrary points, with the same Gaussian influence functions. With actuators on a square lattice it uses their separability (`surface = G_y C G_x^T`), which equals `influence_matrix() @ cmd` to round-off without building the matrix: for 877 actuators on 144×144 pixels, 0.19 ms instead of 4.3 ms (14 ms on one thread) on an 8-core Intel i7-10700 (`examples/dm_surface.py`). `transformed(shift=(dx, dy), rotation_deg=..., magnification=...)` returns the misregistered DM for robustness studies.

```python
actuators = aobasis.make_circular_actuator_grid(8.0, 33)          # pitch 0.25 m
axis = (np.arange(144) - 71.5) * (8.0 / 144)                      # pixel centres
dm = aobasis.DeformableMirror(actuators, pitch=0.25, coupling=0.15, grid=(axis, axis))
surface = dm.surface(np.ones(dm.n_actuators))                     # (144, 144); a batch gives (k, 144, 144)
true_dm = dm.transformed(shift=(0.02, 0.0), rotation_deg=0.5)     # the DM as it really sits in the pupil
```

**Turbulence covariance.** `von_karman_covariance(points_a, points_b=None, r0=..., L0=..., wavelength=500e-9, units="phase" | "opd")` is the von Kármán covariance between points (rad² of phase at the wavelength of `r0`, or m² of OPD), e.g. an MMSE prior on the actuators; `von_karman_covariance_of_distance` takes separations. KL modes are built from the same function. Its structure function tends to `6.88 (r/r0)^(5/3)` for `r << L0`.

**Using and keeping a basis.**
- `command_to_mode_matrix` and `fit_coefficients` give modal coefficients.
- `basis_report` (or `generator.report()`) summarizes rank, conditioning, orthogonality and piston content.
- `save`/`load` (`.npz`) and `save_fits`/`load_fits` store the modes together with the generator's parameters, the `generate()` options, KL eigenvalues and the aobasis version.

**Reproducible.** KL modes follow a fixed sign and degenerate-rotation convention: tip along +x, tilt along +y. The same geometry gives the same modes on every machine, CPU or GPU.

## Tutorials and examples

[`tutorials/`](tutorials) holds eight Jupyter walkthroughs; see [`tutorials/README.md`](tutorials/README.md) for a guide.

| | Notebooks |
|---|---|
| **Getting started** | [01 · Quickstart](tutorials/01_quickstart.ipynb), [02 · A tour of the bases](tutorials/02_tour_of_bases.ipynb) |
| **Everyday use** | [03 · Shaping a basis](tutorials/03_shaping_a_basis.ipynb), [04 · Actuator geometry](tutorials/04_actuator_geometry.ipynb) |
| **In depth** | [05 · KL and turbulence](tutorials/05_kl_and_turbulence.ipynb), [06 · Influence functions and DM KL](tutorials/06_influence_functions.ipynb), [07 · Calibration patterns](tutorials/07_calibration.ipynb), [08 · Saving and sharing](tutorials/08_saving_and_sharing.ipynb) |

[`examples/`](examples) holds ready-to-run scripts (each has `--help`):

```bash
python examples/make_m2c.py kl --grid-size 20 --n-modes 100 --remove tiptilt -o kl.fits --plot kl.png
python examples/dm_kl.py --n-modes 80 -o dmkl.fits
python examples/dm_surface.py --n-pixels 144
python examples/calibration.py
```

## Performance

`python examples/benchmark.py --grid-sizes 16 32 64 --n-modes 100 --gpu --markdown`. Each entry is the best of 3 runs of `generate(100)` (zonal fast: its full pattern set) on a square grid clipped by a 10 m pupil.

Host: an 80-core Arm Neoverse-N1 server, using 4 cores (`OPENBLAS_NUM_THREADS=4`), with an NVIDIA RTX 4060 and an RTX A400 (select one with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...`).

| Basis (100 modes) | 16×16 (172 acts) | 32×32 (740 acts) | 64×64 (3096 acts) |
|---|---|---|---|
| **KL (CPU)** | 0.015 s | 0.138 s | 1.79 s |
| **KL (GPU, RTX 4060)** | 0.012 s | 0.045 s | 0.98 s |
| **KL (GPU, RTX A400)** | 0.018 s | 0.145 s | 4.38 s |
| **Zernike** | 0.003 s | 0.008 s | 0.027 s |
| **Fourier** | 0.005 s | 0.019 s | 0.092 s |
| **Hadamard** | 0.001 s | 0.022 s | 0.146 s |
| **Zonal** | <0.001 s | <0.001 s | 0.002 s |
| **Zonal fast** (3-pitch spacing) | 0.005 s | 0.023 s | 0.122 s |

Full bases (`--n-modes all`) on 3096 actuators take 5.0 s for KL on the CPU (1.6 s on the RTX 4060), 3.3 s for Zernike, 6.9 s for Fourier and 2.9 s for Hadamard. That includes the rank check, which `check_rank=False` skips (a full Zernike basis then takes 0.5 s).

Some notes on these numbers:
- KL costs O(N³). For a few modes the CPU uses a partial eigensolver, which is why 100 modes are faster than the full basis.
- The covariance is evaluated once per distinct actuator separation.
- KL works in float64, so GPU speed follows the card's float64 throughput. The RTX 4060 is about 2× faster than 4 CPU cores at 3096 actuators. The RTX A400, with far less float64 throughput, is slower than the CPU.
- The GPU path always computes every eigenpair (CuPy has no partial eigensolver), so its advantage is largest for full bases.

## Development and testing

```bash
git clone https://github.com/jacotay7/aobasis.git
cd aobasis
pip install -e ".[dev]"
pytest
```

CI runs the test suite on Python 3.8–3.14, on aarch64 and without matplotlib, runs every example script, and executes every tutorial notebook.

CI cannot run the GPU tests: they need CuPy and a CUDA device, and skip without them. Run them locally before changing GPU code:

```bash
pip install cupy-cuda12x
pytest -k gpu -rs   # -rs shows a skip reason if no device is found
```

## Contributing

Contributions are welcome. Open an [issue](https://github.com/jacotay7/aobasis/issues) to report a bug or propose a feature, or send a pull request with a test for the change and an entry in [`CHANGELOG.md`](CHANGELOG.md).

## Contact

Jacob Taylor, jacobataylor7@gmail.com

## License

MIT; see [LICENSE](LICENSE).
