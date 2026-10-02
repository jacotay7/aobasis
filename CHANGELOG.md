# Changelog

## Unreleased

### Fixed

- **Zernike precision at high order (#13).** The radial polynomial is evaluated with the Jacobi three-term recurrence instead of the explicit factorial sum, which cancelled catastrophically from radial order ~46 (Noll j ≈ 1100) on. A full Zernike basis on a 64×64 grid had entries up to 3e13; modes are now accurate to ~1e-13 beyond n = 100.
- **CuPy is imported lazily (#19).** `import aobasis` no longer imports CuPy or builds the GPU kernel; both happen the first time `KLBasisGenerator(..., use_gpu=True)` is created. `aobasis.kl.HAS_CUPY` and `aobasis.kl.cp` are gone; use `aobasis.kl._load_cupy()` if you need to know whether CuPy is available.
- **GPU KL covariance accuracy (#15).** The CuPy `K_{5/6}` kernel switched to a six-term asymptotic expansion at z = 2, where it is only good to ~2e-3, and one of its coefficients was wrong. It now uses Temme's series and Steed's continued fraction (Numerical Recipes `bessik`), which match `scipy.special.kv` to ~1e-14, so `use_gpu=True` gives the same covariance as the CPU to ~1e-12 (it was off by up to 0.2%).
- **Faster rank check (#18).** The rank-deficiency warning estimates the rank with a column-pivoted QR instead of an SVD (same tolerance as `np.linalg.matrix_rank`). Full-size Zernike and Hadamard bases on 3096 actuators build in 4.2 s and 3.1 s instead of 10.6 s and 7.4 s.
- **`ignore_piston` removes piston (#14).** Zernike, Fourier and Hadamard modes with `ignore_piston=True` are now exactly zero-mean, with or without `orthonormalize`; before, only the piston column was dropped, and the sampled modes kept up to 36% of their RMS as piston. Hadamard entries are then no longer ±1. With piston removed, Zernike allows at most `n_actuators - 1` modes, like the other bases.
- **Degenerate grids (#16).** `make_circular_actuator_grid(D, 1)` returns one actuator at the centre and `(D, 2)` raises; both used to return an empty array. Generators reject an empty `positions` array. The docstring states the pitch, `D / (grid_size - 1)`, with the outermost actuators on the rim.
- **Loaded bases (#21).** `ConcreteBasis.generate(n)` stores the returned modes in `modes` like every other generator, and keeps all saved modes in `full_modes`, so a later call can ask for more. `load(path)` accepts the path without the `.npz` suffix that `save` adds.
- **Reproducible KL modes (#20).** KL eigenvectors used to come with an arbitrary sign, and an arbitrary rotation within each group of equal eigenvalues (symmetric pupils have many), so CPU, GPU and different LAPACK builds gave different modes. A fixed convention now picks them (see `KLBasisGenerator.generate`); CPU and GPU modes agree to 1e-8. Individual KL modes therefore differ from 1.2.0 (same eigenvalues and subspaces); re-derive interaction matrices built from KL modes.
- **Fourier conditioning is stated correctly (#40).** Fourier modes are each independent of the ones before them, but a basis near `n_actuators` modes on a circular pupil is numerically singular (σ_min/σ_max ≈ 1e-16 at full size on 1876 actuators), which the docs used to deny. Its warning now points to `orthonormalize=True`, which gives an accurate orthonormal basis with the same nested spans.

### Added

- **`remove=` on Zernike, Fourier, Hadamard and KL `generate()` (#27)** keeps given modes out of the basis: names (`"piston"`, `"tip"`, `"tilt"`, `"tiptilt"`), `(n_actuators, k)` arrays, or a list of them. Zernike and Hadamard skip candidates that lie inside the removed modes (tip and tilt for `"tiptilt"`), Fourier picks frequencies independent of them, and KL diagonalizes `P C P` with `P = I - U Uᵀ`. New helpers `aobasis.project_out(modes, subspace)` and `aobasis.removal_basis(positions, remove)`.
- **`normalize=` on every generator (#26)**: `"rms"`, `"l2"`, `"peak"` or `"pv"`, applied after piston removal and orthonormalization, and the helper `aobasis.normalize_modes`. KL `eigenvalues` are rescaled to stay the variance of each returned mode's coefficient.
- **KL options (#28).** `outer_scale=np.inf` gives Kolmogorov turbulence (with `ignore_piston=True`; checked against Noll's Δ₁ and Δ₃). `r0_wavelength` and `wavelength` report `eigenvalues` at another wavelength. The covariance is evaluated once per distinct actuator separation, and only the leading eigenpairs are computed when few are needed: 100 KL modes on 3096 actuators take 1.8 s instead of 8.1 s, the full basis 5.2 s.
- **Influence-function fitting (#11).** `fit_to_influence_functions(modes, influence_functions)` gives least-squares DM commands whose surfaces best match modes sampled on pupil points, with `rcond` truncation, Tikhonov `regularization`, `orthonormalize` (surfaces orthonormal over the pupil) and per-mode residuals. `gaussian_influence_functions` and `make_pupil_points` build the inputs.
- **`DMKLBasisGenerator` (#29)**: KL modes of a DM from its influence functions by double diagonalization (Gendron 1995). The modes are commands whose DM surfaces are orthonormal over the pupil and whose coefficients are statistically independent; `ignore_piston`/`remove=` keep every surface exactly orthogonal to the removed pupil modes, and the KL sign/rotation convention applies.
- **Richer save files and FITS (#30).** `save` also stores the generator's `parameters`, the `generate_options` of the last call, KL `eigenvalues` and the `aobasis_version`; `load` exposes them on the returned `ConcreteBasis` (older files still load). `save_fits`/`load_fits` write and read the same content as FITS through the new `fits` extra (astropy).
- **Zernike options (#31).** `generate(..., ordering="ansi" | "fringe")` besides Noll; `ZernikeBasisGenerator(..., obscuration=ε)` gives annular Zernike polynomials orthonormal over the annulus (stable to radial order > 100, matching Mahajan's closed forms); `pupil_radius` defaults to the largest actuator radius.
- **Geometry helpers (#32).** `make_hexagonal_actuator_grid(diameter, pitch)`; `make_circular_actuator_grid(..., pitch=)` as an alternative to `grid_size` and `rim=False` for cell-centred actuators; `obscuration=` and spider arms (`n_spiders`, `spider_width`, `spider_angle`) on all three grid helpers.
- **Hadamard constructions and selection (#33).** `construction="smallest"` uses the smallest Sylvester-doubled Paley Hadamard matrix of order ≥ the actuator count (e.g. 104 instead of 128 for 97 actuators; exact, untruncated and orthogonal for the 88-, 276-, 740-, 1876- and 3096-actuator circular grids). `selection="balanced"` takes the most piston-free columns first, exactly zero-mean when such columns exist. New helpers `aobasis.hadamard.hadamard_matrix` and `smallest_hadamard_order`.
- **Basis helpers (#35).** `fit_coefficients(modes, commands)` and `command_to_mode_matrix(modes)` give least-squares modal coefficients (C2M, with `rcond` and `regularization`); `basis_report(modes)` and `BasisGenerator.report()` summarize rank, condition number, largest inter-mode cosine and per-mode piston content.

### Changed

- **CI and packaging (#25).** CI runs on Python 3.8–3.14, on aarch64 (`ubuntu-24.04-arm`) and without the `plot` extra, and on every pull request. The publish workflow runs the tests and checks that the release tag matches the package version. aobasis ships a `py.typed` marker. Plotting tests moved to `tests/test_plotting.py` and skip without matplotlib; GPU tests run whenever CuPy and a CUDA device are present.
- **Unknown `generate()` keywords raise `TypeError` (#17).** Every generator used to accept and silently ignore any keyword, so typos such as `orthonormalise=True` or unsupported options such as `ZonalFastBasisGenerator.generate(orthonormalize=True)` did nothing.
- **Rank warning with `orthonormalize=True`.** The warning now flags modes that are numerically combinations of the modes before them (their orthonormalized versions would be rounding noise) instead of the rank of the raw matrix, so a well-defined orthonormalization of an ill-conditioned basis no longer warns.
- **KL documentation (#24)** states the units: `eigenvalues` are rad² at `wavelength`; positions, `fried_parameter` and `outer_scale` share one length unit; modes are sampled at actuator positions, not fitted to influence functions.
- **Zonal fast needs fewer modes and is faster (#34).** Actuators on any 2-D lattice (square, rectangular, hexagonal, oblique) get the fewest-colour sublattice colouring, compared with the DSATUR colouring, and the smaller wins: on a 32×32 grid, 8 instead of 9 modes at 2.5 pitch and 12 instead of 16 at 3.5 pitch. DSATUR uses a heap (same colourings, ~30× faster: 6000 actuators in 0.1 s). `generate(signs="random", seed=0)` gives random-sign pokes.

## 1.2.0

### Changed

- **matplotlib is optional.** It moved to the `plot` extra (`pip install aobasis[plot]`), so installing aobasis as a library no longer pulls in matplotlib. `plot_basis_modes` and `BasisGenerator.plot` raise an `ImportError` naming the extra when matplotlib is missing. The `dev` extra still includes it.

## 1.1.0

### Fixed

- **Zernike Noll order.** Even `j` are now cosine terms and odd `j` sine terms for every radial order. 1.0.x swapped them for n ≡ 2, 3 (mod 4): for example, `j=5` returned (2, 2) instead of (2, -2).
- **Zernike normalization.** Modes carry the Noll factor `sqrt(n+1)` (times `sqrt(2)` for m ≠ 0), so each has unit RMS over the unit disk.
- **Zernike outside the pupil.** Actuators outside `pupil_radius` are evaluated at their true radius, with a `RuntimeWarning`. 1.0.x clipped the radius but kept the angle, which silently distorted those values.
- **Mode counts.** Zernike and Hadamard raise `ValueError` for more modes than actuators. Every generator warns when the basis is rank-deficient.
- **Fourier rank.** Aliased frequencies and sine terms that vanish on the grid are skipped, so the basis has full rank. `ValueError` is raised if the grid cannot support `n_modes` independent modes.
- **Hadamard dtype.** Modes are float, not int.
- **KL piston removal.** `ignore_piston=True` diagonalizes the piston-removed covariance, so every mode has exactly zero mean. 1.0.x dropped the leading eigenvector, which is only approximately piston.
- **CuPy fallback.** The fallback warns with `warnings.warn` instead of `print`.
- **Loading.** `BasisGenerator.load` records the saved `basis_type` on the returned `ConcreteBasis`, and `SomeGenerator.load` raises if the file holds a different type.
- **Import cost.** `import aobasis` no longer imports `matplotlib.pyplot`; the plotting helper imports it when called.

### Added

- `orthonormalize=True` on the Zernike, Fourier and Hadamard `generate()`, and the helper `aobasis.orthonormalize_modes`. KL accepts the flag too; its modes are already orthonormal.
- `ignore_piston` for Hadamard.
- `aobasis.positions_from_mask(mask, pitch)`.

### Changed

- Zernike amplitudes are scaled by the Noll factors, and several Zernike modes change index (see Fixed). Re-derive any interaction matrices or reconstructors built from 1.0.x Zernike modes.
