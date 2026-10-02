# Changelog

## Unreleased

### Fixed

- **Zernike precision at high order (#13).** The radial polynomial is evaluated with the Jacobi three-term recurrence instead of the explicit factorial sum, which cancelled catastrophically from radial order ~46 (Noll j ≈ 1100) on. A full Zernike basis on a 64×64 grid had entries up to 3e13; modes are now accurate to ~1e-13 beyond n = 100.
- **CuPy is imported lazily (#19).** `import aobasis` no longer imports CuPy or builds the GPU kernel; both happen the first time `KLBasisGenerator(..., use_gpu=True)` is created. `aobasis.kl.HAS_CUPY` and `aobasis.kl.cp` are gone; use `aobasis.kl._load_cupy()` if you need to know whether CuPy is available.
- **GPU KL covariance accuracy (#15).** The CuPy `K_{5/6}` kernel switched to a six-term asymptotic expansion at z = 2, where it is only good to ~2e-3, and one of its coefficients was wrong. It now uses Temme's series and Steed's continued fraction (Numerical Recipes `bessik`), which match `scipy.special.kv` to ~1e-14, so `use_gpu=True` gives the same covariance as the CPU to ~1e-12 (it was off by up to 0.2%).
- **Faster rank check (#18).** The rank-deficiency warning estimates the rank with a column-pivoted QR instead of an SVD (same tolerance as `np.linalg.matrix_rank`). Full-size Zernike and Hadamard bases on 3096 actuators build in 4.2 s and 3.1 s instead of 10.6 s and 7.4 s.

### Changed

- **CI and packaging (#25).** CI runs on Python 3.8–3.14, on aarch64 (`ubuntu-24.04-arm`) and without the `plot` extra, and on every pull request. The publish workflow runs the tests and checks that the release tag matches the package version. aobasis ships a `py.typed` marker. Plotting tests moved to `tests/test_plotting.py` and skip without matplotlib; GPU tests run whenever CuPy and a CUDA device are present.

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
