# Changelog

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
