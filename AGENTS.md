# AGENTS.md

Guide for agents (and humans) working in `aobasis`. Keep it accurate: update
it in the same change that alters layout, commands or conventions, and add a
Gotchas entry when something led you astray.

## What this repository is

Modal bases for adaptive-optics deformable mirrors (KL, DM KL, Zernike,
Fourier, Hadamard, zonal). Every generator takes actuator positions and
returns an `(n_actuators, n_modes)` modal-to-command matrix; the pixel-image
helpers (`zernike_modes_on_mask`, the petal modes) return `(n_modes, ny, nx)`
stacks instead. Pure Python on NumPy and SciPy; CuPy is optional (GPU KL). It
supports Python 3.8-3.14, so no syntax or typing newer than 3.8 at runtime.

## Layout

```text
src/aobasis/base.py      BasisGenerator: options shared by all bases, rank check, save/load
src/aobasis/kl.py        Von Karman / Kolmogorov KL (CPU and CuPy), mode sign/rotation convention
src/aobasis/dm_kl.py     KL of the DM surface from influence functions
src/aobasis/zernike.py   Zernike (Noll/ANSI/Fringe, annular) and index helpers
src/aobasis/fourier.py, hadamard.py, zonal.py   the other bases
src/aobasis/influence.py, utils.py, analysis.py geometry, influence functions, reports
src/aobasis/dm.py        DeformableMirror: Gaussian-IF surfaces (separable fast path), misregistration
src/aobasis/turbulence.py von_karman_covariance: the one von Karman covariance (KL uses it)
src/aobasis/petal.py     petal (fragment-piston) modes of a pupil image; joint petal/low-order fit
tests/                   pytest; GPU tests skip without CuPy and a CUDA device
examples/                runnable scripts (CI runs them); examples/benchmark.py times the bases
tutorials/               notebooks (CI executes them)
```

## Quality gate

```bash
pip install -e ".[dev]"
pytest --cov=src/aobasis --cov-fail-under=80
pytest -k gpu -rs          # on a machine with CuPy and a CUDA device
```

Every change gets a `CHANGELOG.md` entry. Releases: bump `version` in
`pyproject.toml`, date the changelog, then create a GitHub release; the
publish workflow uploads to PyPI.

## Rules

- MIT-compatible code only.
- Modes are a contract: the same geometry and options must give the same
  modes on every machine, CPU or GPU. Performance work must keep outputs
  bit-identical (or say exactly what changes and why in the changelog);
  check with a saved before/after set of bases, not only the tests.
- No planning or status files in the repo. A minor issue worked around
  rather than fixed gets a GitHub issue, linked from the workaround.

## Gotchas

- KL modes are canonicalized (`_canonical_eigenvectors`): signs and rotations
  within degenerate clusters come from projections on circular harmonics, so
  eigensolver differences cancel. Changing the eigensolver call itself (driver,
  `subset_by_index` threshold) still changes the modes in the last bits and
  can rotate near-degenerate pairs; it is not a free speed-up.
- The annular Zernike recurrence coefficients (`_annular_recurrence`) depend on
  how many were requested (the quadrature grows with the count), so annular
  radial functions must not be cached across counts. The circular Jacobi
  recurrence has no such dependence and `generate()` resumes it per `m`.
- `np.unique(..., return_inverse=True)` argsorts its input; on the `N^2/2`
  actuator separations that was the slowest step of the KL covariance. Sort
  the values and `searchsorted` instead (same indices).
- For full bases the rank check (a column-pivoted QR, LAPACK `geqp3`) is most
  of the time of Zernike, Fourier and Hadamard; `check_rank=False` skips it.
- The von Karman covariance uses the rounded `6.88` (in `_von_karman_variance`)
  rather than `2 (24/5 Gamma(6/5))^(5/6) = 6.8839`, so its structure function
  tends to exactly `6.88 (r/r0)^(5/3)`. Changing it changes every KL
  eigenvalue (and modes in the last bits). pyturb's `phase_covariance` uses
  the unrounded value (0.06% higher variance).
- `2 [C(0) - C(r)]` from `von_karman_covariance` loses digits when
  `L0 / r0` is huge (`C(0)` grows as `(L0/r0)^(5/3)`); test the Kolmogorov
  limit with `L0 <= 1e4` m.
- `DeformableMirror` picks the separable path from a cost estimate; a rotated
  DM (or random actuators) has as many lattice rows/columns as actuators and
  falls back to the cached influence matrix (`n_pixels x n_actuators`
  floats). `method="separable"` still works there, only slower.
- The GPU KL path always computes every eigenpair (CuPy has no partial
  solver) and in float64, so a card with little float64 throughput (e.g. an
  RTX A400) is slower than the CPU.
