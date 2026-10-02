# aobasis examples

Scripts that do one job end to end. Run any of them with `--help` for all options. The tutorials in [`tutorials/`](../tutorials) explain the ideas behind them.

| Script | What it does |
|---|---|
| [`quickstart.py`](quickstart.py) | the shortest useful program: KL modes for a DM, saved to `kl_m2c.npz` |
| [`make_m2c.py`](make_m2c.py) | build any basis for any geometry (square/hexagonal grid, mask, positions file, obstruction, spiders) and save it as `.npz` or FITS |
| [`dm_kl.py`](dm_kl.py) | KL modes of a DM from Gaussian or measured influence functions |
| [`compare_bases.py`](compare_bases.py) | turbulence residual versus number of modes for KL, Zernike, Fourier (or DM-KL and fitted Zernikes with `--dm`) |
| [`phase_screens.py`](phase_screens.py) | random Von Kármán phase screens on actuators or pupil pixels, from KL eigenvalues |
| [`calibration.py`](calibration.py) | interaction-matrix error of zonal, Hadamard and zonal-fast calibration on a toy AO system |
| [`benchmark.py`](benchmark.py) | generation time of every basis for several DM sizes |

```bash
python examples/make_m2c.py kl --grid-size 20 --n-modes 100 --remove tiptilt -o kl.fits --plot kl.png
python examples/make_m2c.py hadamard --hex-pitch 0.5 --selection balanced -o hadamard.npz
python examples/dm_kl.py --n-modes 80 -o dmkl.fits
python examples/calibration.py --noise 0.02
```

FITS output needs `pip install "aobasis[fits]"` and plots need `pip install "aobasis[plot]"`.
