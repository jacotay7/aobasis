"""Build a modal-to-command matrix (M2C) and save it.

Examples::

    python examples/make_m2c.py kl --grid-size 20 --n-modes 100 -o kl.fits
    python examples/make_m2c.py zernike --pitch 0.5 --n-modes 50 --orthonormalize --normalize rms -o z.npz
    python examples/make_m2c.py kl --positions my_dm.txt --remove tiptilt --n-modes 200 -o kl.npz --plot kl.png
    python examples/make_m2c.py zonal_fast --mask dm97.npy --mask-pitch 1.0 --min-distance 3 -o fast.npz

The geometry comes from ``--positions`` (an ``(N, 2)`` text or ``.npy`` file in
metres, centred, in DM order), ``--mask`` (a boolean ``.npy`` actuator map), a
hexagonal grid (``--hex-pitch``) or a square grid clipped by the pupil
(``--grid-size`` or ``--pitch``). The output format follows the file suffix:
``.npz`` or ``.fits`` (FITS needs ``pip install "aobasis[fits]"``).
"""

import argparse
from pathlib import Path

import numpy as np

import aobasis

BASES = ("kl", "zernike", "fourier", "hadamard", "zonal", "zonal_fast")


def actuator_positions(args):
    if args.positions:
        path = Path(args.positions)
        positions = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
        return positions - positions.mean(axis=0) if args.centre else positions
    if args.mask:
        return aobasis.positions_from_mask(np.load(args.mask).astype(bool), args.mask_pitch)
    masks = dict(obscuration=args.obscuration, n_spiders=args.spiders, spider_width=args.spider_width)
    if args.hex_pitch:
        return aobasis.make_hexagonal_actuator_grid(args.diameter, args.hex_pitch, **masks)
    if args.pitch:
        return aobasis.make_circular_actuator_grid(args.diameter, pitch=args.pitch, **masks)
    return aobasis.make_circular_actuator_grid(args.diameter, args.grid_size, **masks)


def build(args, positions):
    common = dict(normalize=args.normalize)
    shaped = dict(common, ignore_piston=args.ignore_piston, remove=args.remove or None, orthonormalize=args.orthonormalize)
    n = args.n_modes or len(positions) - (1 if args.ignore_piston else 0) - (2 if args.remove == "tiptilt" else 0)
    if args.basis == "kl":
        gen = aobasis.KLBasisGenerator(
            positions, fried_parameter=args.r0, outer_scale=args.L0, wavelength=args.wavelength, use_gpu=args.gpu
        )
        return gen, gen.generate(n, **shaped)
    if args.basis == "zernike":
        gen = aobasis.ZernikeBasisGenerator(positions, pupil_radius=args.diameter / 2, obscuration=args.obscuration)
        return gen, gen.generate(n, ordering=args.ordering, **shaped)
    if args.basis == "fourier":
        gen = aobasis.FourierBasisGenerator(positions, pupil_diameter=args.diameter)
        return gen, gen.generate(n, **shaped)
    if args.basis == "hadamard":
        gen = aobasis.HadamardBasisGenerator(positions)
        return gen, gen.generate(n, construction="smallest", selection=args.selection, **shaped)
    if args.basis == "zonal":
        gen = aobasis.ZonalBasisGenerator(positions)
        return gen, gen.generate(args.n_modes or len(positions), **common)
    gen = aobasis.ZonalFastBasisGenerator(positions, min_distance=args.min_distance * nearest_pitch(positions))
    return gen, gen.generate(args.n_modes, **common)


def nearest_pitch(positions):
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(positions).query(positions, k=2)
    return float(np.median(distances[:, 1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("basis", choices=BASES)
    geometry = parser.add_argument_group("geometry")
    geometry.add_argument("--diameter", type=float, default=10.0, help="pupil diameter in metres (default 10)")
    geometry.add_argument("--grid-size", type=int, default=20, help="actuators across a square grid (default 20)")
    geometry.add_argument("--pitch", type=float, help="square-grid pitch in metres, instead of --grid-size")
    geometry.add_argument("--hex-pitch", type=float, help="hexagonal grid with this pitch")
    geometry.add_argument("--positions", help="(N, 2) actuator positions, .npy or text, metres")
    geometry.add_argument("--centre", action="store_true", help="subtract the mean of --positions")
    geometry.add_argument("--mask", help="boolean .npy actuator map")
    geometry.add_argument("--mask-pitch", type=float, default=1.0, help="pitch of --mask cells in metres")
    geometry.add_argument("--obscuration", type=float, default=0.0, help="central obstruction (fraction of diameter)")
    geometry.add_argument("--spiders", type=int, default=0, help="number of spider arms")
    geometry.add_argument("--spider-width", type=float, default=0.0, help="spider width in metres")
    modes = parser.add_argument_group("modes")
    modes.add_argument("--n-modes", type=int, help="number of modes (default: as many as possible)")
    modes.add_argument("--keep-piston", dest="ignore_piston", action="store_false", help="keep piston in the basis")
    modes.add_argument("--remove", choices=("tip", "tilt", "tiptilt"), help="also remove these modes")
    modes.add_argument("--orthonormalize", action="store_true")
    modes.add_argument("--normalize", choices=("rms", "l2", "peak", "pv"))
    modes.add_argument("--r0", type=float, default=0.16, help="KL: Fried parameter at 500 nm (default 0.16 m)")
    modes.add_argument("--L0", type=float, default=30.0, help="KL: outer scale (default 30 m; inf for Kolmogorov)")
    modes.add_argument("--wavelength", type=float, help="KL: report eigenvalues at this wavelength (m)")
    modes.add_argument("--gpu", action="store_true", help="KL: use CuPy")
    modes.add_argument("--ordering", choices=("noll", "ansi", "fringe"), default="noll", help="Zernike ordering")
    modes.add_argument("--selection", choices=("first", "balanced"), default="first", help="Hadamard columns")
    modes.add_argument("--min-distance", type=float, default=3.0, help="zonal_fast: in actuator pitches (default 3)")
    parser.add_argument("-o", "--output", help="output .npz or .fits")
    parser.add_argument("--plot", help="save a PNG of the first modes")
    args = parser.parse_args(argv)

    try:
        positions = actuator_positions(args)
        generator, m2c = build(args, positions)
    except ValueError as err:
        parser.error(str(err))
    print(f"{args.basis}: M2C {m2c.shape[0]} actuators x {m2c.shape[1]} modes")
    print(generator.report())
    if args.output:
        if args.output.endswith(".fits"):
            generator.save_fits(args.output, overwrite=True)
        else:
            generator.save(args.output)
        print("saved", args.output)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        generator.plot(count=min(12, m2c.shape[1]), outfile=args.plot, title_prefix=args.basis)
        print("plotted", args.plot)
    return m2c


if __name__ == "__main__":
    main()
