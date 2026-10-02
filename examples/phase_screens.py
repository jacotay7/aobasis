"""Draw random Von Kármán phase screens with KL modes.

With KL modes M and eigenvalues lambda, phi = M a with independent
a_k ~ N(0, lambda_k) has exactly the turbulence statistics of the model,
sampled at the given points (actuators, or pupil pixels with --pixels).

    python examples/phase_screens.py --n-screens 100 -o screens.npy
    python examples/phase_screens.py --pixels 48 --wavelength 1.65e-6 --plot screens.png
"""

import argparse

import numpy as np

import aobasis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diameter", type=float, default=8.0)
    parser.add_argument("--grid-size", type=int, default=16, help="actuators across (ignored with --pixels)")
    parser.add_argument("--pixels", type=int, help="sample on an N-pixel pupil instead of actuators")
    parser.add_argument("--r0", type=float, default=0.15, help="Fried parameter at 500 nm")
    parser.add_argument("--L0", type=float, default=25.0, help="outer scale; inf for Kolmogorov")
    parser.add_argument("--wavelength", type=float, help="phase in radians at this wavelength (default 500 nm)")
    parser.add_argument("--n-modes", type=int, help="KL modes used (default: all but piston)")
    parser.add_argument("--n-screens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", help="save (n_points, n_screens) screens to this .npy")
    parser.add_argument("--plot", help="save a PNG of the first screens")
    args = parser.parse_args(argv)

    if args.pixels:
        points = aobasis.make_pupil_points(args.diameter, args.pixels)
    else:
        points = aobasis.make_circular_actuator_grid(args.diameter, args.grid_size)
    kl = aobasis.KLBasisGenerator(points, fried_parameter=args.r0, outer_scale=args.L0, wavelength=args.wavelength)
    modes = kl.generate(args.n_modes or len(points) - 1, ignore_piston=True)
    rng = np.random.default_rng(args.seed)
    screens = modes @ (rng.standard_normal((modes.shape[1], args.n_screens)) * np.sqrt(kl.eigenvalues)[:, None])
    rms = np.sqrt(np.mean(screens**2, axis=0))
    print(f"{args.n_screens} screens on {len(points)} points; RMS phase {rms.mean():.2f} ± {rms.std():.2f} rad")
    if args.output:
        np.save(args.output, screens)
        print("saved", args.output)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        aobasis.plot_basis_modes(
            screens, points, count=min(8, args.n_screens), outfile=args.plot, title_prefix="screen",
            interpolate=True, cmap="RdBu_r",
        )
        print("plotted", args.plot)
    return screens


if __name__ == "__main__":
    main()
