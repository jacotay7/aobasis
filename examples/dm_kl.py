"""KL modes of a deformable mirror from its influence functions.

Uses Gaussian influence functions on a square grid with one ring of actuators
outside the pupil, or measured ones: --influence takes an (n_actuators, ny, nx)
.npy cube and --pupil the matching boolean (ny, nx) mask; --actuators gives the
actuator positions (.npy or text, metres, same order as the cube).

    python examples/dm_kl.py --n-modes 100 -o dmkl.fits
    python examples/dm_kl.py --influence ifs.npy --pupil pupil.npy --pixel-size 0.02 \\
        --actuators acts.txt --remove tiptilt -o dmkl.npz
"""

import argparse
from pathlib import Path

import numpy as np

import aobasis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diameter", type=float, default=8.0)
    parser.add_argument("--grid-size", type=int, default=16, help="actuators across the pupil (Gaussian DM)")
    parser.add_argument("--coupling", type=float, default=0.15, help="Gaussian inter-actuator coupling")
    parser.add_argument("--pixels", type=int, default=40, help="pupil pixels across (Gaussian DM)")
    parser.add_argument("--influence", help="measured influence functions, (n_actuators, ny, nx) .npy")
    parser.add_argument("--pupil", help="boolean (ny, nx) .npy pupil mask for --influence")
    parser.add_argument("--pixel-size", type=float, help="pupil pixel size in metres for --influence")
    parser.add_argument("--actuators", help="actuator positions for --influence (.npy or text)")
    parser.add_argument("--n-modes", type=int, default=50)
    parser.add_argument("--r0", type=float, default=0.15)
    parser.add_argument("--L0", type=float, default=25.0)
    parser.add_argument("--remove", choices=("tip", "tilt", "tiptilt"))
    parser.add_argument("-o", "--output", help="output .npz or .fits")
    args = parser.parse_args(argv)

    if args.influence:
        if not (args.pupil and args.pixel_size and args.actuators):
            parser.error("--influence needs --pupil, --pixel-size and --actuators")
        pupil = np.load(args.pupil).astype(bool)
        influence = np.load(args.influence)[:, pupil].T
        points = aobasis.positions_from_mask(pupil, args.pixel_size)
        path = Path(args.actuators)
        actuators = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    else:
        points = aobasis.make_pupil_points(args.diameter, args.pixels)
        pitch = args.diameter / (args.grid_size - 1)
        actuators = aobasis.make_circular_actuator_grid(args.diameter + 2 * pitch, pitch=pitch)
        influence = aobasis.gaussian_influence_functions(actuators, points, coupling=args.coupling, pitch=pitch)

    gen = aobasis.DMKLBasisGenerator(actuators, points, influence, fried_parameter=args.r0, outer_scale=args.L0)
    m2c = gen.generate(args.n_modes, ignore_piston=True, remove=args.remove)
    surfaces = gen.surfaces
    print(f"DM-KL: {m2c.shape[0]} actuators x {m2c.shape[1]} modes, {len(points)} pupil points")
    print("surfaces orthonormal:", np.allclose(surfaces.T @ surfaces / len(points), np.eye(m2c.shape[1]), atol=1e-8))
    print("first eigenvalues [rad²]:", np.round(gen.eigenvalues[:5], 3))
    if args.output:
        if args.output.endswith(".fits"):
            gen.save_fits(args.output, overwrite=True)
        else:
            gen.save(args.output)
        print("saved", args.output)
    return m2c


if __name__ == "__main__":
    main()
