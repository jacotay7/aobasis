"""How much turbulence does each basis correct, per mode?

For an orthonormal basis M, the piston-removed Von Kármán variance it captures
is trace(M^T C M). This script compares KL, Zernike and Fourier bases (all
orthonormalized) and, with --dm, DM-KL and fitted Zernikes on a Gaussian-IF
DM, and prints or plots the residual fraction versus the number of modes.

    python examples/compare_bases.py --grid-size 16 --n-modes 100 --plot compare.png
    python examples/compare_bases.py --dm
"""

import argparse

import numpy as np

import aobasis


def residual_curve(modes, cov):
    captured = np.cumsum(np.einsum("ik,ij,jk->k", modes, cov, modes))
    return 1.0 - captured / np.trace(cov)


def actuator_curves(args):
    positions = aobasis.make_circular_actuator_grid(args.diameter, args.grid_size)
    kl = aobasis.KLBasisGenerator(positions, fried_parameter=args.r0, outer_scale=args.L0)
    p = np.eye(len(positions)) - 1.0 / len(positions)
    cov = p @ kl._von_karman_covariance_cpu() @ p
    n = args.n_modes
    return {
        "KL": residual_curve(kl.generate(n, ignore_piston=True), cov),
        "Zernike": residual_curve(
            aobasis.ZernikeBasisGenerator(positions).generate(n, ignore_piston=True, orthonormalize=True), cov
        ),
        "Fourier": residual_curve(
            aobasis.FourierBasisGenerator(positions, args.diameter).generate(n, ignore_piston=True, orthonormalize=True),
            cov,
        ),
    }


def dm_curves(args):
    points = aobasis.make_pupil_points(args.diameter, 40)
    pitch = args.diameter / (args.grid_size - 1)
    actuators = aobasis.make_circular_actuator_grid(args.diameter + 2 * pitch, pitch=pitch)
    influence = aobasis.gaussian_influence_functions(actuators, points, pitch=pitch)
    turbulence = aobasis.KLBasisGenerator(points, fried_parameter=args.r0, outer_scale=args.L0)
    p = np.eye(len(points)) - 1.0 / len(points)
    cov = p @ turbulence._von_karman_covariance_cpu() @ p / len(points) ** 2
    n = args.n_modes
    dmkl = aobasis.DMKLBasisGenerator(actuators, points, influence, fried_parameter=args.r0, outer_scale=args.L0)
    dmkl.generate(n, ignore_piston=True)
    zern = aobasis.ZernikeBasisGenerator(points).generate(n, ignore_piston=True)
    zern_surfaces = influence @ aobasis.fit_to_influence_functions(zern, influence, orthonormalize=True)
    total = np.trace(cov) * len(points)  # S^T S / n = I normalization
    return {
        "DM-KL": 1 - np.cumsum(np.einsum("ik,ij,jk->k", dmkl.surfaces, cov, dmkl.surfaces)) / total,
        "Zernike fitted on DM": 1 - np.cumsum(np.einsum("ik,ij,jk->k", zern_surfaces, cov, zern_surfaces)) / total,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diameter", type=float, default=8.0)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--n-modes", type=int, default=80)
    parser.add_argument("--r0", type=float, default=0.15)
    parser.add_argument("--L0", type=float, default=25.0)
    parser.add_argument("--dm", action="store_true", help="compare on a DM with Gaussian influence functions")
    parser.add_argument("--plot", help="save the curves to this PNG")
    args = parser.parse_args(argv)

    curves = dm_curves(args) if args.dm else actuator_curves(args)
    marks = [k for k in (10, 25, 50, 100, 200) if k <= args.n_modes]
    print("residual / total variance after correcting n modes")
    print(f"{'basis':<22}" + "".join(f"{f'n={k}':>10}" for k in marks))
    for name, curve in curves.items():
        print(f"{name:<22}" + "".join(f"{curve[k - 1]:>10.4f}" for k in marks))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for name, curve in curves.items():
            plt.semilogy(np.arange(1, len(curve) + 1), curve, label=name)
        plt.xlabel("modes corrected")
        plt.ylabel("residual / total variance")
        plt.legend()
        plt.savefig(args.plot, dpi=120)
        print("plotted", args.plot)
    return curves


if __name__ == "__main__":
    main()
