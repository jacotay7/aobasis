"""Compare zonal, Hadamard and zonal-fast interaction-matrix calibration.

A toy system (Gaussian-IF DM with a ring outside the pupil, a Shack-Hartmann-like
gradient sensor) is calibrated with each kind of pattern under slope noise, and
the relative error of each measured interaction matrix is printed. See
tutorials/07_calibration.ipynb for the walkthrough.

    python examples/calibration.py
    python examples/calibration.py --noise 0.01 --min-distance 4 --repeats 3
"""

import argparse

import numpy as np

import aobasis


def slope_sensor(points, n_sub, n_pix, diameter):
    pixel = diameter / n_pix
    col, row = np.rint(points / pixel + 0.5 * (n_pix - 1)).astype(int).T
    size = n_pix // n_sub
    rows, centres = [], []
    for sy in range(n_sub):
        for sx in range(n_sub):
            inside = np.flatnonzero((col // size == sx) & (row // size == sy))
            if len(inside) < size * size / 2:
                continue
            for local in (col[inside] % size, row[inside] % size):
                high, low = inside[local >= size // 2], inside[local < size // 2]
                g = np.zeros(len(points))
                if len(high) and len(low):
                    g[high], g[low] = 1 / len(high), -1 / len(low)
                rows.append(g / (size / 2 * pixel))
                centres.append(points[inside].mean(axis=0))
    return np.array(rows), np.array(centres)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actuators-across", type=int, default=15, help="actuator pitches across the pupil")
    parser.add_argument("--subapertures", type=int, default=12)
    parser.add_argument(
        "--noise", type=float, default=0.04, help="slope noise per frame, relative to the largest response to a unit poke"
    )
    parser.add_argument("--min-distance", type=float, default=5.0, help="zonal fast, in pitches")
    parser.add_argument("--repeats", type=int, default=9, help="zonal-fast repeats for the equal-budget row")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    diameter, n_pix = 10.0, 4 * args.subapertures
    rng = np.random.default_rng(args.seed)
    points = aobasis.make_pupil_points(diameter, n_pix)
    pitch = diameter / args.actuators_across
    actuators = aobasis.make_circular_actuator_grid(diameter + 2 * pitch, pitch=pitch)
    influence = aobasis.gaussian_influence_functions(actuators, points, pitch=pitch)
    wfs, centres = slope_sensor(points, args.subapertures, n_pix, diameter)
    d_true = wfs @ influence
    sigma = args.noise * np.abs(d_true).max()
    n_act = len(actuators)

    def measure(patterns, repeats=1):
        return d_true @ patterns + sigma * rng.standard_normal((len(wfs), patterns.shape[1])) / np.sqrt(repeats)

    def error(d_hat):
        return np.linalg.norm(d_hat - d_true) / np.linalg.norm(d_true)

    distance = np.linalg.norm(centres[:, None, :] - actuators[None], axis=-1)

    def zonal_fast(patterns, y, rho):
        d_hat = np.zeros_like(d_true)
        for k in range(patterns.shape[1]):
            members = np.flatnonzero(patterns[:, k])
            d = distance[:, members]
            nearest, close = members[np.argmin(d, axis=1)], d.min(axis=1) <= rho
            d_hat[np.flatnonzero(close), nearest[close]] = y[close, k]
        return d_hat

    hadamard = aobasis.HadamardBasisGenerator(actuators).generate(n_act, construction="smallest")
    fast = aobasis.ZonalFastBasisGenerator(actuators, min_distance=args.min_distance * pitch).generate()
    rho = args.min_distance / 2 * pitch
    rows = [
        ("zonal", n_act, measure(np.eye(n_act))),
        ("Hadamard", n_act, measure(hadamard) @ np.linalg.pinv(hadamard)),
        ("zonal fast", fast.shape[1], zonal_fast(fast, measure(fast), rho)),
        (f"zonal fast x{args.repeats}", args.repeats * fast.shape[1], zonal_fast(fast, measure(fast, args.repeats), rho)),
    ]
    print(f"{n_act} actuators, {len(wfs)} slopes, noise {args.noise} of the largest IM entry")
    print(f"{'patterns':<16}{'frames':>7}{'IM error':>10}")
    for name, frames, d_hat in rows:
        print(f"{name:<16}{frames:>7}{error(d_hat):>10.3f}")
    return {name: error(d_hat) for name, _, d_hat in rows}


if __name__ == "__main__":
    main()
