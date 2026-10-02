"""Time basis generation for several DM sizes.

Each entry is the best of --repeats runs of ``generate(n_modes)`` on a square
grid clipped by a 10 m pupil (``n_modes`` is capped at what the basis allows).

    python examples/benchmark.py
    python examples/benchmark.py --grid-sizes 16 32 64 --n-modes 100 --markdown
    python examples/benchmark.py --n-modes all --gpu
"""

import argparse
import time
import warnings

import aobasis


def generators(positions, pitch, gpu):
    yield "KL (CPU)", lambda: aobasis.KLBasisGenerator(positions), {"ignore_piston": True}
    if gpu:
        yield "KL (GPU)", lambda: aobasis.KLBasisGenerator(positions, use_gpu=True), {"ignore_piston": True}
    yield "Zernike", lambda: aobasis.ZernikeBasisGenerator(positions, pupil_radius=5.0), {"ignore_piston": True}
    yield "Fourier", lambda: aobasis.FourierBasisGenerator(positions, pupil_diameter=10.0), {"ignore_piston": True}
    yield "Hadamard", lambda: aobasis.HadamardBasisGenerator(positions), {"construction": "smallest"}
    yield "Zonal", lambda: aobasis.ZonalBasisGenerator(positions), {}
    yield "Zonal fast", lambda: aobasis.ZonalFastBasisGenerator(positions, min_distance=3 * pitch), None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--n-modes", default="100", help="modes per basis, or 'all'")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", action="store_true", help="also time KL on the GPU (CuPy)")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args(argv)

    results = {}
    actuators = {}
    for size in args.grid_sizes:
        positions = aobasis.make_circular_actuator_grid(10.0, size)
        actuators[size] = len(positions)
        pitch = 10.0 / (size - 1)
        n = len(positions) - 1 if args.n_modes == "all" else min(int(args.n_modes), len(positions) - 1)
        for name, make, options in generators(positions, pitch, args.gpu):
            best = float("inf")
            for _ in range(args.repeats):
                gen = make()
                start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # rank warnings for aliased high orders
                    if options is None:
                        gen.generate()  # zonal fast: the full set of patterns
                    else:
                        gen.generate(n, **options)
                best = min(best, time.perf_counter() - start)
            results.setdefault(name, {})[size] = best

    columns = [f"{size}x{size} ({actuators[size]} acts)" for size in args.grid_sizes]
    label = "all modes" if args.n_modes == "all" else f"{args.n_modes} modes"
    if args.markdown:
        print(f"| Basis ({label}) | " + " | ".join(columns) + " |")
        print("|---|" + "---|" * len(columns))
        for name, row in results.items():
            print(f"| **{name}** | " + " | ".join(_format(row[s]) for s in args.grid_sizes) + " |")
    else:
        print(f"{'Basis (' + label + ')':<22}" + "".join(f"{c:>22}" for c in columns))
        for name, row in results.items():
            print(f"{name:<22}" + "".join(f"{_format(row[s]):>22}" for s in args.grid_sizes))
    return results


def _format(seconds):
    return "<0.001 s" if seconds < 1e-3 else f"{seconds:.3f} s"


if __name__ == "__main__":
    main()
