"""DM surfaces: the separable fast path against the influence matrix, and misregistration.

Builds a Fried-geometry DM (a square actuator grid clipped to the pupil plus
half a pitch) with Gaussian influence functions on a pixel grid, checks that
the separable surface equals ``influence_matrix() @ cmd``, times both, and
reports how much a misregistered DM's surface differs from the calibrated one.

    python examples/dm_surface.py
    python examples/dm_surface.py --n-actuators 33 --n-pixels 144 --batch 10
"""

import argparse
import time

import numpy as np

import aobasis


def fried_dm(diameter, n_across, n_pixels, coupling):
    pitch = diameter / (n_across - 1)
    axis = (np.arange(n_across) - 0.5 * (n_across - 1)) * pitch
    xx, yy = np.meshgrid(axis, axis)
    keep = np.hypot(xx, yy) <= 0.5 * diameter + 0.5 * pitch * np.sqrt(2)
    actuators = np.column_stack((xx[keep], yy[keep]))
    pixels = (np.arange(n_pixels) - 0.5 * (n_pixels - 1)) * (diameter / n_pixels)
    return aobasis.DeformableMirror(actuators, pitch=pitch, coupling=coupling, grid=(pixels, pixels))


def _best(function, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diameter", type=float, default=8.0)
    parser.add_argument("--n-actuators", type=int, default=33, help="actuators across the diameter")
    parser.add_argument("--n-pixels", type=int, default=144)
    parser.add_argument("--coupling", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=1, help="commands per surface() call")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--shift", type=float, default=0.1, help="misregistration shift along x, in pitches")
    parser.add_argument("--rotation", type=float, default=1.0, help="misregistration rotation in degrees")
    args = parser.parse_args(argv)

    dm = fried_dm(args.diameter, args.n_actuators, args.n_pixels, args.coupling)
    rng = np.random.default_rng(0)
    cmd = rng.standard_normal((dm.n_actuators, args.batch)) if args.batch > 1 else rng.standard_normal(dm.n_actuators)

    build = _best(dm.influence_matrix, 1)
    fast = _best(lambda: dm.surface(cmd, method="separable"), args.repeats)
    slow = _best(lambda: dm.surface(cmd, method="matrix"), args.repeats)
    a, b = dm.surface(cmd, method="separable"), dm.surface(cmd, method="matrix")
    error = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    print(f"{dm.n_actuators} actuators on {dm.shape[0]}x{dm.shape[1]} pixels, {args.batch} command(s) per call")
    print(f"  separable surface:      {fast * 1e3:8.3f} ms")
    print(f"  influence matrix @ cmd: {slow * 1e3:8.3f} ms  ({slow / fast:.0f}x slower; building the matrix: {build:.2f} s)")
    print(f"  max relative difference: {error:.1e}")

    true = dm.transformed(shift=(args.shift * dm.pitch, 0.0), rotation_deg=args.rotation)
    pupil = np.hypot(*np.meshgrid(dm.grid[1], dm.grid[0])) <= 0.5 * args.diameter
    single = cmd if cmd.ndim == 1 else cmd[:, 0]
    diff = (true.surface(single) - dm.surface(single))[pupil]
    ref = dm.surface(single)[pupil]
    misregistration = float(np.std(diff) / np.std(ref))
    print(
        f"  misregistered by {args.shift:g} pitch and {args.rotation:g} deg "
        f"(separable: {true.separable}): surface error {misregistration:.1%} RMS"
    )
    return {"separable_s": fast, "matrix_s": slow, "max_rel_error": error, "misregistration_rms": misregistration}


if __name__ == "__main__":
    main()
