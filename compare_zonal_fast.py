import argparse
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from aobasis import ZonalFastBasisGenerator


def make_integer_circular_grid(grid_size: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build a unit-pitch square lattice clipped by a circular aperture."""
    axis = np.arange(grid_size, dtype=float) - 0.5 * (grid_size - 1)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    full_positions = np.column_stack((xx.ravel(), yy.ravel()))
    full_indices = np.column_stack(np.unravel_index(np.arange(grid_size * grid_size), (grid_size, grid_size)))

    mask = np.sum(full_positions**2, axis=1) <= radius**2 + 1e-12
    return full_positions[mask], full_indices[mask]


def build_corner_subgrid_basis(grid_indices: np.ndarray, spacing: int) -> np.ndarray:
    """Group actuators by their row and column residue classes modulo spacing."""
    if spacing <= 0:
        raise ValueError("spacing must be a positive integer.")
    if grid_indices.ndim != 2 or grid_indices.shape[1] != 2:
        raise ValueError("grid_indices must have shape (n_actuators, 2).")

    residues = np.mod(grid_indices, spacing)
    groups = {}
    for actuator_index, residue in enumerate(residues):
        key = (int(residue[0]), int(residue[1]))
        groups.setdefault(key, []).append(actuator_index)

    ordered_groups = [groups[key] for key in sorted(groups)]
    basis = np.zeros((grid_indices.shape[0], len(ordered_groups)), dtype=float)
    for mode_index, actuator_group in enumerate(ordered_groups):
        basis[actuator_group, mode_index] = 1.0
    return basis


def minimum_pairwise_distance(positions: np.ndarray, active_indices: np.ndarray) -> float:
    if active_indices.size < 2:
        return np.inf

    active_positions = positions[active_indices]
    deltas = active_positions[:, None, :] - active_positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    upper_triangle = distances[np.triu_indices(active_positions.shape[0], k=1)]
    return float(upper_triangle.min())


def validate_basis(positions: np.ndarray, basis: np.ndarray, min_distance: float) -> Tuple[bool, float]:
    if basis.shape[0] != positions.shape[0]:
        raise ValueError("basis row count must match the number of actuator positions.")

    covered = np.allclose(basis.sum(axis=1), 1.0)
    mode_min_distances: List[float] = []

    for mode_index in range(basis.shape[1]):
        active_indices = np.flatnonzero(basis[:, mode_index] > 0.5)
        mode_min_distances.append(minimum_pairwise_distance(positions, active_indices))

    worst_case_distance = min(mode_min_distances) if mode_min_distances else np.inf
    spacing_ok = worst_case_distance >= min_distance - 1e-12
    return covered and spacing_ok, worst_case_distance


def compare_for_spacing(positions: np.ndarray, grid_indices: np.ndarray, spacing: int) -> dict:
    naive_basis = build_corner_subgrid_basis(grid_indices, spacing)
    fast_basis = ZonalFastBasisGenerator(positions, min_distance=float(spacing)).generate()

    naive_valid, naive_min_distance = validate_basis(positions, naive_basis, float(spacing))
    fast_valid, fast_min_distance = validate_basis(positions, fast_basis, float(spacing))

    naive_modes = int(naive_basis.shape[1])
    fast_modes = int(fast_basis.shape[1])
    reduction = naive_modes - fast_modes
    reduction_fraction = reduction / naive_modes if naive_modes else 0.0

    return {
        "spacing": spacing,
        "n_actuators": int(positions.shape[0]),
        "naive_modes": naive_modes,
        "fast_modes": fast_modes,
        "reduction": reduction,
        "reduction_fraction": reduction_fraction,
        "naive_valid": naive_valid,
        "fast_valid": fast_valid,
        "naive_min_distance": naive_min_distance,
        "fast_min_distance": fast_min_distance,
    }


def parse_distances(values: Sequence[int]) -> List[int]:
    unique_values = sorted({int(value) for value in values})
    if not unique_values:
        raise ValueError("At least one spacing value must be provided.")
    if any(value <= 0 for value in unique_values):
        raise ValueError("Spacing values must all be positive integers.")
    return unique_values


def print_report(results: Iterable[dict]) -> None:
    header = (
        f"{'D':>4} {'Actuators':>10} {'Naive':>8} {'Fast':>8} {'Saved':>8} {'Saved %':>9} "
        f"{'Naive OK':>9} {'Fast OK':>8} {'Naive min d':>12} {'Fast min d':>11}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['spacing']:>4d} "
            f"{result['n_actuators']:>10d} "
            f"{result['naive_modes']:>8d} "
            f"{result['fast_modes']:>8d} "
            f"{result['reduction']:>8d} "
            f"{100.0 * result['reduction_fraction']:>8.2f}% "
            f"{str(result['naive_valid']):>9} "
            f"{str(result['fast_valid']):>8} "
            f"{result['naive_min_distance']:>12.3f} "
            f"{result['fast_min_distance']:>11.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a naive corner-anchored D x D subgrid zonal-fast basis against "
            "the graph-coloring zonal-fast basis on a circular aperture."
        )
    )
    parser.add_argument("--grid-size", type=int, default=60, help="Number of points along each side of the square lattice.")
    parser.add_argument("--radius", type=float, default=30.0, help="Circular aperture radius in lattice-pitch units.")
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10],
        help="Integer spacing values D to test, in lattice-pitch units.",
    )
    parser.add_argument(
        "--fail-if-not-better",
        action="store_true",
        help="Exit with status 1 if the zonal-fast basis is not strictly better for every tested spacing.",
    )
    args = parser.parse_args()

    distances = parse_distances(args.distances)
    positions, grid_indices = make_integer_circular_grid(args.grid_size, args.radius)
    results = [compare_for_spacing(positions, grid_indices, spacing) for spacing in distances]

    print(
        f"Circular aperture on a {args.grid_size}x{args.grid_size} unit-pitch lattice, "
        f"radius={args.radius:.1f}, actuators inside pupil={positions.shape[0]}"
    )
    print_report(results)

    wins = [result for result in results if result["fast_modes"] < result["naive_modes"]]
    ties = [result for result in results if result["fast_modes"] == result["naive_modes"]]
    losses = [result for result in results if result["fast_modes"] > result["naive_modes"]]

    print()
    print(f"Fast basis uses fewer modes for {len(wins)} / {len(results)} tested spacings.")
    if ties:
        tied_spacings = ", ".join(str(result["spacing"]) for result in ties)
        print(f"Tied spacings: {tied_spacings}")
    if losses:
        loss_spacings = ", ".join(str(result["spacing"]) for result in losses)
        print(f"Fast basis used more modes at: {loss_spacings}")

    if not losses and not ties:
        print("Verdict: zonal-fast is strictly better than the naive corner-subgrid basis for every tested spacing.")
    elif losses:
        print("Verdict: this zonal-fast implementation does not beat the naive corner-subgrid baseline on this geometry.")
        print("The graph coloring is producing a valid basis, but not a minimal one for these spacings.")
    else:
        print("Verdict: zonal-fast matches the naive construction for some spacings and never improves on it in this run.")

    if args.fail_if_not_better and (losses or ties):
        raise SystemExit(1)


if __name__ == "__main__":
    main()