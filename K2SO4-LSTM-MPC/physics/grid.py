"""Logarithmic size/volume grids, seed initialization and conservative redistribution. Bin states are absolute particle counts per kilogram of water."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .kinetics import solubility


@dataclass(frozen=True)
class SizeGrid:
    """Fixed size edges, pivots and particle volumes in SI units."""

    pivots_m: np.ndarray
    volumes_m3: np.ndarray
    length_edges_m: np.ndarray
    volume_edges_m3: np.ndarray


@dataclass(frozen=True)
class InitialInventory:
    """Initial dissolved/crystal inventory on the constant-water-mass basis."""

    solution_volume_l: float
    solution_mass_kg: float
    water_mass_kg: float
    dissolved_solute_mass_kg: float
    seed_mass_kg: float
    initial_concentration_kg_kg_water: float
    seed_counts_per_kg_water: np.ndarray


def make_log_grid(cfg: ModelConfig) -> SizeGrid:
    """Construct logarithmically spaced particle-size pivots and bin edges."""
    pivots = np.geomspace(
        cfg.grid.min_um * 1.0e-6,
        cfg.grid.max_um * 1.0e-6,
        cfg.grid.bins,
        dtype=float,
    )
    ratio = pivots[1] / pivots[0]
    length_edges = np.empty(pivots.size + 1, dtype=float)
    length_edges[1:-1] = np.sqrt(pivots[:-1] * pivots[1:])
    length_edges[0] = pivots[0] / np.sqrt(ratio)
    length_edges[-1] = pivots[-1] * np.sqrt(ratio)
    return SizeGrid(
        pivots_m=pivots,
        volumes_m3=pivots**3,
        length_edges_m=length_edges,
        volume_edges_m3=length_edges**3,
    )


def initial_inventory(grid: SizeGrid, cfg: ModelConfig) -> InitialInventory:
    """Build initial seed counts and dissolved/crystal inventory."""
    concentration = float(solubility(cfg.batch.start_temperature_c))
    solute_mass_fraction = concentration / (1.0 + concentration)
    total_volume_m3 = cfg.vessel.slurry_volume_l * 1.0e-3
    seed_fraction = cfg.seed.mass_fraction_of_dissolved
    # Include seed crystal volume when converting slurry volume to solution volume.
    volume_multiplier = (
        1.0
        + seed_fraction
        * solute_mass_fraction
        * cfg.vessel.solution_density_kg_m3
        / cfg.vessel.crystal_density_kg_m3
    )
    solution_volume_m3 = total_volume_m3 / volume_multiplier
    solution_mass_kg = cfg.vessel.solution_density_kg_m3 * solution_volume_m3
    water_mass_kg = solution_mass_kg / (1.0 + concentration)
    dissolved_mass_kg = concentration * water_mass_kg
    seed_mass_kg = seed_fraction * dissolved_mass_kg

    sizes_um = grid.pivots_m * 1.0e6
    support = (sizes_um >= cfg.seed.min_um) & (sizes_um <= cfg.seed.max_um)
    log_sigma = np.log(cfg.seed.sigma_g)
    # Evaluate log-space number weights, then normalize to the specified seed mass.
    number_weights = np.zeros_like(grid.pivots_m)
    number_weights[support] = np.exp(
        -0.5
        * (
            (np.log(sizes_um[support]) - np.log(cfg.seed.d50_um))
            / log_sigma
        )
        ** 2
    )
    mass_per_kg_water = seed_mass_kg / water_mass_kg
    represented_mass = (
        cfg.vessel.crystal_density_kg_m3
        * cfg.vessel.volume_shape_factor
        * np.dot(number_weights, grid.volumes_m3)
    )
    if represented_mass <= 0.0:
        raise ValueError("Seed distribution has zero represented crystal mass")
    seed_counts = number_weights * mass_per_kg_water / represented_mass

    return InitialInventory(
        solution_volume_l=solution_volume_m3 * 1.0e3,
        solution_mass_kg=solution_mass_kg,
        water_mass_kg=water_mass_kg,
        dissolved_solute_mass_kg=dissolved_mass_kg,
        seed_mass_kg=seed_mass_kg,
        initial_concentration_kg_kg_water=concentration,
        seed_counts_per_kg_water=seed_counts,
    )


def redistribute_by_volume(
    counts: np.ndarray,
    target_volumes_m3: np.ndarray,
    grid: SizeGrid,
    *,
    underflow: str = "discard",
) -> np.ndarray:
    """Redistribute absolute counts using particle-volume weights."""
    source_counts = np.asarray(counts, dtype=float)
    targets = np.asarray(target_volumes_m3, dtype=float)
    if source_counts.shape != targets.shape:
        raise ValueError("counts and target_volumes_m3 must have identical shapes")
    if np.any(source_counts < 0.0):
        raise ValueError("counts must be nonnegative")
    if underflow not in {"discard", "first_mass"}:
        raise ValueError("underflow must be 'discard' or 'first_mass'")

    output = np.zeros_like(grid.volumes_m3)
    active = (source_counts > 0.0) & np.isfinite(targets) & (targets > 0.0)
    if not np.any(active):
        return output

    active_counts = source_counts[active]
    active_targets = targets[active]
    first_volume = grid.volumes_m3[0]
    last_volume = grid.volumes_m3[-1]

    below = active_targets < first_volume
    if underflow == "first_mass" and np.any(below):
        output[0] += np.sum(active_counts[below] * active_targets[below] / first_volume)

    # Preserve overflow volume with an equivalent count in the last bin.
    above = active_targets > last_volume
    if np.any(above):
        output[-1] += np.sum(active_counts[above] * active_targets[above] / last_volume)

    inside = ~(below | above)
    inside_counts = active_counts[inside]
    inside_targets = active_targets[inside]
    if inside_targets.size == 0:
        return output

    upper = np.searchsorted(grid.volumes_m3, inside_targets, side="left")
    at_first = upper == 0
    if np.any(at_first):
        output[0] += np.sum(inside_counts[at_first])

    between = ~at_first
    if np.any(between):
        high = upper[between]
        low = high - 1
        target = inside_targets[between]
        count = inside_counts[between]
        low_volume = grid.volumes_m3[low]
        high_volume = grid.volumes_m3[high]
        # Interpolate in volume to preserve count and volume between adjacent pivots.
        high_weight = (target - low_volume) / (high_volume - low_volume)
        low_weight = 1.0 - high_weight
        np.add.at(output, low, count * low_weight)
        np.add.at(output, high, count * high_weight)
    return output
