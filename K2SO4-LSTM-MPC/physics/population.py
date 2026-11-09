"""Agglomeration and breakage operators, conservative daughter redistribution and positivity-limited internal steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .grid import SizeGrid, redistribute_by_volume
from .kinetics import breakage_selection_rate


@dataclass(frozen=True)
class PopulationOperators:
    """Precomputed conservative agglomeration and breakage operators."""

    grid: SizeGrid
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_child_volumes_m3: np.ndarray
    pair_birth_low: np.ndarray
    pair_birth_high: np.ndarray
    pair_birth_low_weight: np.ndarray
    pair_birth_high_weight: np.ndarray
    breakage_selection_s: np.ndarray
    breakage_daughter_matrix: np.ndarray

    @classmethod
    def build(cls, grid: SizeGrid, cfg: ModelConfig) -> "PopulationOperators":
        """Precompute collision redistribution and the binary-breakage daughter matrix."""
        # Enumerate each unordered collision pair once, including equal-size pairs.
        pair_i, pair_j = np.triu_indices(grid.pivots_m.size)
        child_volumes = grid.volumes_m3[pair_i] + grid.volumes_m3[pair_j]
        pair_birth_low, pair_birth_high, low_weight, high_weight = _fixed_pivot_map(
            child_volumes, grid
        )
        selection = np.asarray(
            breakage_selection_rate(grid.volumes_m3, cfg), dtype=float
        )

        # Use midpoint quadrature over daughter volume fractions for binary breakage.
        
        quadrature_points = cfg.numerical.breakage_quadrature_points
        fractions = (np.arange(quadrature_points, dtype=float) + 0.5) / quadrature_points
        daughter_matrix = np.empty(
            (grid.pivots_m.size, grid.pivots_m.size), dtype=float
        )
        daughter_counts = np.full(quadrature_points, 2.0 / quadrature_points)
        for parent_index, parent_volume in enumerate(grid.volumes_m3):
            daughter_matrix[:, parent_index] = redistribute_by_volume(
                daughter_counts,
                fractions * parent_volume,
                grid,
                underflow="first_mass",
            )

        return cls(
            grid=grid,
            pair_i=pair_i,
            pair_j=pair_j,
            pair_child_volumes_m3=child_volumes,
            pair_birth_low=pair_birth_low,
            pair_birth_high=pair_birth_high,
            pair_birth_low_weight=low_weight,
            pair_birth_high_weight=high_weight,
            breakage_selection_s=selection,
            breakage_daughter_matrix=daughter_matrix,
        )


def _fixed_pivot_map(
    target_volumes_m3: np.ndarray,
    grid: SizeGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map target particle volumes to adjacent pivots with volume-conserving weights."""
    targets = np.asarray(target_volumes_m3, dtype=float)
    size = targets.size
    low = np.empty(size, dtype=int)
    high = np.empty(size, dtype=int)
    low_weight = np.zeros(size, dtype=float)
    high_weight = np.zeros(size, dtype=float)
    last = grid.volumes_m3.size - 1

    # Preserve overflow volume with an equivalent count at the last pivot.
    above = targets > grid.volumes_m3[-1]
    low[above] = last
    high[above] = last
    low_weight[above] = targets[above] / grid.volumes_m3[-1]

    inside = ~above
    upper = np.searchsorted(grid.volumes_m3, targets[inside], side="left")
    upper = np.maximum(upper, 1)
    lower = upper - 1
    inside_targets = targets[inside]
    lower_volume = grid.volumes_m3[lower]
    upper_volume = grid.volumes_m3[upper]
    upper_weight = (inside_targets - lower_volume) / (upper_volume - lower_volume)
    low[inside] = lower
    high[inside] = upper
    low_weight[inside] = 1.0 - upper_weight
    high_weight[inside] = upper_weight
    return low, high, low_weight, high_weight


def agglomeration_rate(
    counts_per_kg_water: np.ndarray,
    operators: PopulationOperators,
    cfg: ModelConfig,
) -> np.ndarray:
    """Return net count rates from constant-kernel agglomeration."""
    counts = np.asarray(counts_per_kg_water, dtype=float)
    if counts.shape != operators.grid.pivots_m.shape:
        raise ValueError("counts have the wrong shape")
    if np.any(counts < 0.0):
        raise ValueError("counts must be nonnegative")

    # Equal-size collisions have a factor of one half to avoid double counting.
    event_rates = (
        cfg.kinetics.agglomeration_kernel
        * counts[operators.pair_i]
        * counts[operators.pair_j]
    )
    diagonal = operators.pair_i == operators.pair_j
    event_rates[diagonal] *= 0.5

    losses = (
        cfg.kinetics.agglomeration_kernel * counts * float(np.sum(counts))
    )
    births = np.bincount(
        operators.pair_birth_low,
        weights=event_rates * operators.pair_birth_low_weight,
        minlength=counts.size,
    ) + np.bincount(
        operators.pair_birth_high,
        weights=event_rates * operators.pair_birth_high_weight,
        minlength=counts.size,
    )
    return births - losses


def breakage_rate(
    counts_per_kg_water: np.ndarray,
    operators: PopulationOperators,
    cfg: ModelConfig,
) -> np.ndarray:
    """Return daughter births minus parent deaths from breakage."""
    del cfg
    counts = np.asarray(counts_per_kg_water, dtype=float)
    if counts.shape != operators.grid.pivots_m.shape:
        raise ValueError("counts have the wrong shape")
    if np.any(counts < 0.0):
        raise ValueError("counts must be nonnegative")

    parent_event_rates = operators.breakage_selection_s * counts
    births = operators.breakage_daughter_matrix @ parent_event_rates
    return births - parent_event_rates


def death_limited_step_s(
    counts_per_kg_water: np.ndarray,
    operators: PopulationOperators,
    cfg: ModelConfig,
) -> float:
    """Bound the explicit substep by the maximum fractional particle-death rate."""
    counts = np.asarray(counts_per_kg_water, dtype=float)
    total_count = max(float(np.sum(counts)), 0.0)
    agglomeration_frequency = cfg.kinetics.agglomeration_kernel * total_count
    breakage_frequency = float(np.max(operators.breakage_selection_s))
    maximum_death_frequency = agglomeration_frequency + breakage_frequency
    if maximum_death_frequency <= 0.0:
        return cfg.numerical.max_internal_step_s
    return min(
        cfg.numerical.max_internal_step_s,
        cfg.numerical.max_fractional_death / maximum_death_frequency,
    )
