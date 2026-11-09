"""Volume-weighted CSD percentiles with logarithmic interpolation within size bins."""

from __future__ import annotations

import numpy as np

from .config import ModelConfig
from .grid import SizeGrid
from .simulation import SimulationResult


def volume_percentiles(
    counts_per_kg_water: np.ndarray,
    grid: SizeGrid,
) -> tuple[float, float, float, float]:
    """Return volume-based D10, D50, D90 and Span using the common bin convention."""
    counts = np.maximum(np.asarray(counts_per_kg_water, dtype=float), 0.0)
    volume_weights = counts * grid.volumes_m3
    total = float(np.sum(volume_weights))
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    cumulative = np.cumsum(volume_weights) / total
    diameters_um: list[float] = []
    for probability in (0.10, 0.50, 0.90):
        index = min(
            int(np.searchsorted(cumulative, probability, side="left")),
            counts.size - 1,
        )
        lower_probability = 0.0 if index == 0 else float(cumulative[index - 1])
        upper_probability = float(cumulative[index])
        fraction = (probability - lower_probability) / (
            upper_probability - lower_probability
        )
        lower_edge_m = float(grid.length_edges_m[index])
        upper_edge_m = float(grid.length_edges_m[index + 1])
        log_length_m = np.log(lower_edge_m) + fraction * (
            np.log(upper_edge_m) - np.log(lower_edge_m)
        )
        diameters_um.append(float(np.exp(log_length_m) * 1.0e6))

    d10, d50, d90 = diameters_um
    span = (d90 - d10) / d50 if d50 > 0.0 else 0.0
    return d10, d50, d90, span


def trajectory_metrics(
    result: SimulationResult,
    cfg: ModelConfig,
) -> dict[str, np.ndarray]:
    """Return volume percentiles, Span, yield and mass-balance error at each saved time."""
    del cfg
    percentiles = np.asarray(
        [volume_percentiles(counts, result.grid) for counts in result.counts_per_kg_water],
        dtype=float,
    )
    return {
        "D10_um": percentiles[:, 0],
        "D50_um": percentiles[:, 1],
        "D90_um": percentiles[:, 2],
        "Span": percentiles[:, 3],
        "yield_fraction": result.yield_fraction.copy(),
        "relative_mass_error": result.relative_mass_error.copy(),
    }
