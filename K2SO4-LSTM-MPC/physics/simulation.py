"""Full discretized-CSD batch PBM with Strang splitting. Agglomeration and breakage half-steps surround a nucleation-growth step. Dissolved concentration follows the fixed total-solute inventory; temperature is supplied externally."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .grid import InitialInventory, SizeGrid, initial_inventory, make_log_grid, redistribute_by_volume
from .kinetics import (
    linear_growth_rate,
    primary_nucleation,
    secondary_nucleation,
    solubility,
    supersaturation_ratio,
)
from .population import (
    PopulationOperators,
    agglomeration_rate,
    breakage_rate,
    death_limited_step_s,
)
from .profiles import TemperatureProfile


@dataclass(frozen=True)
class SimulationResult:
    """Saved batch times, full CSD, temperatures, concentrations, quality and mass-balance diagnostics."""

    time_s: np.ndarray
    temperature_c: np.ndarray
    counts_per_kg_water: np.ndarray
    concentration_kg_kg_water: np.ndarray
    crystal_mass_kg_kg_water: np.ndarray
    supersaturation_ratio: np.ndarray
    yield_fraction: np.ndarray
    relative_mass_error: np.ndarray
    grid: SizeGrid
    inventory: InitialInventory


@dataclass(frozen=True)
class PlantSample:
    """One physical plant observation at the current physical time."""

    time_s: float
    temperature_c: float
    counts_per_kg_water: np.ndarray
    concentration_kg_kg_water: float
    crystal_mass_kg_kg_water: float
    supersaturation_ratio: float
    yield_fraction: float
    relative_mass_error: float


def _crystal_mass_kg_kg_water(
    counts: np.ndarray,
    grid: SizeGrid,
    cfg: ModelConfig,
) -> float:
    """Return crystal mass in kilograms per kilogram of water."""
    return float(
        cfg.vessel.crystal_density_kg_m3
        * cfg.vessel.volume_shape_factor
        * np.dot(counts, grid.volumes_m3)
    )


def _nucleation_growth_step(
    counts: np.ndarray,
    concentration: float,
    temperature_c: float,
    duration_s: float,
    grid: SizeGrid,
    cfg: ModelConfig,
) -> np.ndarray:
    """Translate particle sizes by growth or dissolution, then add newly nucleated particles."""
    equilibrium = float(solubility(temperature_c))
    delta_c = concentration - equilibrium
    growth_rate = linear_growth_rate(delta_c, cfg)

    # Remove particles whose translated size falls below the grid.
    target_lengths = np.maximum(grid.pivots_m + growth_rate * duration_s, 0.0)
    transported = redistribute_by_volume(
        counts,
        target_lengths**3,
        grid,
        underflow="discard",
    )

    magma_density = _crystal_mass_kg_kg_water(counts, grid, cfg)
    nucleation_rate = primary_nucleation(delta_c, cfg) + secondary_nucleation(
        delta_c, magma_density, cfg
    )
    if nucleation_rate > 0.0:
        nuclei_count = nucleation_rate * duration_s
        nucleus_length_m = cfg.grid.nucleus_size_um * 1.0e-6
        transported += redistribute_by_volume(
            np.asarray([nuclei_count]),
            np.asarray([nucleus_length_m**3]),
            grid,
            underflow="first_mass",
        )
    return transported


def _mechanism_coupled_interval(
    counts: np.ndarray,
    total_solute: float,
    start_time_s: float,
    duration_s: float,
    profile: TemperatureProfile,
    grid: SizeGrid,
    operators: PopulationOperators,
    cfg: ModelConfig,
) -> np.ndarray:
    """Advance the coupled mechanisms through bounded Strang-splitting intervals."""
    updated = counts.copy()
    elapsed_s = 0.0
    while elapsed_s < duration_s - 1.0e-12:
        substep_s = min(
            cfg.numerical.mechanism_coupling_substep_s,
            duration_s - elapsed_s,
        )
        updated = _strang_coupled_step(
            updated,
            total_solute,
            start_time_s + elapsed_s,
            substep_s,
            profile,
            grid,
            operators,
            cfg,
        )
        if not np.isfinite(updated).all() or np.min(updated) < -1.0e-10:
            raise RuntimeError("Strang coupling produced an invalid CSD")
        if (
            _crystal_mass_kg_kg_water(updated, grid, cfg)
            > total_solute * (1.0 + 1.0e-12)
        ):
            raise RuntimeError(
                "Crystal mass exhausted liquid solute during Strang interval"
            )
        elapsed_s += substep_s
    return updated


def _population_step(
    counts: np.ndarray,
    duration_s: float,
    operators: PopulationOperators,
    cfg: ModelConfig,
) -> np.ndarray:
    """Advance agglomeration and breakage with death-limited explicit Euler substeps."""
    updated = counts.copy()
    remaining = float(duration_s)
    while remaining > 1e-12:
        substep = min(remaining, death_limited_step_s(updated, operators, cfg))
        derivative = agglomeration_rate(updated, operators, cfg) + breakage_rate(
            updated, operators, cfg
        )
        trial = updated + substep * derivative
        if np.min(trial) < -1e-7:
            substep *= 0.5
            if substep < 1e-9:
                raise RuntimeError("Population substep could not maintain nonnegative counts")
            continue
        updated = np.maximum(trial, 0.0)
        remaining -= substep
    return updated


def _strang_coupled_step(
    counts: np.ndarray,
    total_solute: float,
    start_time_s: float,
    duration_s: float,
    profile: TemperatureProfile,
    grid: SizeGrid,
    operators: PopulationOperators,
    cfg: ModelConfig,
) -> np.ndarray:
    """Apply population half-steps around a full nucleation-growth step at the midpoint temperature."""
    half_advanced = _population_step(counts, 0.5 * duration_s, operators, cfg)
    crystal_mass = _crystal_mass_kg_kg_water(half_advanced, grid, cfg)
    concentration = total_solute - crystal_mass
    if concentration < -1.0e-12:
        raise RuntimeError("Crystal mass exhausted liquid solute during Strang step")
    midpoint_time_s = start_time_s + 0.5 * duration_s
    ngd_advanced = _nucleation_growth_step(
        half_advanced,
        max(concentration, 0.0),
        profile.temperature_at(midpoint_time_s),
        duration_s,
        grid,
        cfg,
    )
    return _population_step(ngd_advanced, 0.5 * duration_s, operators, cfg)


class BatchPlant:
    """Mutable full-PBM plant, advanced by one configured sampling interval."""

    def __init__(
        self,
        cfg: ModelConfig,
        grid: SizeGrid,
        inventory: InitialInventory,
        operators: PopulationOperators,
    ) -> None:
        self.cfg = cfg
        self.grid = grid
        self.inventory = inventory
        self.operators = operators
        self._counts = inventory.seed_counts_per_kg_water.copy()
        self._initial_seed_magma = _crystal_mass_kg_kg_water(
            self._counts, self.grid, self.cfg
        )
        # Total solute includes dissolved solute and seed crystals, per kilogram of water.
        self._total_solute = (
            inventory.initial_concentration_kg_kg_water
            + self._initial_seed_magma
        )
        self._current_time_s = 0.0
        self._temperature_c = float(cfg.batch.start_temperature_c)
        self._total_time_s = float(cfg.batch.total_minutes * 60.0)
        self._samples: list[PlantSample] = [self._make_sample()]

    @classmethod
    def initialize(cls, cfg: ModelConfig) -> "BatchPlant":
        """Initialize the full population and the fixed global solute inventory."""
        grid = make_log_grid(cfg)
        inventory = initial_inventory(grid, cfg)
        operators = PopulationOperators.build(grid, cfg)
        return cls(cfg, grid, inventory, operators)

    @property
    def current(self) -> PlantSample:
        """Return the most recently saved plant sample."""
        return self._samples[-1]

    def _make_sample(self) -> PlantSample:
        crystal_mass = _crystal_mass_kg_kg_water(
            self._counts, self.grid, self.cfg
        )
        concentration = self._total_solute - crystal_mass
        if concentration < -1.0e-12:
            raise RuntimeError(
                f"Negative liquid concentration near t={self._current_time_s:.3f} s"
            )
        concentration = max(concentration, 0.0)
        counts = self._counts.copy()
        counts.setflags(write=False)
        saturation = float(
            supersaturation_ratio(concentration, self._temperature_c)
        )
        yield_fraction = (
            crystal_mass - self._initial_seed_magma
        ) / max(
            self.inventory.initial_concentration_kg_kg_water,
            np.finfo(float).eps,
        )
        relative_mass_error = (
            concentration + crystal_mass - self._total_solute
        ) / self._total_solute
        return PlantSample(
            time_s=float(self._current_time_s),
            temperature_c=float(self._temperature_c),
            counts_per_kg_water=counts,
            concentration_kg_kg_water=float(concentration),
            crystal_mass_kg_kg_water=float(crystal_mass),
            supersaturation_ratio=saturation,
            yield_fraction=float(yield_fraction),
            relative_mass_error=float(relative_mass_error),
        )

    def advance(self, next_temperature_c: float) -> PlantSample:
        """Advance the physical PBM over one sampling interval toward the next actual temperature endpoint."""
        next_temperature = float(next_temperature_c)
        if not np.isfinite(next_temperature):
            raise ValueError("next temperature must be finite")
        if self._current_time_s >= self._total_time_s - 1.0e-12:
            raise RuntimeError("cannot advance beyond the batch endpoint")

        target_time_s = min(
            self._current_time_s + self.cfg.batch.sample_seconds,
            self._total_time_s,
        )
        interval_profile = TemperatureProfile(
            time_s=np.asarray([self._current_time_s, target_time_s]),
            temperature_c=np.asarray([self._temperature_c, next_temperature]),
            kind="closed_loop_interval",
        )
        current_time_s = self._current_time_s
        counts = self._counts
        while current_time_s < target_time_s - 1.0e-12:
            duration_s = min(
                self.cfg.numerical.max_internal_step_s,
                target_time_s - current_time_s,
            )
            candidate = _mechanism_coupled_interval(
                counts,
                self._total_solute,
                current_time_s,
                duration_s,
                interval_profile,
                self.grid,
                self.operators,
                self.cfg,
            )
            candidate_mass = _crystal_mass_kg_kg_water(
                candidate, self.grid, self.cfg
            )
            if candidate_mass > self._total_solute * (1.0 + 1.0e-12):
                raise RuntimeError(
                    "Crystal mass exhausted liquid solute near "
                    f"t={current_time_s:.3f} s"
                )
            counts = candidate
            current_time_s += duration_s

        self._counts = counts
        self._current_time_s = target_time_s
        self._temperature_c = next_temperature
        sample = self._make_sample()
        self._samples.append(sample)
        return sample

    def result(self) -> SimulationResult:
        """Return the saved physical trajectory without reconstructing states from quality metrics."""
        samples = tuple(self._samples)
        return SimulationResult(
            time_s=np.asarray([sample.time_s for sample in samples], dtype=float),
            temperature_c=np.asarray(
                [sample.temperature_c for sample in samples], dtype=float
            ),
            counts_per_kg_water=np.stack(
                [sample.counts_per_kg_water for sample in samples]
            ),
            concentration_kg_kg_water=np.asarray(
                [sample.concentration_kg_kg_water for sample in samples],
                dtype=float,
            ),
            crystal_mass_kg_kg_water=np.asarray(
                [sample.crystal_mass_kg_kg_water for sample in samples],
                dtype=float,
            ),
            supersaturation_ratio=np.asarray(
                [sample.supersaturation_ratio for sample in samples],
                dtype=float,
            ),
            yield_fraction=np.asarray(
                [sample.yield_fraction for sample in samples], dtype=float
            ),
            relative_mass_error=np.asarray(
                [sample.relative_mass_error for sample in samples], dtype=float
            ),
            grid=self.grid,
            inventory=self.inventory,
        )


def simulate(profile: TemperatureProfile, cfg: ModelConfig) -> SimulationResult:
    """Simulate a complete batch on the sampling grid."""
    expected_count = (
        int(
            round(
                cfg.batch.total_minutes
                * 60.0
                / cfg.batch.sample_seconds
            )
        )
        + 1
    )
    if profile.time_s.size != expected_count:
        raise ValueError("profile sample count does not match batch configuration")
    expected_times = np.linspace(
        0.0, cfg.batch.total_minutes * 60.0, expected_count
    )
    if not np.allclose(profile.time_s, expected_times, rtol=0.0, atol=1.0e-12):
        raise ValueError("profile times do not match batch sampling grid")

    plant = BatchPlant.initialize(cfg)
    if not np.isclose(
        profile.temperature_c[0], plant.current.temperature_c, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "profile initial temperature does not match batch configuration"
        )
    for next_temperature in profile.temperature_c[1:]:
        plant.advance(float(next_temperature))
    return plant.result()
