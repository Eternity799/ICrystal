"""Typed configuration for the vessel, seeds, kinetics, grid, batch and PBM integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VesselConfig:
    """Slurry volume, phase densities and crystal volume shape factor."""
    slurry_volume_l: float
    solution_density_kg_m3: float
    crystal_density_kg_m3: float
    volume_shape_factor: float


@dataclass(frozen=True)
class SeedConfig:
    """Seed loading and log-space size-distribution parameters."""
    mass_fraction_of_dissolved: float
    d50_um: float
    sigma_g: float
    min_um: float
    max_um: float


@dataclass(frozen=True)
class GridConfig:
    """Particle-size pivot limits, bin count and nucleus size."""
    bins: int
    min_um: float
    max_um: float
    nucleus_size_um: float


@dataclass(frozen=True)
class KineticsConfig:
    """Nucleation, growth, dissolution, agglomeration and breakage coefficients."""
    kn1: float
    a1: float
    kn2: float
    a2: float
    kg_m_s: float
    g: float
    breakage_a: float
    breakage_m: float
    agglomeration_kernel: float
    kd_m_s: float
    dissolution_order: float


@dataclass(frozen=True)
class BatchConfig:
    """Temperatures, cooling rate, batch duration and sampling interval."""
    start_temperature_c: float
    end_temperature_c: float
    nominal_cooling_rate_c_min: float
    hold_minutes: float
    total_minutes: float
    sample_seconds: float


@dataclass(frozen=True)
class DatasetConfig:
    """Sampling limits for temperature paths and initial seed conditions."""
    debug_trajectories: int
    formal_trajectories: int
    random_seed: int
    min_temperature_c: float
    max_temperature_c: float
    min_rate_c_min: float
    max_rate_c_min: float
    supersaturation_floor: float
    seed_d50_min_um: float
    seed_d50_max_um: float
    seed_sigma_g_min: float
    seed_sigma_g_max: float
    seed_mass_fraction_min: float
    seed_mass_fraction_max: float


@dataclass(frozen=True)
class NumericalConfig:
    """PBM substep limits, fractional death bound and breakage quadrature size."""
    max_internal_step_s: float
    mechanism_coupling_substep_s: float
    max_fractional_death: float
    mass_tolerance_relative: float
    breakage_quadrature_points: int


@dataclass(frozen=True)
class ModelConfig:
    """Physical model sections and their provenance."""
    provenance: dict[str, str]
    vessel: VesselConfig
    seed: SeedConfig
    grid: GridConfig
    kinetics: KineticsConfig
    batch: BatchConfig
    dataset: DatasetConfig
    numerical: NumericalConfig

    def validate(self) -> None:
        """Check supported physical bounds and PBM integration settings."""
        if self.vessel.slurry_volume_l <= 0.0:
            raise ValueError("vessel.slurry_volume_l must be positive")
        if self.vessel.solution_density_kg_m3 <= 0.0:
            raise ValueError("vessel.solution_density_kg_m3 must be positive")
        if self.vessel.crystal_density_kg_m3 <= 0.0:
            raise ValueError("vessel.crystal_density_kg_m3 must be positive")
        if self.grid.bins < 2:
            raise ValueError("grid.bins must be at least 2")
        if self.grid.min_um <= 0.0 or self.grid.min_um >= self.grid.max_um:
            raise ValueError("grid.min_um must be positive and smaller than grid.max_um")
        if not self.grid.min_um <= self.grid.nucleus_size_um <= self.grid.max_um:
            raise ValueError("grid.nucleus_size_um must lie inside the size grid")
        if not self.grid.min_um <= self.seed.min_um < self.seed.max_um <= self.grid.max_um:
            raise ValueError("seed support must lie inside the size grid")
        if self.seed.sigma_g <= 1.0:
            raise ValueError("seed.sigma_g must be greater than 1")
        if self.batch.sample_seconds <= 0.0 or self.batch.total_minutes <= 0.0:
            raise ValueError("batch timing values must be positive")
        if self.numerical.max_internal_step_s <= 0.0:
            raise ValueError("numerical.max_internal_step_s must be positive")
        if self.numerical.mechanism_coupling_substep_s <= 0.0:
            raise ValueError(
                "numerical.mechanism_coupling_substep_s must be positive"
            )
        if (
            self.numerical.mechanism_coupling_substep_s
            > self.numerical.max_internal_step_s
        ):
            raise ValueError(
                "numerical.mechanism_coupling_substep_s cannot exceed "
                "max_internal_step_s"
            )
        if not 0.0 < self.numerical.max_fractional_death < 1.0:
            raise ValueError("numerical.max_fractional_death must lie between 0 and 1")
        if self.dataset.supersaturation_floor <= 0.0:
            raise ValueError("dataset.supersaturation_floor must be positive")
        if not 0.0 < self.dataset.seed_d50_min_um < self.dataset.seed_d50_max_um:
            raise ValueError("dataset seed D50 bounds are invalid")
        if not 1.0 < self.dataset.seed_sigma_g_min < self.dataset.seed_sigma_g_max:
            raise ValueError("dataset seed sigma_g bounds are invalid")
        if not 0.0 < self.dataset.seed_mass_fraction_min < self.dataset.seed_mass_fraction_max:
            raise ValueError("dataset seed mass-fraction bounds are invalid")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required configuration mapping."""
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid configuration section: {name}")
    return value


def load_config(path: str | Path) -> ModelConfig:
    """Load YAML sections into ModelConfig and validate their bounds."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    cfg = ModelConfig(
        provenance={str(k): str(v) for k, v in _section(raw, "provenance").items()},
        vessel=VesselConfig(**_section(raw, "vessel")),
        seed=SeedConfig(**_section(raw, "seed")),
        grid=GridConfig(**_section(raw, "grid")),
        kinetics=KineticsConfig(**_section(raw, "kinetics")),
        batch=BatchConfig(**_section(raw, "batch")),
        dataset=DatasetConfig(**_section(raw, "dataset")),
        numerical=NumericalConfig(**_section(raw, "numerical")),
    )
    cfg.validate()
    return cfg
