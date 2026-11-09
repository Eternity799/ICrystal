"""Nominal K2SO4 solubility, supersaturation, growth/dissolution, and primary/secondary nucleation correlations with explicit units."""

from __future__ import annotations

import numpy as np

from .config import ModelConfig


def solubility(temperature_c: float | np.ndarray) -> float | np.ndarray:
    """Equilibrium concentration in kilograms of solute per kilogram of water."""
    temperature = np.asarray(temperature_c, dtype=float)
    value = 6.29e-2 + 2.46e-3 * temperature - 7.14e-6 * temperature**2
    return float(value) if value.ndim == 0 else value


def supersaturation_ratio(
    concentration_kg_kg_water: float | np.ndarray,
    temperature_c: float | np.ndarray,
) -> float | np.ndarray:
    """Return concentration divided by equilibrium solubility at the given temperature."""
    value = np.asarray(concentration_kg_kg_water, dtype=float) / np.asarray(
        solubility(temperature_c), dtype=float
    )
    return float(value) if value.ndim == 0 else value


def primary_nucleation(delta_c: float, cfg: ModelConfig) -> float:
    """Return the primary nucleation rate, with zero nucleation below saturation."""
    drive = max(float(delta_c), 0.0)
    if drive == 0.0:
        return 0.0
    return cfg.kinetics.kn1 * drive**cfg.kinetics.a1


def secondary_nucleation(
    delta_c: float,
    magma_density_kg_kg_water: float,
    cfg: ModelConfig,
) -> float:
    """Return the secondary nucleation rate from supersaturation and crystal mass loading."""
    drive = max(float(delta_c), 0.0)
    magma_density = max(float(magma_density_kg_kg_water), 0.0)
    if drive == 0.0 or magma_density == 0.0:
        return 0.0
    return cfg.kinetics.kn2 * drive**cfg.kinetics.a2 * magma_density


def linear_growth_rate(delta_c: float, cfg: ModelConfig) -> float:
    """Return the linear size-change rate in metres per second; dissolution is negative."""
    drive = float(delta_c)
    if drive >= 0.0:
        return cfg.kinetics.kg_m_s * drive**cfg.kinetics.g
    return -cfg.kinetics.kd_m_s * (-drive) ** cfg.kinetics.dissolution_order


def breakage_selection_rate(
    particle_volume_m3: float | np.ndarray,
    cfg: ModelConfig,
) -> float | np.ndarray:
    """Return the size-dependent breakage frequency in inverse seconds."""
    volume = np.maximum(np.asarray(particle_volume_m3, dtype=float), 0.0)
    value = cfg.kinetics.breakage_a * volume**cfg.kinetics.breakage_m
    return float(value) if value.ndim == 0 else value
