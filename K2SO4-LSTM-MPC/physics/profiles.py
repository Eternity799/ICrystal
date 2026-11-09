"""Piecewise-linear temperature paths evaluated at physical time. These interpolate temperature, not the size-distribution transport operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


PROFILE_KINDS = ("linear", "piecewise", "staircase", "smooth_random")


@dataclass(frozen=True)
class TemperatureProfile:
    """Piecewise-linear physical temperature profile."""

    time_s: np.ndarray
    temperature_c: np.ndarray
    kind: str

    def __post_init__(self) -> None:
        if self.time_s.ndim != 1 or self.temperature_c.shape != self.time_s.shape:
            raise ValueError("temperature profile arrays must be aligned one-dimensional arrays")
        if self.time_s.size < 2 or np.any(np.diff(self.time_s) <= 0.0):
            raise ValueError("temperature profile times must be strictly increasing")
        if not np.isfinite(self.temperature_c).all():
            raise ValueError("temperature profile must be finite")

    def temperature_at(self, time_s: float) -> float:
        """Interpolate the physical temperature at a time given in seconds."""
        return float(np.interp(float(time_s), self.time_s, self.temperature_c))


def _sample_times(cfg: ModelConfig) -> np.ndarray:
    """Return the sampling times, including both batch endpoints."""
    total_s = cfg.batch.total_minutes * 60.0
    count = int(round(total_s / cfg.batch.sample_seconds)) + 1
    return np.linspace(0.0, total_s, count)


def nominal_profile(cfg: ModelConfig) -> TemperatureProfile:
    """Cool at the nominal rate, then hold at the final temperature."""
    time_s = _sample_times(cfg)
    cooling_rate_c_s = cfg.batch.nominal_cooling_rate_c_min / 60.0
    temperature = np.maximum(
        cfg.batch.end_temperature_c,
        cfg.batch.start_temperature_c - cooling_rate_c_s * time_s,
    )
    return TemperatureProfile(time_s=time_s, temperature_c=temperature, kind="nominal")


def slow_then_fast_profile(cfg: ModelConfig) -> TemperatureProfile:
    """Construct the fixed slow-fast schedule, clipped at the final temperature."""
    time_s = _sample_times(cfg)
    time_min = time_s / 60.0
    temperature = np.where(
        time_min <= 80.0,
        cfg.batch.start_temperature_c - 0.15 * time_min,
        48.0 - 0.35 * (time_min - 80.0),
    )
    temperature = np.maximum(cfg.batch.end_temperature_c, temperature)
    return TemperatureProfile(
        time_s=time_s,
        temperature_c=temperature,
        kind="slow_then_fast",
    )


def _temperatures_from_minute_rates(rates_c_min: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    """Integrate interval rates from the starting temperature and set the final endpoint."""
    dt_min = cfg.batch.sample_seconds / 60.0
    temperatures = np.concatenate(
        [
            np.asarray([cfg.batch.start_temperature_c]),
            cfg.batch.start_temperature_c + np.cumsum(rates_c_min * dt_min),
        ]
    )
    temperatures[-1] = cfg.batch.end_temperature_c
    return temperatures


def _linear_random_rates(rng: np.random.Generator, cfg: ModelConfig, intervals: int) -> np.ndarray:
    """Sample a constant cooling rate with a random arrival time and a final hold."""
    duration_min = rng.uniform(100.0, cfg.batch.total_minutes)
    sample_minutes = np.arange(intervals + 1) * cfg.batch.sample_seconds / 60.0
    rate = (
        cfg.batch.end_temperature_c - cfg.batch.start_temperature_c
    ) / duration_min
    temperature = cfg.batch.start_temperature_c + rate * np.minimum(
        sample_minutes, duration_min
    )
    temperature = np.maximum(temperature, cfg.batch.end_temperature_c)
    return np.diff(temperature) / (cfg.batch.sample_seconds / 60.0)


def _piecewise_rates(rng: np.random.Generator, cfg: ModelConfig, intervals: int) -> np.ndarray:
    """Sample cooling, reheating and final-cooling rates over three equal segments."""
    if intervals % 3 != 0:
        raise ValueError("piecewise profile requires a sample count divisible into three segments")
    segment = intervals // 3
    target_rate_sum = (
        cfg.batch.end_temperature_c - cfg.batch.start_temperature_c
    ) / (segment * cfg.batch.sample_seconds / 60.0)
    for _ in range(1000):
        first = rng.uniform(-0.42, -0.28)
        second = rng.uniform(0.03, 0.12)
        third = target_rate_sum - first - second
        rates = np.repeat([first, second, third], segment)
        temperatures = _temperatures_from_minute_rates(rates, cfg)
        if (
            cfg.dataset.min_rate_c_min <= third <= cfg.dataset.max_rate_c_min
            and temperatures.min() >= cfg.dataset.min_temperature_c - 1e-12
            and temperatures.max() <= cfg.dataset.max_temperature_c + 1e-12
        ):
            return rates
    raise RuntimeError("Unable to construct a bounded piecewise temperature profile")


def _staircase_rates(rng: np.random.Generator, cfg: ModelConfig, intervals: int) -> np.ndarray:
    """Distribute cooling increments over evenly spaced intervals separated by holds."""
    cooling_rate = -rng.uniform(0.30, 0.45)
    total_drop = cfg.batch.start_temperature_c - cfg.batch.end_temperature_c
    full_cooling_intervals = int(total_drop // (-cooling_rate))
    remainder_drop = total_drop - full_cooling_intervals * (-cooling_rate)
    active_count = full_cooling_intervals + int(remainder_drop > 1e-12)
    positions = np.unique(
        np.floor(np.linspace(0, intervals - 1, active_count)).astype(int)
    )
    rates = np.zeros(intervals, dtype=float)
    rates[positions[:full_cooling_intervals]] = cooling_rate
    if remainder_drop > 1e-12:
        rates[positions[full_cooling_intervals]] = -remainder_drop
    return rates


def _smooth_random_rates(rng: np.random.Generator, cfg: ModelConfig, intervals: int) -> np.ndarray:
    """Sample zero-mean sinusoidal rate perturbations and reject paths outside the bounds."""
    midpoint = (np.arange(intervals, dtype=float) + 0.5) / intervals
    base_rate = (
        cfg.batch.end_temperature_c - cfg.batch.start_temperature_c
    ) / cfg.batch.total_minutes
    for _ in range(1000):
        phase_1 = rng.uniform(0.0, 2.0 * np.pi)
        phase_2 = rng.uniform(0.0, 2.0 * np.pi)
        variation = (
            rng.uniform(0.08, 0.18) * np.sin(2.0 * np.pi * midpoint + phase_1)
            + rng.uniform(0.02, 0.07) * np.sin(6.0 * np.pi * midpoint + phase_2)
        )
        variation -= variation.mean()
        rates = base_rate + variation
        temperatures = _temperatures_from_minute_rates(rates, cfg)
        if (
            rates.min() >= cfg.dataset.min_rate_c_min
            and rates.max() <= cfg.dataset.max_rate_c_min
            and temperatures.min() >= cfg.dataset.min_temperature_c - 1e-12
            and temperatures.max() <= cfg.dataset.max_temperature_c + 1e-12
        ):
            return rates
    raise RuntimeError("Unable to construct a bounded smooth-random temperature profile")


def random_profile(
    kind: str,
    rng: np.random.Generator,
    cfg: ModelConfig,
) -> TemperatureProfile:
    """Generate a temperature path from the selected random-profile family."""
    if kind not in PROFILE_KINDS:
        raise ValueError(f"Unknown profile kind: {kind}")
    time_s = _sample_times(cfg)
    intervals = time_s.size - 1
    if kind == "linear":
        rates = _linear_random_rates(rng, cfg, intervals)
    elif kind == "piecewise":
        rates = _piecewise_rates(rng, cfg, intervals)
    elif kind == "staircase":
        rates = _staircase_rates(rng, cfg, intervals)
    else:
        rates = _smooth_random_rates(rng, cfg, intervals)
    temperature = _temperatures_from_minute_rates(rates, cfg)
    return TemperatureProfile(time_s=time_s, temperature_c=temperature, kind=kind)
