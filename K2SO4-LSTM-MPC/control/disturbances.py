"""Stored T/C observations; sensor noise never changes the physical plant state."""
import numpy as np


def noise_sequence(samples, temperature_std_c, concentration_relative_std, seed):
    if (not isinstance(samples,int) or samples<=0 or not isinstance(seed,int) or seed<0
        or not np.isfinite([temperature_std_c,concentration_relative_std]).all()
        or temperature_std_c<0 or concentration_relative_std<0):
        raise ValueError('Require positive sample count, nonnegative integer seed and finite nonnegative noise standard deviations')
    # Separate streams keep each sensor's prefix identical across run lengths.
    streams=np.random.SeedSequence(seed).spawn(2)
    t=np.random.default_rng(streams[0]).standard_normal(samples)*temperature_std_c
    c=np.random.default_rng(streams[1]).standard_normal(samples)*concentration_relative_std
    return np.column_stack([t,c])


class MeasurementHistory:
    def __init__(self, sequence):
        self.sequence=np.array(sequence,dtype=float,copy=True)
        if self.sequence.ndim!=2 or self.sequence.shape[1]!=2 or not np.isfinite(self.sequence).all():
            raise ValueError('Expected finite [time, temperature_error/relative_concentration_error] sequence')
        self.states=[];self.temperatures=[]

    def add(self, true_state, true_temperature):
        step=len(self.states)
        if step>=len(self.sequence):raise ValueError('Measurement sequence exhausted')
        state=np.array(true_state,dtype=float,copy=True)
        if state.ndim!=1 or len(state)<2 or not np.isfinite(state).all() or (state<0).any() or not np.isfinite(true_temperature):
            raise ValueError('Invalid physical observation source')
        dt,dc=self.sequence[step]
        state[0]*=1.+dc
        temperature=float(true_temperature+dt)
        if not np.isfinite(state[0]) or state[0]<0 or not np.isfinite(temperature):
            raise ValueError('Noise produced an invalid measurement; no clipping or resampling performed')
        self.states.append(state);self.temperatures.append(temperature)

    def window(self,steps=10):
        if steps<1 or not self.states:raise ValueError('No measurement history or invalid window length')
        return np.array(self.states[-steps:],copy=True),np.array(self.temperatures[-steps:],copy=True)

# Exogenous temperature execution errors and plant-only kinetic mismatch.
from dataclasses import replace
import numpy as np


def temperature_errors(samples, std_c, rho, seed):
    """AR(1), initialized at zero; std_c is the asymptotic standard deviation.

    Generated before optimization; independent of sensor streams. The initial
    inventory and temperature are unchanged. No clipping or random walk.
    """
    if (not isinstance(samples,int) or samples<1 or not isinstance(seed,int) or seed<0
        or not np.isfinite([std_c,rho]).all() or std_c<0 or not 0<=rho<1):
        raise ValueError('Require samples>=1, seed>=0, finite std>=0 and 0<=rho<1')
    z=np.random.default_rng(np.random.SeedSequence([seed,719])).standard_normal(samples-1)
    errors=np.zeros(samples)
    for i in range(1,samples):errors[i]=rho*errors[i-1]+std_c*np.sqrt(1-rho*rho)*z[i-1]
    return errors


def perturbed_config(cfg, growth_factor, nucleation_factor):
    if not np.isfinite([growth_factor,nucleation_factor]).all() or min(growth_factor,nucleation_factor)<=0:
        raise ValueError('Kinetic multipliers must be finite and positive')
    kinetics=replace(cfg.kinetics,kg_m_s=cfg.kinetics.kg_m_s*growth_factor,
                     kn1=cfg.kinetics.kn1*nucleation_factor,kn2=cfg.kinetics.kn2*nucleation_factor)
    result=replace(cfg,kinetics=kinetics)
    result.validate()
    return result


def advance_disturbed(plant, command, error):
    """PBM interpolates between actual temperature endpoints over the minute."""
    actual=float(command+error)
    if not np.isfinite(actual):raise ValueError('Nonfinite actual temperature')
    return plant.advance(actual)


def temperature_checks(temperatures, constraints):
    t=np.asarray(temperatures,dtype=float)
    rates=np.diff(t)  # This driver is restricted to one-minute sampling.
    return dict(minimum_actual_temperature_c=float(t.min()),maximum_actual_temperature_c=float(t.max()),
        temperature_violation_samples=int(np.sum((t<constraints['min_temperature_c']-1e-7)
                                                 |(t>constraints['max_temperature_c']+1e-7))),
        rate_violation_intervals=int(np.sum((rates<constraints['min_rate_c_min']-1e-7)
                                            |(rates>constraints['max_rate_c_min']+1e-7))))
