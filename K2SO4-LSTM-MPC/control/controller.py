"""Constrained MPC decisions using an external forecast function."""
from collections import OrderedDict
import time
import numpy as np
from scipy.optimize import minimize
from physics.kinetics import solubility
from .objective import terminal_components


def rate_matrix(horizon):
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    horizon = int(horizon)
    lengths = [1, min(4, horizon-1)]
    q, r = divmod(horizon-sum(lengths), 3)
    lengths += [q+(i < r) for i in range(3)]
    lengths = [n for n in lengths if n]
    return np.repeat(np.eye(len(lengths)), lengths, axis=0)


class TrialProblem:
    """Prediction function accepts H temperatures, returns H x 6 quality and validity."""
    def __init__(self, predict, temperature, previous_rate, horizon, config,
                 quality_min=None, quality_max=None):
        self.predict = predict
        self.temperature = float(temperature)
        self.previous_rate = float(previous_rate)
        self.matrix = rate_matrix(horizon)
        self.config = config
        self.cache = OrderedDict()
        self.trace = []
        self.best = None
        self.deadline = None
        self.quality_min = quality_min
        self.quality_max = quality_max
        if not np.isfinite([temperature, previous_rate]).all():
            raise ValueError('nonfinite current condition')

    def evaluate(self, x):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError('cooperative optimization deadline reached')
        x = np.asarray(x, dtype=float)
        if x.shape != (self.matrix.shape[1],) or not np.isfinite(x).all():
            raise ValueError('invalid rate blocks')
        key = x.tobytes()
        if key in self.cache:
            return self.cache[key]
        started = time.perf_counter()
        rates = self.matrix @ x
        temperatures = self.temperature + np.cumsum(rates)  # exactly 60 seconds
        prediction = self.predict(temperatures)
        quality = np.asarray(prediction['quality'], dtype=float)
        valid = np.asarray(prediction['diagnostics']['valid'], dtype=bool)
        if quality.shape != (len(rates), 6) or valid.shape != (len(rates),):
            raise ValueError('unexpected prediction dimensions')
        c = self.config['constraints']; n = self.config['normalization']
        w = self.config['weights']; target = self.config['target']
        finite = np.isfinite(quality).all()
        saturation = quality[:, 0] / solubility(temperatures)
        margins = np.concatenate([
            temperatures-c['min_temperature_c'], c['max_temperature_c']-temperatures,
            x-c['min_rate_c_min'], c['max_rate_c_min']-x,
            np.nan_to_num(saturation-c['supersaturation_floor'], nan=-1e6,
                          posinf=-1e6, neginf=-1e6),
            np.nan_to_num(quality[:, 0], nan=-1e6, posinf=-1e6, neginf=-1e6),
            [0. if finite and valid.all() else -1.]])
        if finite and valid.all():
            terminal = quality[-1]
            components = terminal_components(terminal, self.config)
            components.update(
                move=w['move']*float(np.mean((rates/n['rate_c_min'])**2)),
                smooth=w['smooth']*float(np.mean((np.diff(np.r_[self.previous_rate, rates])/n['rate_change_c_min'])**2)))
            if self.quality_min is not None:
                scale = np.maximum(np.asarray(self.quality_max)-self.quality_min, 1e-12)
                excursion = (np.maximum(self.quality_min-quality, 0)+
                             np.maximum(quality-self.quality_max, 0))/scale
                components['domain'] = w.get('training_domain', 0)*float(np.mean(excursion**2))
            objective = float(sum(components.values()))
        else:
            components = {'invalid_penalty': 1e12}; objective = 1e12
        if not np.isfinite(objective):
            objective = 1e12; margins[-1] = -1.
        record = dict(rates=x.copy(), temperatures=temperatures, quality=quality,
                      objective=objective, components=components, margins=margins,
                      feasible=bool(np.min(margins) >= -1e-7),
                      valid=bool(finite and valid.all()),
                      minimum_saturation=float(np.min(saturation)) if finite else None,
                      prediction_seconds=time.perf_counter()-started)
        self.trace.append(dict(evaluation=len(self.trace)+1, objective=objective,
                               feasible=record['feasible'], valid=record['valid'],
                               minimum_margin=float(margins.min()),
                               prediction_seconds=record['prediction_seconds'],
                               rates=x.tolist()))
        if record['feasible'] and (self.best is None or objective < self.best['objective']):
            self.best = record
        self.cache[key] = record
        if len(self.cache) > 16:
            self.cache.popitem(last=False)
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError('deadline exceeded during prediction')
        return record


def solve_trial(problem, initial_rates, *, budget=30., maxiter=50, fd_step=1e-3):
    if not np.isfinite([budget, fd_step]).all() or budget <= 0 or fd_step <= 0 or maxiter < 1:
        raise ValueError('positive budget, iterations and finite-difference step required')
    # Report seed evaluation separately; optimizer timing includes all its model calls.
    baseline = problem.evaluate(initial_rates)
    count = len(problem.trace)
    started = time.perf_counter(); problem.deadline = started+budget
    result = None; timed_out = False
    c = problem.config['constraints']
    try:
        result = minimize(lambda x: problem.evaluate(x)['objective'], initial_rates,
                          method='SLSQP',
                          bounds=[(c['min_rate_c_min'], c['max_rate_c_min'])]*problem.matrix.shape[1],
                          constraints=[{'type': 'ineq', 'fun': lambda x: problem.evaluate(x)['margins']}],
                          options={'maxiter': maxiter, 'ftol': 1e-6, 'eps': fd_step})
    except TimeoutError:
        timed_out = True
    elapsed = time.perf_counter()-started
    problem.deadline = None
    # Do not run an unbudgeted final prediction: returned SLSQP point should be cached.
    final = None if result is None else problem.cache.get(np.asarray(result.x, dtype=float).tobytes())
    success = bool(result is not None and result.success and final is not None and final['feasible'])
    return dict(baseline=baseline, candidate=problem.best,
                solver_success=success, timed_out=timed_out,
                optimizer_reported_success=bool(result is not None and result.success),
                final_point_feasible=bool(final is not None and final['feasible']),
                solver_message='time budget exceeded' if result is None else str(result.message),
                solver_iterations=None if result is None else int(result.nit),
                solve_seconds=elapsed, within_budget=bool(elapsed <= budget),
                rollout_evaluations=len(problem.trace)-count,
                scope='single decision only; no plant action or closed-loop qualification')
