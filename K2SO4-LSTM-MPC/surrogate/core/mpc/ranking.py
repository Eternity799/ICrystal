"""Score saved terminal candidates with one frozen control objective.

This module never runs a PBM, optimizer, model, or plant. ``counts``, C and T
include the shared origin state and every future minute. The caller must supply
the same saved actual temperature plan to surrogate and truth scoring, and
``metadata={'terminal': True, 'origin': origin, 'horizon': H}`` must establish
``origin + H == 180``. Short windows cannot masquerade as batch endpoints.

Quality order is C, Y, D10, D50, D90, Span, with Y=(C0-C)/C0. The CSD quantiles
come directly from ``core.evaluation.cqa``. A positive training-domain weight
requires a frozen [minimum, maximum] pair of six-vectors, unless the frozen
objective explicitly records ``training_domain='disabled'``. Invalid scores
have ``objective=None`` and a finite negative minimum margin. Valid but
infeasible scores retain their objective for diagnostics, never for selection.
"""
import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from ..evaluation import cqa


CONSTRAINT_TOLERANCE = 1e-7
DEFAULT_SOLUBILITY = [.0629, .00246, -7.14e-6]
OBJECTIVE_NAME = 'd50_span_priority_v1'
_SECTIONS = {
    'target': ('d50_um', 'yield_fraction', 'span_upper'),
    'constraints': ('min_temperature_c', 'max_temperature_c',
                    'min_rate_c_min', 'max_rate_c_min', 'supersaturation_floor'),
    'normalization': ('d50_um', 'yield_fraction', 'span', 'rate_c_min', 'rate_change_c_min'),
    'weights': ('d50_terminal', 'yield_terminal', 'span_terminal',
                'span_excess', 'move', 'smooth'),
}


def _validated_objective(objective):
    if not isinstance(objective, Mapping) or objective.get('objective') != OBJECTIVE_NAME:
        raise ValueError('Expected d50_span_priority_v1 objective')
    result = {key: copy.deepcopy(objective.get(key)) for key in ('objective', *_SECTIONS)}
    for section, keys in _SECTIONS.items():
        values = result[section]
        if not isinstance(values, Mapping):
            raise ValueError(f'Missing {section} mapping')
        for key in keys:
            if key not in values or isinstance(values[key], (bool, np.bool_)):
                raise ValueError(f'Missing or invalid {section}.{key}')
            try:
                numeric = float(values[key])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Invalid {section}.{key}') from exc
            if not np.isfinite(numeric):
                raise ValueError(f'Nonfinite {section}.{key}')
            if section == 'normalization' and numeric <= 0:
                raise ValueError(f'Positive normalization.{key} required')
            if section == 'weights' and numeric < 0:
                raise ValueError(f'Nonnegative weights.{key} required')
            values[key] = numeric
    constraints = result['constraints']
    for lower, upper in (('min_temperature_c', 'max_temperature_c'),
                         ('min_rate_c_min', 'max_rate_c_min')):
        if constraints[lower] > constraints[upper]:
            raise ValueError(f'Invalid constraints interval: {lower}, {upper}')
    try:
        domain_weight = float(result['weights'].get('training_domain', 0.))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid weights.training_domain') from exc
    if not np.isfinite(domain_weight) or domain_weight < 0:
        raise ValueError('Nonnegative finite weights.training_domain required')
    result['weights']['training_domain'] = domain_weight
    mode = objective.get('training_domain', 'required' if domain_weight > 0 else 'disabled')
    if mode not in ('required', 'disabled'):
        raise ValueError("training_domain must be 'required' or explicitly 'disabled'")
    result['training_domain'] = mode
    coefficient_keys = [key for key in ('coefficients', 'solubility', 'solubility_coefficients') if key in objective]
    coefficients = objective[coefficient_keys[0]] if coefficient_keys else DEFAULT_SOLUBILITY
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
        raise ValueError('coefficients must contain three finite solubility polynomial values')
    for key in coefficient_keys[1:]:
        other = np.asarray(objective[key], dtype=float)
        if other.shape != (3,) or not np.array_equal(coefficients, other):
            raise ValueError('Conflicting frozen solubility coefficients')
    result['coefficients'] = coefficients.tolist()
    return result


def freeze_objective(path):
    """Read YAML without editing it; return detached sections plus exact provenance.

    ``source_content`` holds the complete original YAML, ``source_sha256`` hashes
    its exact bytes and ``source_path`` is absolute. Unrelated solver/model/noise
    configuration stays out of the scoring sections. Missing solubility values
    use the original K2SO4 polynomial, explicitly recorded in the snapshot.
    """
    source = Path(path).resolve()
    content = source.read_bytes()
    text = content.decode('utf-8')
    config = yaml.safe_load(text)
    frozen = _validated_objective(config)
    frozen.update(source_path=str(source), source_sha256=hashlib.sha256(content).hexdigest(),
                  source_content=text)
    return frozen


def _terminal_metadata(metadata, horizon):
    if not isinstance(metadata, Mapping):
        raise ValueError('terminal branch metadata is required')
    terminal = metadata.get('terminal')
    if not isinstance(terminal, (bool, np.bool_)) or not terminal:
        raise ValueError('metadata must identify a terminal=True branch')
    resolved = {}
    for key, aliases in (('origin', ('origin', 'origin_min')),
                         ('horizon', ('horizon', 'horizon_min', 'horizon_steps'))):
        present = [metadata[k] for k in aliases if k in metadata]
        if not present:
            raise ValueError(f'terminal metadata requires {key}')
        for value in present:
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value:
                raise ValueError(f'terminal metadata {key} must be an integer')
        if any(value != present[0] for value in present[1:]):
            raise ValueError(f'Conflicting terminal metadata {key}')
        resolved[key] = int(present[0])
    if resolved['horizon'] != horizon:
        raise ValueError('terminal metadata horizon does not match the full saved path')
    if resolved['origin'] < 0 or resolved['origin'] >= 180 or resolved['origin'] + horizon != 180:
        raise ValueError('terminal metadata must establish origin + horizon == 180')
    return dict(terminal=True, **resolved, terminal_min=180)


def _invalid(reason, *, horizon=None, training_domain=None, metadata=None):
    return dict(valid=False, feasible=False, objective=None, components={},
                minimum_margin=-1., minimum_saturation=None, reason=str(reason),
                terminal_quality=None, horizon=horizon, training_domain=training_domain,
                metadata=metadata, constraint_violations=[], constraint_tolerance=CONSTRAINT_TOLERANCE)


def trajectory_score(counts, concentration, temperatures, initial_concentration,
                     particle_mass, edges, objective, previous_rate=0.0,
                     quality_domain=None, *, metadata=None):
    """Return a serializable terminal score for one complete saved candidate path.

    Arrays must have shapes (H+1, 200), (H+1,), (H+1,), (200,) and (201,).
    ``initial_concentration`` is batch C0, not the origin concentration or total
    inventory including seed. Temperature rates are the one-minute differences
    of the *actual* supplied plan. Only future states enter path constraints and
    the domain penalty, matching ``control.controller.TrialProblem``.
    Runtime state/configuration errors return an explicit invalid result.
    """
    horizon = None
    domain_mode = None
    resolved_metadata = None
    try:
        config = _validated_objective(objective)
        domain_mode = config['training_domain']
        n = np.asarray(counts, dtype=float)
        c = np.asarray(concentration, dtype=float)
        t = np.asarray(temperatures, dtype=float)
        pm = np.asarray(particle_mass, dtype=float)
        size_edges = np.asarray(edges, dtype=float)
        if n.ndim != 2 or n.shape[1] != 200 or n.shape[0] < 2:
            raise ValueError('counts must have shape (H+1, 200), H >= 1')
        horizon = len(n) - 1
        resolved_metadata = _terminal_metadata(metadata, horizon)
        if c.shape != (horizon+1,) or t.shape != c.shape:
            raise ValueError('C/T paths must include the initial state and all H future states')
        if pm.shape != (200,) or size_edges.shape != (201,):
            raise ValueError('particle_mass/edges must match the frozen 200-bin grid')
        if not all(np.isfinite(array).all() for array in (n, c, t, pm, size_edges)):
            raise ValueError('nonfinite saved trajectory or grid')
        if np.any(n < 0) or np.any(pm <= 0):
            raise ValueError('negative counts or nonpositive particle mass')
        if np.any(size_edges <= 0) or np.any(np.diff(size_edges) <= 0):
            raise ValueError('size edges must be positive and strictly increasing')
        c0 = float(initial_concentration)
        previous = float(previous_rate)
        if not np.isfinite([c0, previous]).all() or c0 <= 0:
            raise ValueError('positive finite initial_concentration and finite previous_rate required')
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            mass = n @ pm
            if not np.isfinite(mass).all() or np.any(mass <= 0):
                raise ValueError('zero or nonfinite crystal mass in saved trajectory')
            quantiles = np.stack([cqa(row, pm, size_edges) for row in n[1:]])
            quality = np.column_stack((c[1:], (c0-c[1:])/c0, quantiles))
            if not np.isfinite(quality).all() or np.any(quality[:, 5] < 0):
                raise ValueError('invalid mass quantiles or quality')
            rates = np.diff(t)
            solubility = np.polynomial.polynomial.polyval(t[1:], config['coefficients'])
            if not np.isfinite(solubility).all() or np.any(solubility <= 0):
                raise ValueError('invalid frozen solubility on the temperature path')
            saturation = c[1:] / solubility
            if not np.isfinite(rates).all() or not np.isfinite(saturation).all():
                raise ValueError('nonfinite actual temperature rate or saturation')
            constraints = config['constraints']; norm = config['normalization']
            weights = config['weights']; target = config['target']
            q = quality[-1]
            components = dict(
                d50=float(weights['d50_terminal']*((q[3]-target['d50_um'])/norm['d50_um'])**2),
                yield_term=float(weights['yield_terminal']*((q[1]-target['yield_fraction'])/norm['yield_fraction'])**2),
                span=float(weights['span_terminal']*q[5]/norm['span']),
                span_excess=float(weights['span_excess']*(max(q[5]-target['span_upper'], 0.)/norm['span'])**2),
                move=float(weights['move']*np.mean((rates/norm['rate_c_min'])**2)),
                smooth=float(weights['smooth']*np.mean((np.diff(np.r_[previous, rates])/norm['rate_change_c_min'])**2)),
                domain=0.)
            if domain_mode != 'disabled' and weights['training_domain'] > 0:
                if quality_domain is None:
                    raise ValueError('positive training_domain weight requires frozen quality_domain')
                domain = np.asarray(quality_domain, dtype=float)
                if domain.shape != (2, 6) or not np.isfinite(domain).all() or np.any(domain[1] < domain[0]):
                    raise ValueError('quality_domain must be finite ordered [minimum, maximum] six-vectors')
                lower, upper = domain
                scale = np.maximum(upper-lower, 1e-12)
                excursion = (np.maximum(lower-quality, 0.) + np.maximum(quality-upper, 0.))/scale
                components['domain'] = float(weights['training_domain']*np.mean(excursion**2))
            cost = float(sum(components.values()))
            if not np.isfinite(cost) or not np.isfinite(list(components.values())).all():
                raise ValueError('nonfinite trajectory objective')
            margins = dict(
                min_temperature=t[1:]-constraints['min_temperature_c'],
                max_temperature=constraints['max_temperature_c']-t[1:],
                min_rate=rates-constraints['min_rate_c_min'],
                max_rate=constraints['max_rate_c_min']-rates,
                supersaturation=saturation-constraints['supersaturation_floor'],
                concentration=c[1:])
            all_margins = np.concatenate([*margins.values(), [0.]])
            if not np.isfinite(all_margins).all():
                raise ValueError('nonfinite constraint margin')
            minimum_margin = float(np.min(all_margins))
            violations = [key for key, values in margins.items() if np.min(values) < -CONSTRAINT_TOLERANCE]
        return dict(valid=True, feasible=not violations, objective=cost, components=components,
                    minimum_margin=minimum_margin, minimum_saturation=float(np.min(saturation)),
                    reason=None if not violations else 'constraint_violation: ' + ', '.join(violations),
                    terminal_quality=q.tolist(), horizon=horizon, training_domain=domain_mode,
                    metadata=resolved_metadata, constraint_violations=violations,
                    constraint_tolerance=CONSTRAINT_TOLERANCE, objective_name=OBJECTIVE_NAME,
                    objective_source_sha256=objective.get('source_sha256'))
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        return _invalid(exc, horizon=horizon, training_domain=domain_mode, metadata=resolved_metadata)


def _score_state(score):
    """Treat inconsistent valid flags and nonfinite/absent costs as invalid."""
    if not isinstance(score, Mapping):
        return False, False, None
    try:
        cost = score.get('objective')
        if cost is None or isinstance(cost, (bool, np.bool_)):
            return False, False, None
        cost = float(cost)
        valid = bool(score.get('valid', False)) and bool(np.isfinite(cost))
        feasible = valid and bool(score.get('feasible', False))
        return valid, feasible, cost if valid else None
    except (TypeError, ValueError, OverflowError):
        return False, False, None


def compare_candidates(predicted_scores, truth_scores, tie_tolerance):
    """Compare one paired group of correlated candidate alternatives.

    Selection uses only valid, feasible finite costs, with index as a deterministic
    tie breaker. Truth's best comes from *all* truth-feasible candidates, including
    candidates falsely rejected by the model. ``valid_pairs`` counts both-valid
    finite pairs; ``feasible_pairs`` counts pairs feasible on both sides.

    A near tie between the best two eligible costs on either side makes the group
    uninformative. ``mismatch`` is None for uninformative groups; it is never a
    fabricated successful ranking. Regret requires a truth-feasible selected
    candidate and truth best. Normalization divides the raw cost difference by
    max(worst truth-feasible cost - best truth-feasible cost, 1e-12), using all
    truth-feasible candidates. For informative finite groups whose selected
    truth is feasible, normalized regret lies in [0, 1], including when the best
    truth cost is zero. An invalid/infeasible truth selection is reported
    separately with regret None. Candidate counts are not counts of independent
    experiments.
    """
    try:
        tolerance = float(tie_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('finite nonnegative tie_tolerance required') from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('finite nonnegative tie_tolerance required')
    predicted = list(predicted_scores); truth = list(truth_scores)
    if len(predicted) != len(truth):
        raise ValueError('predicted and truth scores must be paired candidates')
    ps = [_score_state(row) for row in predicted]
    ts = [_score_state(row) for row in truth]
    eligible_predicted = sorted((i for i, state in enumerate(ps) if state[1]), key=lambda i: (ps[i][2], i))
    eligible_truth = sorted((i for i, state in enumerate(ts) if state[1]), key=lambda i: (ts[i][2], i))
    best_predicted = eligible_predicted[0] if eligible_predicted else None
    best_truth = eligible_truth[0] if eligible_truth else None
    predicted_tie = len(eligible_predicted) > 1 and ps[eligible_predicted[1]][2]-ps[best_predicted][2] <= tolerance
    truth_tie = len(eligible_truth) > 1 and ts[eligible_truth[1]][2]-ts[best_truth][2] <= tolerance
    selected_truth_invalid = best_predicted is not None and not ts[best_predicted][0]
    selected_truth_infeasible = best_predicted is not None and ts[best_predicted][0] and not ts[best_predicted][1]
    selected_eligible = best_predicted is not None and ts[best_predicted][1]
    truth_cost = ts[best_predicted][2] if selected_eligible else None
    regret = max(truth_cost-ts[best_truth][2], 0.) if selected_eligible and best_truth is not None else None
    denominator = max(ts[eligible_truth[-1]][2]-ts[best_truth][2], 1e-12) if best_truth is not None else None
    normalized_regret = regret/denominator if regret is not None else None
    if normalized_regret is not None and not np.isfinite(normalized_regret):
        normalized_regret = None
    informative = bool(selected_eligible and len(eligible_truth) >= 2 and len(eligible_predicted) >= 2
                       and not truth_tie and not predicted_tie)
    if best_truth is None:
        reason = 'no_truth_feasible_candidate'
    elif best_predicted is None:
        reason = 'no_predicted_feasible_candidate'
    elif selected_truth_invalid:
        reason = 'selected_truth_invalid'
    elif selected_truth_infeasible:
        reason = 'selected_truth_infeasible'
    elif truth_tie or predicted_tie:
        reason = 'near_tie'
    elif not informative:
        reason = 'insufficient_comparable_candidates'
    else:
        reason = None
    return dict(
        best_predicted_index=best_predicted, best_truth_index=best_truth,
        truth_cost_of_predicted_choice=truth_cost,
        selected_truth_objective=ts[best_predicted][2] if best_predicted is not None else None,
        best_truth_objective=ts[best_truth][2] if best_truth is not None else None,
        regret=regret, normalized_regret=normalized_regret,
        normalization_denominator=denominator,
        normalization='truth_feasible_candidate_cost_span', informative=informative,
        tie=bool(truth_tie or predicted_tie), truth_tie=bool(truth_tie), predicted_tie=bool(predicted_tie),
        mismatch=bool(regret > tolerance) if informative else None,
        selected_truth_invalid=bool(selected_truth_invalid), selected_truth_infeasible=bool(selected_truth_infeasible),
        false_feasible_count=sum(p[1] and t[0] and not t[1] for p, t in zip(ps, ts)),
        false_infeasible_count=sum(p[0] and not p[1] and t[1] for p, t in zip(ps, ts)),
        false_rejected_feasible_count=sum(t[1] and not p[1] for p, t in zip(ps, ts)),
        predicted_feasible_truth_invalid_count=sum(p[1] and not t[0] for p, t in zip(ps, ts)),
        valid_pairs=sum(p[0] and t[0] for p, t in zip(ps, ts)),
        feasible_pairs=sum(p[1] and t[1] for p, t in zip(ps, ts)),
        predicted_feasible_count=len(eligible_predicted), truth_feasible_count=len(eligible_truth),
        invalid_predicted_count=sum(not p[0] for p in ps), invalid_truth_count=sum(not t[0] for t in ts),
        candidate_count=len(ps), tie_tolerance=tolerance, reason=reason,
        unit='one correlated candidate group; candidate_count is not independent experiments')
