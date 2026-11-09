"""Subset loading and balanced parent-level sampling for control-oriented training.

Subset selection uses manifest metadata only.  Response batches sample a kind,
then a mother trajectory, then one group; a mother with many groups therefore
does not receive more probability merely because it has more files.
"""
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..data import GRID_KEYS, ROLES, SampleFactory, read_manifest, read_npz, stack_samples, validate_arrays


KINDS = ('short', 'long', 'terminal')


def _validate_manifest(manifest):
    """Keep read_manifest's checks when an already read manifest is supplied."""
    if manifest.get('schema_version') != 1 or manifest.get('status') not in ('dataset_complete_not_trained', '\u5b8c\u6574CSD\u6570\u636e\u751f\u6210\u5b8c\u6210_\u672a\u8bad\u7ec3'):
        raise ValueError('A fully generated schema-1 nominal dataset is required; partial datasets cannot be trained.')
    plan, completed = manifest['plan'], manifest['completed']
    expected = {entry['id']:entry for entry in plan}
    actual = {entry['id']:entry for entry in completed}
    if len(expected) != len(plan) or len(actual) != len(completed):
        raise ValueError('Duplicate base-trajectory IDs.')
    if set(expected) != set(actual):
        raise ValueError('The planned and completed base-trajectory IDs differ.')
    for entry in completed:
        if (entry['role'] not in ROLES or
                any(entry[key] != expected[entry['id']][key] for key in ('role', 'family'))):
            raise ValueError('Completed roles or trajectory families differ from the frozen plan.')
        filenames = [item['file'] for item in entry['files']]
        if len(set(filenames)) != len(filenames):
            raise ValueError('A base trajectory has duplicate file records.')
    return expected, actual


def _bool_scalar(value, name):
    scalar = np.asarray(value)
    if scalar.shape != () or scalar.dtype.kind != 'b':
        raise ValueError(f"{name} must be a Boolean scalar.")
    return bool(scalar)


def _terminal(group):
    # Reaching minute 180 is insufficient: an ordinary origin=179, H=1 group
    # also reaches minute 180, but is explicitly terminal=False in this design.
    if 'terminal' in group:
        return _bool_scalar(group['terminal'], 'terminal')
    if 'terminal_plan' in group:
        return _bool_scalar(group['terminal_plan'], 'terminal_plan')
    return False


def _nominal(entry, manifest):
    if 'nominal_initial' in entry:
        return _bool_scalar(entry['nominal_initial'], 'nominal_initial')
    every = manifest.get('resolved_config', {}).get('initial', {}).get('nominal_every', 4)
    seed = entry.get('seed')
    return bool(every and seed and int(seed[-1]) % int(every) == 0)


def _has_terminal(entry):
    for group in entry['groups']:
        if not _terminal(group):
            continue
        offsets = group.get('offsets_c')
        if offsets is None or np.any(np.asarray(offsets) != 0):
            return True
    return False


def select_parent_ids(manifest, role, per_family, seed):
    """Return sorted manifest-only parent IDs in one frozen role.

    ``per_family=0`` keeps all parents in that role.  Otherwise every family
    receives the same number, capped by the smallest family's availability.
    The nominal/varied quota follows each family's original ratio, with both
    represented when at least two slots are available.  Terminal parents have
    priority within these strata, so terminal_every=12 cannot turn the subset
    into an all-nominal sample when nominal_every=4. A single-parent smoke
    subset prioritizes a terminal parent; initial-stratum coverage needs two
    slots. Ordinary parents are retained when at least two slots permit it.
    """
    expected, completed = _validate_manifest(manifest)
    if role not in ROLES:
        raise ValueError('Invalid dataset role.')
    if (isinstance(per_family, (bool, np.bool_)) or not isinstance(per_family, (int, np.integer))
            or per_family < 0):
        raise ValueError('per_family must be a nonnegative integer.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    entries = sorted((entry for entry in expected.values() if entry['role'] == role), key=lambda x:x['id'])
    if not entries:
        raise ValueError('The selected role has no base trajectories.')
    if not per_family:
        return [entry['id'] for entry in entries]
    families = {}
    for entry in entries:
        families.setdefault(entry['family'], []).append(entry)
    size = min(int(per_family), min(len(rows) for rows in families.values()))
    rng = np.random.default_rng(int(seed))
    selected = []
    for family in sorted(families):
        rows = families[family]
        terminal_ids = {entry['id'] for entry in rows if _has_terminal(completed[entry['id']])}
        if size == 1:
            priority = [entry for entry in rows if entry['id'] in terminal_ids] or rows
            selected.append(priority[int(rng.integers(len(priority)))]['id'])
            continue
        nominal = [entry for entry in rows if _nominal(entry, manifest)]
        varied = [entry for entry in rows if not _nominal(entry, manifest)]
        nominal_size = int(np.floor(size*len(nominal)/len(rows)+.5))
        if size >= 2 and nominal and varied:
            nominal_size = min(size-1, max(1, nominal_size))
        nominal_size = min(len(nominal), max(size-len(varied), nominal_size))
        family_selected = []
        for pool, quota in ((nominal, nominal_size), (varied, size-nominal_size)):
            terminal = [entry for entry in pool if entry['id'] in terminal_ids]
            ordinary = [entry for entry in pool if entry['id'] not in terminal_ids]
            order = [terminal[int(i)] for i in rng.permutation(len(terminal))]
            order += [ordinary[int(i)] for i in rng.permutation(len(ordinary))]
            family_selected.extend(order[:quota])
        if all(entry['id'] in terminal_ids for entry in family_selected):
            # Keep the initial-stratum quota while avoiding an all-terminal
            # subset whenever a nonterminal replacement in that stratum exists.
            for position, entry in enumerate(family_selected):
                ordinary = [candidate for candidate in rows if candidate['id'] not in terminal_ids
                            and _nominal(candidate, manifest) == _nominal(entry, manifest)]
                if ordinary:
                    family_selected[position] = ordinary[int(rng.integers(len(ordinary)))]
                    break
        selected.extend(entry['id'] for entry in family_selected)
    return sorted(selected)


def _branch_metadata(item, arrays):
    manifest_terminal = _terminal(item)
    array_terminal = _terminal(arrays)
    manifest_has_terminal = any(key in item for key in ('terminal', 'terminal_plan'))
    arrays_have_terminal = any(key in arrays for key in ('terminal', 'terminal_plan'))
    if manifest_has_terminal and arrays_have_terminal and manifest_terminal != array_terminal:
        raise ValueError('Branch terminal metadata differ between manifest and NPZ.')
    terminal = manifest_terminal if manifest_has_terminal else array_terminal
    manifest_pattern = item.get('pattern')
    array_pattern = str(arrays['plan_pattern']) if 'plan_pattern' in arrays else None
    if manifest_pattern is not None and array_pattern is not None and manifest_pattern != array_pattern:
        raise ValueError('Branch pattern metadata differ between manifest and NPZ.')
    pattern = manifest_pattern if manifest_pattern is not None else (array_pattern or 'offset')
    if not isinstance(pattern, str) or not pattern:
        raise ValueError('Branch pattern must be a nonempty string.')
    return terminal, pattern


def load_selected_dataset(root, roles, parent_ids, manifest=None):
    """Load selected parents only, with label, role and branch consistency checks.

    ``parent_ids`` maps exactly the requested roles to nonempty ID lists.
    Role membership is checked for every requested ID before hashing or opening
    any NPZ.  Unselected parents and roles never reach ``read_npz``.
    """
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or not set(roles) <= set(ROLES):
        raise ValueError('Invalid dataset role.')
    root = Path(root)
    manifest = read_manifest(root) if manifest is None else manifest
    expected, _ = _validate_manifest(manifest)
    if not isinstance(parent_ids, dict) or set(parent_ids) != set(roles):
        raise ValueError('parent_ids must match exactly the requested roles.')
    selected = {}
    for role in roles:
        ids = parent_ids[role]
        if not isinstance(ids, (list, tuple)) or not ids or len(set(ids)) != len(ids):
            raise ValueError('Selected base-trajectory IDs are empty or duplicated.')
        if any(parent not in expected or expected[parent]['role'] != role for parent in ids):
            raise ValueError('A selected base-trajectory ID is missing or belongs to a different role.')
        selected[role] = set(ids)
    result = {role:[] for role in roles}
    reference, seen = None, {}
    for entry in manifest['completed']:
        if entry['role'] not in selected or entry['id'] not in selected[entry['role']]:
            continue  # Filtering precedes hash construction, file hashing and NPZ access.
        hashes = {item['file']:item['sha256'] for item in entry['files']}
        arrays = read_npz(root, entry['id']+'.npz', hashes)
        validate_arrays(arrays, (181, 200))
        if (str(arrays['role']) != entry['role'] or str(arrays['parent_id']) != entry['id']
                or not np.array_equal(arrays['time_s'], np.arange(181)*60.)):
            raise ValueError('Base-trajectory ID, role or time mismatch.')
        if reference is None:
            reference = {key:arrays[key] for key in GRID_KEYS}
        if any(not np.array_equal(arrays[key], reference[key]) for key in GRID_KEYS):
            raise ValueError('Size grid and material properties must be identical within one run.')
        digest = hashlib.sha256(arrays['counts'].tobytes()+arrays['concentration'].tobytes()+
                                arrays['temperature'].tobytes()).hexdigest()
        if digest in seen and seen[digest] != entry['role']:
            raise ValueError('Different roles contain exactly duplicated dynamic trajectories.')
        seen[digest] = entry['role']
        arrays.update(id=entry['id'], role=entry['role'], family=entry['family'], groups=[])
        for item in entry['groups']:
            branch = read_npz(root, item['file'], hashes)
            offsets = branch['offsets_c']; origin = int(branch['origin_min'])
            if offsets.ndim != 1 or not np.isfinite(offsets).all() or len(set(offsets.tolist())) != len(offsets):
                raise ValueError('Invalid branch temperature offsets.')
            zeros = np.flatnonzero(offsets == 0)
            horizon = int(item['horizon_steps'])
            if (len(zeros) != 1 or origin != item['origin_min'] or not 0 <= origin < 180
                    or not 1 <= horizon <= 180-origin):
                raise ValueError('Invalid branch baseline, origin or horizon.')
            validate_arrays(branch, (len(offsets), horizon+1, 200))
            if str(branch['role']) != entry['role'] or str(branch['parent_id']) != entry['id']:
                raise ValueError('Branch role or ID differs from its base trajectory.')
            if any(not np.array_equal(arrays[key], branch[key]) for key in (*GRID_KEYS, 'initial_total_solute')):
                raise ValueError('A branch changed the global inventory or size grid.')
            for key in ('counts', 'concentration', 'temperature'):
                if not np.allclose(branch[key][:, 0], arrays[key][origin], rtol=1e-12, atol=1e-14):
                    raise ValueError('Paired branches do not start from the same true state.')
                if not np.allclose(branch[key][int(zeros[0])], arrays[key][origin:origin+horizon+1],
                                   rtol=1e-8, atol=1e-12):
                    raise ValueError('The zero-offset branch does not reproduce its base trajectory.')
            expected_time = np.broadcast_to(np.arange(origin, origin+horizon+1)*60., (len(offsets), horizon+1))
            if not np.array_equal(branch['time_s'], expected_time):
                raise ValueError('Invalid branch time indices.')
            terminal, pattern = _branch_metadata(item, branch)
            if terminal and horizon != 180-origin:
                raise ValueError('Terminal branches must extend to the batch endpoint.')
            order = [int(zeros[0])] + [i for i in range(len(offsets)) if i != int(zeros[0])]
            branch = {key:(value[order] if key in ('counts', 'concentration', 'temperature', 'valid_mask',
                                                  'time_s', 'saturation', 'offsets_c') else value)
                      for key, value in branch.items()}
            branch.update(origin=origin, file=item['file'], horizon=horizon, terminal=terminal, pattern=pattern)
            arrays['groups'].append(branch)
        result[entry['role']].append(arrays)
    if any(not rows for rows in result.values()):
        raise ValueError('The selected role has no base trajectories.')
    for rows in result.values():
        rows.sort(key=lambda row:row['id'])
    return result


class MPCSampleFactory(SampleFactory):
    """Shared histories, initial concentration C0, balanced response sampling and coverage counters.

    ``seen_parents`` maps parent IDs to draws; ``seen_groups`` maps
    ``(parent_id, file)`` to response draws. ``draws`` counts mother-window and
    response-group draws, without counting each temperature offset separately.
    ``availability`` reports parent/group counts per kind; ``last_mixture``
    records the effective probabilities after absent kinds are removed.
    """
    def __init__(self, rows, history=10, time_edges=(0, 10, 60, 120, 180)):
        super().__init__(rows, history, time_edges)
        self.seen_parents = Counter()
        self.seen_groups = Counter()
        self.draws = 0
        self.by_kind = {kind:[] for kind in KINDS}
        for row, branches in self.paired:
            for kind in KINDS:
                available = [branch for branch in branches if self._kind(branch) == kind]
                if available:
                    self.by_kind[kind].append((row, available))
        self.availability = {kind:dict(parents=len(parents), groups=sum(len(branches) for _, branches in parents))
                             for kind, parents in self.by_kind.items()}
        self.last_mixture = {kind:0. for kind in KINDS}

    @staticmethod
    def _kind(branch):
        if _terminal(branch):
            return 'terminal'
        return 'short' if branch['horizon'] <= 10 else 'long'

    def window(self, row, origin, horizon):
        sample = super().window(row, origin, horizon)
        sample['initial_concentration'] = torch.tensor(float(row['concentration'][0]), dtype=torch.float64)
        self.seen_parents[row['id']] += 1
        self.draws += 1
        return sample

    def pair(self, row, branch, horizon, device='cpu'):
        sample = super().pair(row, branch, horizon, device)
        sample['initial_concentration'] = torch.full((len(branch['offsets_c']),), float(row['concentration'][0]),
                                                     dtype=torch.float64, device=device)
        self.seen_parents[row['id']] += 1
        self.seen_groups[(row['id'], branch['file'])] += 1
        self.draws += 1
        return sample

    def sample_pairs(self, groups, horizon, rng, device='cpu', mixture=None, full_terminal=False):
        """Draw response groups, retaining full terminal labels when requested.

        A missing kind loses its weight and the remaining weights are
        normalized. ``mixture=None`` gives equal weight to available kinds.
        Only groups containing a nonzero offset are eligible.  Mother sampling
        itself remains inherited from SampleFactory, with true history ending
        at the origin and future true states stored only as targets.
        """
        if isinstance(groups, bool) or not isinstance(groups, (int, np.integer)) or groups < 0:
            raise ValueError('groups must be a nonnegative integer.')
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 180:
            raise ValueError('horizon must be an integer from 1 to 180.')
        if not groups:
            return []
        if not self.paired:
            raise ValueError('No nonzero temperature branches are available; response training cannot be skipped silently.')
        mixture = {kind:1. for kind in KINDS} if mixture is None else mixture
        if not isinstance(mixture, dict) or set(mixture) != set(KINDS):
            raise ValueError('mixture must contain short, long and terminal probabilities.')
        weights = np.array([float(mixture[kind]) for kind in KINDS], dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Mixture probabilities must be finite, nonnegative and have positive total weight.')
        weights *= np.array([bool(self.by_kind[kind]) for kind in KINDS])
        if weights.sum() <= 0:
            raise ValueError('No branch kind with positive weight is available; inspect availability.')
        probabilities = weights/weights.sum()
        self.last_mixture = {kind:float(weight) for kind, weight in zip(KINDS, probabilities)}
        samples = []
        for _ in range(groups):
            kind = KINDS[int(rng.choice(len(KINDS), p=probabilities))]
            parents = self.by_kind[kind]
            row, branches = parents[int(rng.integers(len(parents)))]
            branch = branches[int(rng.integers(len(branches)))]
            sample_horizon = branch['horizon'] if full_terminal and kind == 'terminal' else horizon
            samples.append(self.pair(row, branch, sample_horizon, device))
        return samples
