"""Complete nominal PBM data with parent-level role isolation; training never opens test states."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .features import state_features, action_features
from .io import sha256

GRID_KEYS = ('particle_mass_kg', 'size_edges_m', 'size_pivots_m')
ROLES = ('train', 'validation', 'test')


def read_manifest(root):
    root = Path(root)
    m = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if m.get('schema_version') != 1 or m.get('status') not in ('dataset_complete_not_trained', '\u5b8c\u6574CSD\u6570\u636e\u751f\u6210\u5b8c\u6210_\u672a\u8bad\u7ec3'):
        raise ValueError('A fully generated schema-1 nominal dataset is required; partial datasets cannot be trained.')
    plan, completed = m['plan'], m['completed']
    expected = {e['id']:e for e in plan}
    if len(expected) != len(plan) or len({e['id'] for e in completed}) != len(completed):
        raise ValueError('Duplicate base-trajectory IDs.')
    if set(expected) != {e['id'] for e in completed}:
        raise ValueError('The planned and completed base-trajectory IDs differ.')
    for e in completed:
        if e['role'] not in ROLES or any(e[k] != expected[e['id']][k] for k in ('role', 'family')):
            raise ValueError('Completed roles or trajectory families differ from the frozen plan.')
        filenames = [x['file'] for x in e['files']]
        if len(set(filenames)) != len(filenames):
            raise ValueError('A base trajectory has duplicate file records.')
    return m


def read_npz(root, name, hashes):
    if Path(name).name != name or Path(name).suffix != '.npz':
        raise ValueError('Manifest files must be NPZ filenames in the dataset directory.')
    path = Path(root)/name
    if name not in hashes or sha256(path) != hashes[name]:
        raise ValueError(f"Dataset checksum mismatch: {name}")
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def validate_arrays(a, shape):
    n = a['counts']; c = a['concentration']; t = a['temperature']
    if n.shape != shape or c.shape != shape[:-1] or t.shape != c.shape:
        raise ValueError('CSD, concentration or temperature label dimensions are invalid.')
    for k in ('counts', 'concentration', 'temperature', 'initial_total_solute', *GRID_KEYS):
        if not np.isfinite(a[k]).all():
            raise ValueError(f"Nonfinite labels: {k}")
    pm, edges, pivots = [a[k] for k in GRID_KEYS]
    total = float(a['initial_total_solute'])
    if (pm.shape != (200,) or pivots.shape != (200,) or edges.shape != (201,)
            or np.any(pm <= 0) or np.any(pivots <= 0) or np.any(np.diff(edges) <= 0)
            or np.any(pivots <= edges[:-1]) or np.any(pivots >= edges[1:])):
        raise ValueError('Invalid size grid or per-particle mass.')
    if total <= 0 or np.any(n < 0) or np.any(c < 0):
        raise ValueError('Invalid inventory, counts or concentration.')
    mass = (n*pm).sum(-1)
    if np.any(n.sum(-1) <= 0) or np.any(mass <= 0):
        raise ValueError('This formulation requires positive particle number and crystal mass after seeding.')
    if np.max(np.abs(c+mass-total))/total > 1e-8:
        raise ValueError('CSD and concentration labels violate the global material inventory.')
    if a['valid_mask'].shape != c.shape or not a['valid_mask'].all():
        raise ValueError('Missing or invalid labels are not accepted.')
    if int(a['schema_version']) != 1 or float(a['interval_seconds']) != 60:
        raise ValueError('Dataset format or sampling interval mismatch.')


def load_dataset(root, roles=('train', 'validation')):
    if not roles or len(set(roles)) != len(roles) or not set(roles) <= set(ROLES):
        raise ValueError('Invalid dataset role.')
    root = Path(root); manifest = read_manifest(root)
    result = {role:[] for role in roles}; reference = None; seen = {}
    for entry in manifest['completed']:
        if entry['role'] not in roles:
            continue  # Filter roles before opening trajectory files.
        hashes = {item['file']:item['sha256'] for item in entry['files']}
        a = read_npz(root, entry['id']+'.npz', hashes)
        validate_arrays(a, (181, 200))
        if (str(a['role']) != entry['role'] or str(a['parent_id']) != entry['id']
                or not np.array_equal(a['time_s'], np.arange(181)*60.)):
            raise ValueError('Base-trajectory ID, role or time mismatch.')
        if reference is None:
            reference = {k:a[k] for k in GRID_KEYS}
        if any(not np.array_equal(a[k], reference[k]) for k in GRID_KEYS):
            raise ValueError('Size grid and material properties must be identical within one run.')
        digest = hashlib.sha256(a['counts'].tobytes()+a['concentration'].tobytes()+a['temperature'].tobytes()).hexdigest()
        if digest in seen and seen[digest] != entry['role']:
            raise ValueError('Different roles contain exactly duplicated dynamic trajectories.')
        seen[digest] = entry['role']
        a.update(id=entry['id'], role=entry['role'], family=entry['family'], groups=[])
        for item in entry['groups']:
            b = read_npz(root, item['file'], hashes)
            offsets = b['offsets_c']; origin = int(b['origin_min'])
            if offsets.ndim != 1 or not np.isfinite(offsets).all() or len(set(offsets.tolist())) != len(offsets):
                raise ValueError('Invalid branch temperature offsets.')
            zeros = np.flatnonzero(offsets == 0)
            h = int(item['horizon_steps'])
            if len(zeros) != 1 or origin != item['origin_min'] or not 0 <= origin < 180 or not 1 <= h <= 180-origin:
                raise ValueError('Invalid branch baseline, origin or horizon.')
            validate_arrays(b, (len(offsets), h+1, 200))
            if str(b['role']) != entry['role'] or str(b['parent_id']) != entry['id']:
                raise ValueError('Branch role or ID differs from its base trajectory.')
            if any(not np.array_equal(a[k], b[k]) for k in (*GRID_KEYS, 'initial_total_solute')):
                raise ValueError('A branch changed the global inventory or size grid.')
            for k in ('counts', 'concentration', 'temperature'):
                if not np.allclose(b[k][:, 0], a[k][origin], rtol=1e-12, atol=1e-14):
                    raise ValueError('Paired branches do not start from the same true state.')
                if not np.allclose(b[k][int(zeros[0])], a[k][origin:origin+h+1], rtol=1e-8, atol=1e-12):
                    raise ValueError('The zero-offset branch does not reproduce its base trajectory.')
            expected_time = np.broadcast_to(np.arange(origin, origin+h+1)*60., (len(offsets), h+1))
            if not np.array_equal(b['time_s'], expected_time):
                raise ValueError('Invalid branch time indices.')
            order = [int(zeros[0])] + [i for i in range(len(offsets)) if i != int(zeros[0])]
            b = {k:(v[order] if k in ('counts', 'concentration', 'temperature', 'valid_mask', 'time_s', 'saturation', 'offsets_c') else v)
                 for k,v in b.items()}
            b.update(origin=origin, file=item['file'], horizon=h)
            a['groups'].append(b)
        result[entry['role']].append(a)
    if any(not rows for rows in result.values()):
        raise ValueError('The selected role has no base trajectories.')
    return result


def fit_scaler(rows, coefficients):
    if not rows or any(row['role'] != 'train' for row in rows):
        raise ValueError('Scales may only be fitted to train-role base trajectories.')
    n = np.concatenate([row['counts'] for row in rows])
    floor = max(float(np.quantile(n.sum(-1), .9))/200*1e-6, 1e-12)
    count_scale = np.maximum(np.quantile(n, .9, axis=0), floor)
    pivots = rows[0]['size_pivots_m']; pm = rows[0]['particle_mass_kg']
    moments = np.stack([(n*pivots**j).sum(-1) for j in range(4)], -1)
    moment_scale = np.maximum(np.median(moments, axis=0), 1e-30)
    cs = coefficients
    drives = np.concatenate([r['concentration']-(cs[0]+cs[1]*r['temperature']+cs[2]*r['temperature']**2) for r in rows])
    drive_scale = max(float(np.median(np.abs(drives))), 1e-4)
    state, action, dq = [], [], []
    tensors = [torch.tensor(x, dtype=torch.float64) for x in (count_scale, moment_scale, pivots, pm, coefficients)]
    for row in rows:
        nn, cc, tt = [torch.as_tensor(row[k], dtype=torch.float64)[None] for k in ('counts', 'concentration', 'temperature')]
        total = torch.tensor([float(row['initial_total_solute'])], dtype=torch.float64)
        state.append(state_features(nn, cc, tt, total, *tensors, drive_scale)[0].numpy())
        action.append(action_features(cc[:, :-1], tt[:, :-1], tt[:, 1:], tensors[-1], drive_scale)[0].numpy())
        q = np.log1p(row['counts']/count_scale)
        dq.append(np.diff(q, axis=0))
    state = np.concatenate(state); action = np.concatenate(action)
    state_floors = np.r_[np.full(200, .05), .001, 1., .1, np.full(4, .05), .005, .01, .02]
    action_floors = np.array([1., .02, .1, .1, .01, .001])
    return dict(count_scale=count_scale.tolist(), moment_scale=moment_scale.tolist(), drive_scale=drive_scale,
                state_mean=state.mean(0).tolist(), state_std=np.maximum(state.std(0), state_floors).tolist(),
                action_mean=action.mean(0).tolist(), action_std=np.maximum(action.std(0), action_floors).tolist(),
                delta_q_scale=np.maximum(np.sqrt((np.concatenate(dq)**2).mean(0)), .02).tolist(),
                concentration_scale=max(float(state[:, 200].std()), .001),
                train_parent_ids=[r['id'] for r in rows])


def stack_samples(samples, device='cpu'):
    return {key:torch.stack([s[key] for s in samples]).to(device) for key in samples[0]}


class SampleFactory:
    def __init__(self, rows, history=10, time_edges=(0, 10, 60, 120, 180)):
        self.rows, self.history, self.time_edges = rows, history, time_edges
        self.paired = [(row, [b for b in row['groups'] if len(b['offsets_c']) > 1]) for row in rows]
        self.paired = [(row, groups) for row, groups in self.paired if groups]

    def history_at(self, row, origin):
        indices = np.maximum(np.arange(origin-self.history+1, origin+1), 0)
        return {key:torch.tensor(row[source][indices].copy(), dtype=torch.float64)
                for key,source in [('n', 'counts'), ('c', 'concentration'), ('t', 'temperature')]}

    def window(self, row, origin, horizon):
        if not 0 <= origin < 180 or not 1 <= horizon <= 180-origin:
            raise ValueError('Invalid window origin or horizon.')
        out = self.history_at(row, origin)
        out.update(total=torch.tensor(float(row['initial_total_solute']), dtype=torch.float64),
                   future_t=torch.tensor(row['temperature'][origin+1:origin+horizon+1].copy(), dtype=torch.float64),
                   target_n=torch.tensor(row['counts'][origin+1:origin+horizon+1].copy(), dtype=torch.float64),
                   target_c=torch.tensor(row['concentration'][origin+1:origin+horizon+1].copy(), dtype=torch.float64))
        return out

    def sample(self, batch_size, horizon, rng, device='cpu'):
        samples = []
        bands = [(a, min(b, 181-horizon)) for a,b in zip(self.time_edges[:-1], self.time_edges[1:])
                 if a < min(b, 181-horizon)]
        for _ in range(batch_size):
            row = self.rows[int(rng.integers(len(self.rows)))]
            a,b = bands[int(rng.integers(len(bands)))]
            samples.append(self.window(row, int(rng.integers(a,b)), horizon))
        return stack_samples(samples, device)

    def pair(self, row, branch, horizon, device='cpu'):
        h = min(horizon, branch['horizon']); size = len(branch['offsets_c'])
        history = self.history_at(row, branch['origin'])
        return dict(**{k:v[None].repeat(size, *([1]*v.ndim)).to(device) for k,v in history.items()},
                    total=torch.full((size,), float(row['initial_total_solute']), dtype=torch.float64, device=device),
                    future_t=torch.tensor(branch['temperature'][:, 1:h+1].copy(), dtype=torch.float64, device=device),
                    target_n=torch.tensor(branch['counts'][:, 1:h+1].copy(), dtype=torch.float64, device=device),
                    target_c=torch.tensor(branch['concentration'][:, 1:h+1].copy(), dtype=torch.float64, device=device))

    def sample_pairs(self, groups, horizon, rng, device='cpu'):
        if groups and not self.paired:
            raise ValueError('No nonzero temperature branches are available; response training cannot be skipped silently.')
        samples = []
        for _ in range(groups):
            row, branches = self.paired[int(rng.integers(len(self.paired)))]
            b = branches[int(rng.integers(len(branches)))]
            samples.append(self.pair(row, b, horizon, device))
        return samples
