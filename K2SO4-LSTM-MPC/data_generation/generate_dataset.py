"""Generate a nominal full-CSD dataset."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import yaml
if __package__:
    from .dataset import BASE, ROLES, generate_parent, load_pbm, make_plan, resolve, sha256, validate_config, write_json
else:
    from dataset import BASE, ROLES, generate_parent, load_pbm, make_plan, resolve, sha256, validate_config, write_json
if __package__:
    from .dataset_design import planned_steps
else:
    from dataset_design import planned_steps
if __package__:
    from .dataset_quality import numerical_check, numerical_parent_ids, write_coverage
else:
    from dataset_quality import numerical_check, numerical_parent_ids, write_coverage

def report(out, manifest):
    rows = manifest['completed']
    expected = manifest['plan']
    groups = [g for row in rows for g in row['groups']]
    summary = dict(status=manifest['status'], mode=manifest['mode'], completed_parents=len(rows), planned_parents=len(expected), counts_by_role={r: sum((x['role'] == r for x in rows)) for r in ROLES}, groups=len(groups), branches=sum((len(g['offsets_c']) for g in groups)), skipped_branches=sum((len(x['skipped_branches']) for x in rows)), baseline_only_groups=sum((len(g['offsets_c']) == 1 for g in groups)), one_sided_groups=sum((any((v > 0 for v in g['offsets_c'])) != any((v < 0 for v in g['offsets_c'])) for g in groups)), terminal_groups=sum((g.get('terminal', False) for g in groups)), groups_by_pattern={p: sum((g.get('pattern', 'offset') == p for g in groups)) for p in sorted({g.get('pattern', 'offset') for g in groups})}, pbm_steps=sum((x.get('pbm_steps_executed', 180 + sum((g['horizon_steps'] * len(g['offsets_c']) for g in x['groups']))) for x in rows)), trajectory_resamples=sum((x.get('trajectory_attempts', 1) - 1 for x in rows)), extinction_branches=sum(('complete dissolution' in item['reason'] for x in rows for item in x['skipped_branches'])), coverage=[dict(family=f, role=r, count=sum((x['family'] == f and x['role'] == r for x in rows))) for f in manifest['families'] for r in ROLES], max_relative_mass_error=max([a['max_relative_mass_error'] for x in rows for a in [x['audit']] + [a for g in x['groups'] for a in g['audits']]], default=None), max_reproduction_relative_l1=max([g['reproduction_relative_l1'] for g in groups], default=None), committed_compute_seconds=sum((x['elapsed_seconds'] for x in rows)), numerical_check_count=len(manifest.get('numerical_checks', [])), numerical_check_seconds=sum((x['elapsed_seconds'] for x in manifest.get('numerical_checks', []))), note='Counts and conservation checks do not establish numerical convergence, sufficient coverage or surrogate accuracy. Independent test trajectories do not participate in training.')
    write_coverage(out, manifest)
    write_json(out / 'summary.json', summary)
    lines = ['# Full-CSD dataset generation report', '', f"Status: {manifest['status']}", f"Parent trajectories: {len(rows)}/{len(expected)}; branch groups: {len(groups)}; valid branches: {summary['branches']}", '', '| Family | Role | Completed parents |', '|---|---|---:|']
    lines += [f"| {x['family']} | {x['role']} | {x['count']} |" for x in summary['coverage']]
    lines += ['', '## Dataset definition', '', '- Each parent has 181 states from 0 to 180 minutes, with absolute counts in all 200 size bins.', '- Assign train/validation/test roles before generation; every action branch inherits its parent role.', '- Kinetics are nominal and measurement noise is absent. Temperatures are actual, interval-linear PBM inputs, not commands with hidden actuation error.', '- Do not discard samples solely because the saturation ratio is below 1. Local reheating uses the existing dissolution model and its engineering assumptions.', '- The training domain requires positive crystal number/mass. Resample extinct parent plans and separately record/exclude extinct branches; do not classify them as PBM numerical failures.', f"- Parent-plan resamples after complete dissolution: {summary['trajectory_resamples']}; extinct branches: {summary['extinction_branches']}.", '- Size, initial-state and kinetic definitions are stored in configuration snapshots and the manifest. Generation alone provides no training or closed-loop conclusion.', f"- Skipped infeasible branches: {summary['skipped_branches']}; baseline-only groups: {summary['baseline_only_groups']}.", '- Branches within a group are correlated; sampling should prevent them from dominating parent trajectories.', '- This report audits generation completeness, physical validity and replay. Coverage sufficiency and learning curves require separate evaluation.', '', '## Actions and quality checks', '', '- Revision-2 offsets_c gives the first-step temperature difference from the parent; plan_pattern and duration_steps define subsequent actions.', '- offset shifts future temperatures; rate_block changes rates for a finite duration; change_then_hold changes the first temperature and then holds. The latter two resume parent rates afterward.', '- Zero offset is always the exact parent path, not a zero-amplitude hold. Save each full actual-temperature branch plan.', '- Horizons are usually 10 steps, with selected 30-step and periodic terminal branches. Future true states are not fed into neural rollout.', f"- Terminal branch groups: {summary['terminal_groups']}; action patterns: {summary['groups_by_pattern']}.", '- coverage_features.csv / coverage_histograms.csv: per-role state ranges, means and occupancy.', '- saturation_action_coverage.csv: separate saturation-rate occupancy for parent intervals and first branch actions.', '- branch_plans.csv / response_strength.csv: horizons, patterns and full-CSD response strengths, not independent-sample counts.', '- first_bin_number_fraction / last_bin_mass_fraction measure boundary occupancy, not boundary flux or support sufficiency.', f"- Completed numerical checks: {summary['numerical_check_count']}; additional time: {summary['numerical_check_seconds'] / 60:.1f} minutes.", '- *_numerical.csv/json: minute-wise differences after time-step halving and grid refinement under the same temperature path; no automatic convergence claim.', '- Candidate plans can later be scored using a common MPC objective; generation embeds no fixed objective weights and does not change training losses.', '', '## Next steps', '', 'Inspect smoke coverage, responses and numerical differences before a pilot dataset; review labels and learning behavior before formal generation.', 'Use the release training entry point with --dataset set explicitly to the completed output directory.', 'Pilot data cannot be labelled formal, and partial datasets cannot be used for training. Changed configuration or source requires a new dataset directory.']
    (out / 'dataset_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary

def main():
    parser = argparse.ArgumentParser(description='Full-CSD data generation from a noise-free nominal PBM; no neural training.')
    parser.add_argument('--config', type=Path, default=BASE / 'dataset.yaml')
    parser.add_argument('--mode', choices=['smoke', 'pilot', 'formal'], default='smoke')
    parser.add_argument('--resume', type=Path, help='Resume an existing output directory; configuration and source must match exactly.')
    parser.add_argument('--limit', type=int, default=0, help='Maximum new parent trajectories this invocation; 0 completes the plan.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must not be negative.')
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    cfg['_base'] = str(config_path.parent)
    validate_config(cfg)
    api = load_pbm(cfg)
    base = api['load_config'](api['config_path'])
    if base.grid.bins != 200:
        raise ValueError('This interface requires a 200-bin PBM; the CSD dimension is not changed silently.')
    if not base.seed.min_um <= cfg['initial']['seed_d50_um'][0] <= cfg['initial']['seed_d50_um'][1] <= base.seed.max_um:
        raise ValueError('The sampled seed D50 range lies outside the seed support.')
    if cfg.get('design_revision', 1) == 2 and cfg['initial']['nominal_every']:
        if not cfg['coverage']['temperature_c'][0] <= base.batch.start_temperature_c <= cfg['coverage']['temperature_c'][1]:
            raise ValueError('The nominal initial temperature is outside coverage; adjust the range or disable nominal anchors.')
    fingerprint = dict(config=sha256(config_path), pbm_config=sha256(api['config_path']), code={str(p.relative_to(api['source'])): sha256(p) for p in sorted(api['source'].rglob('*.py'))}, generator={p.name: sha256(p) for p in (BASE / 'dataset.py', BASE / 'generate_dataset.py', BASE / 'dataset_design.py', BASE / 'dataset_quality.py')})
    plan = make_plan(cfg, args.mode)
    numerical_ids = numerical_parent_ids(cfg, args.mode, plan)
    if args.resume:
        out = args.resume.resolve()
        if not (out / 'manifest.json').is_file():
            raise ValueError('The resume directory has no manifest.json.')
    else:
        out = resolve(cfg, 'output_root') / args.mode / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_utc')
        out.mkdir(parents=True, exist_ok=False)
    lock = out / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as stream:
        import os
        stream.write(f'pid={os.getpid()}\n')
    manifest = None
    try:
        if args.resume:
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            if manifest['fingerprint'] != fingerprint or manifest['mode'] != args.mode or manifest['plan'] != plan:
                manifest = None
                raise ValueError('Source, configuration, mode or plan changed; start a new run rather than mixing incompatible data.')
            for row in manifest['completed']:
                for item in row['files']:
                    if sha256(out / item['file']) != item['sha256']:
                        raise ValueError(f"Committed-file checksum failed: {item['file']}")
            for check in manifest.get('numerical_checks', []):
                for item in check['files']:
                    if sha256(out / item['file']) != item['sha256']:
                        raise ValueError(f"Numerical-check file checksum failed: {item['file']}")
        else:
            shutil.copyfile(config_path, out / 'dataset_config_snapshot.yaml')
            shutil.copyfile(api['config_path'], out / 'pbm_config_snapshot.yaml')
            source_snapshot = out / 'source_snapshot'
            source_snapshot.mkdir()
            for relative in fingerprint['code']:
                dest = source_snapshot / 'pbm' / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(api['source'] / relative, dest)
            for name in fingerprint['generator']:
                shutil.copyfile(BASE / name, source_snapshot / name)
            manifest = dict(schema_version=1, status='generating', mode=args.mode, families=cfg['families'], design_revision=cfg.get('design_revision', 1), fingerprint=fingerprint, resolved_config=cfg, plan=plan, completed=[], numerical_plan=numerical_ids, numerical_checks=[])
        manifest['status'] = 'generating'
        manifest.pop('error', None)
        write_json(out / 'manifest.json', manifest)
        done = {x['id'] for x in manifest['completed']}
        todo = [row for row in plan if row['id'] not in done]
        if args.limit:
            todo = todo[:args.limit]
        work = {entry['id']: planned_steps(cfg, entry) for entry in plan}
        print(f'Output: {out}\nPlanned parents: {len(plan)}; completed: {len(done)}; this invocation: {len(todo)}.\nPlanned candidate PBM minute-steps: {sum(work.values())} (excluding parent resampling/numerical checks); infeasible branches are skipped, and no training is launched.', flush=True)
        for entry in todo:
            print(f"Generating {entry['id']}: 180-step parent and short branches...", flush=True)
            start = time.perf_counter()
            row = generate_parent(api, base, cfg, entry, out)
            row['elapsed_seconds'] = time.perf_counter() - start
            manifest['completed'].append(row)
            write_json(out / 'manifest.json', manifest)
            completed = manifest['completed']
            executed = sum((x.get('pbm_steps_executed', 180 + sum((g['horizon_steps'] * len(g['offsets_c']) for g in x['groups']))) for x in completed))
            seconds_per_step = sum((x['elapsed_seconds'] for x in completed)) / executed
            completed_ids = {x['id'] for x in completed}
            remaining_steps = sum((v for key, v in work.items() if key not in completed_ids))
            print(f"Committed {len(completed)}/{len(plan)}; this parent: {row['elapsed_seconds'] / 60:.1f} minutes; remaining plan estimated from completed PBM steps: {seconds_per_step * remaining_steps / 60:.1f} minutes (excluding numerical checks).", flush=True)
        audited = {x['parent_id'] for x in manifest.get('numerical_checks', [])}
        for entry in manifest['completed']:
            if entry['id'] not in numerical_ids or entry['id'] in audited:
                continue
            result, names = numerical_check(api, base, cfg, entry, out)
            result['files'] = [dict(file=name, sha256=sha256(out / name)) for name in names]
            manifest.setdefault('numerical_checks', []).append(result)
            write_json(out / 'manifest.json', manifest)
        if len(manifest['completed']) == len(plan) and {x['parent_id'] for x in manifest.get('numerical_checks', [])} != set(numerical_ids):
            raise ValueError('The representative numerical-check plan is incomplete.')
        manifest['status'] = 'dataset_complete_not_trained' if len(manifest['completed']) == len(plan) else 'partial_resumable'
        write_json(out / 'manifest.json', manifest)
        summary = report(out, manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(out)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status='interrupted_committed_parents_resumable', error=f'{type(exc).__name__}: {exc}')
            write_json(out / 'manifest.json', manifest)
            report(out, manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)
if __name__ == '__main__':
    main()
