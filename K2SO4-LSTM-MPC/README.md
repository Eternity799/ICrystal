# K2SO4 full-CSD LSTM model predictive control

Reference implementation for seeded potassium sulfate-water batch cooling crystallization: population balance modelling, dataset generation, full-CSD LSTM training and prediction, constrained MPC, and offline PBM cooling optimization (B3). The study uses simulated data and a PBM plant.

## Installation and checks

Use Python 3.10 or newer. The PBM runs on the CPU; PyTorch inference and training support CPU or CUDA. CuPy is not required.

```powershell
python -m pip install -r requirements.txt
python run.py check
python run.py predict --device cpu
python run.py simulate --minutes 2
```

`check` verifies source and asset hashes and the frozen baseline settings. `predict` runs three recursive steps from the supplied training-history example; `simulate` advances the PBM for two minutes. These examples check the interfaces, not independent prediction accuracy.

Computations save results in UTC-named directories under `outputs/`. Keep source files and configurations fixed during a run. A change to a frozen file invalidates its recorded hash and requires a new version.

## Code layout

| Location | Contents |
|---|---|
| `run.py` | Command-line entry point |
| `release_runtime.py` | Input validation, model loading, policy comparisons and numerical exports |
| `physics/` | PBM configuration, grid, kinetics, population operators, integration and metrics |
| `control/` | Terminal objective, SLSQP controller, action selection, B3 parameterization and disturbances |
| `surrogate/core/` | Features, LSTM, losses, sampling and inference |
| `surrogate/core/mpc/` | Control-oriented training and action-response supervision |
| `surrogate/continuation/` | Continuation training |
| `data_generation/` | Base trajectories, paired action branches and data-quality checks |
| `configs/` | Physical, control, B3 and disturbance settings |
| `assets/` | Final weights, two frozen B3 baselines, scoring domain and example history |

## Surrogate model

The input comprises ten historical states with 210 features per state, plus six features describing the next candidate temperature action. Each state includes a 200-bin absolute number CSD, concentration and temperature. A 128-dimensional embedding and a 128-unit LSTM encode the history. A separate 128-dimensional current-state/action path is combined with the history representation in a 256-dimensional fusion layer.

The output head supplies 200 nonnegative additions and 200 retention ratios. These update the current CSD; an inventory projection limits crystal mass to available solute. Concentration is recovered from the fixed total inventory. D10, D50, D90, Span and Yield are calculated from the predicted state.

Recursive prediction appends the predicted CSD and concentration and the candidate temperature to the rolling window. Future PBM states and disturbance realizations are unavailable to the surrogate.

## Control comparison

```powershell
python run.py compare --mode smoke --case nominal --device cpu
python run.py compare --mode run --case nominal --target main --device auto
python run.py compare --mode run --case combined --target target700 --seed 20261020 --device auto
```

The four policies are LSTM-MPC, linear cooling, slow-fast empirical cooling and frozen B3. They share the PBM, inventory, duration, metrics and disturbance arrays. After ten minutes of common linear startup, MPC acts every minute and forecasts to the batch endpoint at minute 180. The decision uses five rate blocks rather than 170 independent temperature variables.

The `main` D50 target is 897.7452105387908 micrometres; `target700` uses a separate frozen 700-micrometre B3. The objective includes Span minimization and its upper-limit penalty, with a Yield tracking weight of 0.01. The command-temperature and rate bounds, supersaturation floor of 0.99, 30-second cooperative solve budget and 60-second action-selection fallback follow the paper settings. Execution errors can move actual temperature outside command bounds.

Smoke mode stops the plant at minute 12 and permits one SLSQP iteration per decision, while each forecast still reaches minute 180. A smoke result therefore does not compare final product quality. Full runs must be requested with `--mode run`.

Disturbance amplitudes are set in `configs/disturbance.yaml`. T/C sensor noise uses `SeedSequence(seed).spawn(2)`. Execution error uses `SeedSequence([seed, 719])`, an AR(1) process with pre-clip standard deviation A/2 and bounds +/-A. Dynamic growth and nucleation factors use `SeedSequence([seed, 4545])` with independent minute-wise uniform draws in [0.55, 1.45]. Factors multiply nominal coefficients; both nucleation coefficients share the nucleation factor. Every policy receives the same realizations.

Comparisons save configurations, commands, plant trajectories, full CSD arrays, disturbances, decisions and failures. Hardware and wall-clock solve limits can affect the chosen actions. The comparison command starts a new run; it does not resume a partial batch.

To select a newly trained model and a newly optimized baseline:

```powershell
python run.py compare --mode smoke --model-run "PATH_TO_COMPLETED_MODEL_RUN" --b3-run "PATH_TO_B3_RUN" --device cpu
```

Model runs must be complete, with compatible source, grid and objective settings. Active training locks are rejected. A D50 target override is allowed while the remaining frozen protocol fields stay matched. B3 file hashes, physical configuration, numerical sources and control settings are checked before replay.

## Offline B3 search

```powershell
python run.py optimize-baseline --target main --max-evaluations 300
```

The default parameterization fixes the startup temperature at minute 10 and optimizes six temperatures at minutes 30, 60, 90, 120, 150 and 180. SLSQP starts from linear, empirical and fast-slow profiles and uses the nominal PBM and matched objective and constraints. Results are best-found feasible incumbents within the recorded budget. New searches are saved separately from the bundled baselines.

## Dataset generation and training

The complete 1800-trajectory dataset is distributed separately. Its base-trajectory split is 1260 train, 270 validation and 270 test. Action branches inherit their parent's role. Labels use the actual temperature applied to the PBM.

```powershell
python run.py generate --mode smoke --limit 1
python run.py generate --mode formal --limit 0
python run.py train --mode formal --arm replay_csd --dataset "PATH_TO_COMPLETE_DATASET" --no-plot
python run.py evaluate --run "PATH_TO_MODEL_RUN" --role validation --no-plot
```

Replace the example paths with actual directories. Formal generation is computationally expensive. To resume generation with the same source and configuration, use `python run.py generate --mode formal --resume "PATH_TO_GENERATION_RUN" --limit 0`.

Readers accept schema-1 records with either the original completion status or the English `dataset_complete_not_trained` status. Array keys, units, parent roles and checksums are retained. Partial datasets and incompatible configurations are rejected.

The paper arm, `replay_csd`, uses number- and mass-weighted relative L1, log, change-scaled, concentration and inventory losses, together with same-history action-response supervision. `control_priority` provides the matched ablation.

The final model followed from-scratch training and continuations of 1000 and 3000 updates; the selected epoch in the latter continuation was 58. The recipes are `surrogate/mpc_continuation.yaml` and `surrogate/mpc_continuation_3h.yaml`. Set `parent_run` to a compatible completed training run, then use:

```powershell
python run.py continue-training --config surrogate/mpc_continuation.yaml --mode formal
```

`assets/model/best.pt` contains inference weights, the scaler and grid. It excludes optimizer state and cannot resume training. Retain full training-run metadata for continuation. Use `--role test` for final evaluation after model selection.

## Verification

```powershell
python -m unittest discover -s tests -v
python tests/run_e2e.py
```

The end-to-end test generates nine real 180-minute PBM trajectories with reduced branch budgets, performs two training updates and one continuation update, evaluates validation/test predictions, exports a limited-budget B3 result, and runs three 12-minute four-policy comparisons. It takes several minutes on the verified CPU environment. Test configurations and outputs are saved under `outputs/e2e/`; paper defaults are retained. An interrupted test can resume with `--resume "PATH_TO_E2E_RUN"`.

Numerical parity, relocation and reduced-budget workflow checks are described in `VERIFICATION.md` and `END_TO_END_VERIFICATION.md`. Full-scale training and a complete 180-minute closed-loop experiment have not been repeated with this package.

## Units and provenance

- CSD: absolute particles per kilogram of water per bin.
- Concentration: kilograms of dissolved K2SO4 per kilogram of water.
- Length: metres internally; D-values in micrometres.
- Time: seconds in the PBM, one-minute data/control intervals, 180-minute batches.
- Yield: `(C0 - C)/C0`, excluding initial seed mass from newly crystallized product.
- Volume-based D-values follow the logarithmic-bin convention; Span is `(D90-D10)/D50`.

`DATA.md` describes the bundled arrays. Figure source data and full run archives are separate research-data assets. `SOURCE_MAP.json` records original project-relative sources; `release_manifest.json` records package file hashes. `assets/model/provenance.json` records the original and exported checkpoint identities and tensor equality. A public reuse license must be added before distribution; see `LICENSE_NOTICE.md`.
