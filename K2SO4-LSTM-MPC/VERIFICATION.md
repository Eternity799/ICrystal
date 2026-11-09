# Verification

Verified on 2026-11-06 using the CPU environment recorded in `environment_verified.json`.

## Numerical and interface checks

| Check | Result |
|---|---|
| Python compilation | All 52 files passed |
| Contract tests | All 11 passed |
| Checkpoint export | All 24 state tensors identical to the final paper checkpoint |
| Inference parity | One-step outputs and three-step recursive CSD, concentration and validity identical to the original implementation on the supplied training-history example |
| PBM parity | Parsed configuration, initialization and two minute advances identical to the original counts and concentrations |
| Controller parity | Rate matrices, candidate temperatures, quality, constraint margins, objective components and feasibility matched at horizons 1, 3, 10 and 170 using a deterministic forecast fixture |
| Disturbance parity | Sensor, bounded execution-error and dynamic-kinetics sequences identical to the paper functions for the checked seed |
| Data-generation plan | Smoke parent-trajectory plan matched the original |
| Source preservation | Executable numerical syntax unchanged after comment/docstring editing and import relocation; 46 mapped original source/configuration hashes unchanged |
| Relocation | An external temporary copy passed input checks, CPU prediction and workflow help commands with `PYTHONPATH` cleared |

The contract tests cover manifest tampering, objective and quality conventions, frozen model loading, inventory closure, random streams, dataset status compatibility, invalid predictions, completed-model selection, active-run rejection, B3 export/reload and JSON serialization of paths.

Source and documentation scans exclude generated outputs and caches. The escaped legacy dataset-status constant remains for compatibility.

## Workflow execution

Real PBM generation, neural updates, validation/test prediction, B3 export/replay and four-policy control were exercised with reduced budgets. See `END_TO_END_VERIFICATION.md` for the stage counts. A separate B3 test completed eight full-batch objective evaluations and one independent replay, including finite-difference evaluations, and exported a feasible incumbent.

The model checkpoint contains unchanged numeric tensors, scaler and grid with portable metadata. It is an inference checkpoint, rather than an optimizer-resume checkpoint; both original and exported identities are recorded in `assets/model/provenance.json`. Source and architecture checks remain enabled.

## Scope

These checks establish numerical parity for the tested cases and execution of the reduced-budget workflow. They do not establish exact reproduction of all paper results. The full 1800-trajectory dataset, full-budget training, full-budget B3 search, repeated-disturbance suite and complete 180-minute MPC loop have not been rerun with this package. No GPU performance benchmark was performed.

```powershell
python run.py check
python -m unittest discover -s tests -v
python tests/run_e2e.py
```
