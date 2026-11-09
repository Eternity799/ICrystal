# End-to-end verification

The following workflow was executed on 2026-11-06 with real PBM labels and neural updates.

| Stage | Executed scope | Result |
|---|---|---|
| Data generation | Nine 180-minute PBM trajectories: three each for train, validation and test, with ordinary and terminal paired-action branches | Passed |
| Training | Two optimizer updates using the formal execution path and original architecture, losses and sampling code | Passed |
| Validation | All three validation parents: true-history one-step prediction, recursive prediction and action responses | Passed |
| Continuation | One optimizer update from the newly trained checkpoint, retaining source and dataset identity checks | Passed |
| Test evaluation | Three test parents excluded from training and continuation | Passed |
| B3 search and export | One objective evaluation and an independent PBM replay | Passed |
| Frozen-model comparison | Linear, slow-fast, frozen B3 and paper LSTM-MPC to minute 12 | Passed; two accepted feasible actions and no hold fallbacks |
| New-model comparison | Newly continued checkpoint and exported B3, four policies to minute 12 | Passed |
| Disturbed comparison | New-model/new-B3 comparison with shared execution error, T/C sensor noise and dynamic kinetics | Passed |

Every MPC forecast reaches the endpoint at minute 180. The short control runs test orchestration and action execution; their minute-12 quality values are not final-product results.

Separate test configurations reduce branch counts, training updates and the B3 search budget. All base trajectories retain the real 180-minute duration. The `formal` mode selects the execution path needed to test continuation; the nine-trajectory fixture is not the full paper dataset. The two-update model is used to test stage compatibility, not prediction accuracy.

The checks inspect saved artifacts as well as exit codes: four policy records, 13 minute states, full 200-bin CSD arrays, finite quality values and decision/action counters. The frozen-model nominal test requires at least one accepted feasible action; a comparison containing only hold fallbacks fails.

A separate search completed eight full-batch PBM objective evaluations and one independent replay, exercising finite-difference optimization and feasible-incumbent export. Its baseline was connected to the continued model in a disturbed 12-minute comparison, which executed two feasible actions without hold fallback. This limited search does not establish convergence.

Full-dataset generation, full-budget training and optimization, the complete 180-minute MPC loop, disturbance repeats and GPU timing remain outside this verification scope.

```powershell
python tests/run_e2e.py
```

The summary under `outputs/e2e/` records commands, budgets, exit codes, artifact paths and action counts. Run outputs and logs are excluded from the source ZIP.
