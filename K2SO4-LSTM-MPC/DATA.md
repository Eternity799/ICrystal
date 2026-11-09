# Data assets

`assets/example_history.npz` contains one real training-role history: `n[10,200]`, `c[10]`, `t[10]`, scalar `initial_total`, `future_t[3]`, `target_n[3,200]`, and `target_c[3]`. It supports an interface/parity check and is not an independent test set.

`assets/b3_main.npz` and `assets/b3_target700.npz` retain the saved nominal PBM baseline arrays. Their companion JSON files contain the corresponding frozen objective and best-found status. The six-dimensional scoring domain is stored separately in `assets/quality_domain.json`.

The complete training/validation/test dataset is not bundled. Readers use schema-1 manifests with hashed NPZ records, fixed 200-bin grids and parent-level roles. Generator output includes each 181-state base trajectory and same-history candidate-action branches. A branch's role cannot differ from its parent. Only explicitly requested roles are opened by the training/evaluation readers.

All figure source data, independent test results, disturbance repeats and full run archives remain separate research-data assets. Do not relabel the small training example as validation or test data, or treat correlated minutes/branches as independent batches.
