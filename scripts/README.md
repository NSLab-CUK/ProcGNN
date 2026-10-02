# Script index

Run scripts from the repository root with `PYTHONPATH=src`. Consult `--help` before launching expensive training or evaluation.

| Task | Entry point |
|---|---|
| Print the final experiment plan | `run_final_experiments.py` |
| Train the process surrogate | `train_process_surrogate.py` |
| Process/joint cross-validation | `run_process_kfold_experiments.py` |
| Single-process unseen transfer | `run_single_process_full_unseen_experiments.py` |
| Transfer data efficiency | `run_transfer_data_efficiency_experiments.py` |
| Final baselines | `run_final_baseline_experiments.py` |
| Final sensitivity analyses | `run_final_sensitivity_experiments.py` |
| Final SHAP analyses | `run_final_shap_experiments.py` |
| Aggregate completed experiments | `aggregate_final_experiments.py` |
| Aggregate computational efficiency | `aggregate_final_efficiency.py` |
| Preview/recoverably remove disposable caches | `cleanup_obsolete_artifacts.ps1` |

`plot_*`, `build_*`, `summarize_*`, and audit scripts are retained for reproducing already generated figures and tables. Paths in artifact READMEs/manifests take precedence over this entry-point index. In particular, `plot_transfer_0_10_model_based_projection.py` produces a model-based projection, not measured transfer data.

`run_final_protocol_*.sh` and `run_final_phase*.sh` are launchers for the original multi-server setup; inspect device assignments, data paths, and `BASELINE_REPO` first.

The cleanup helper protects the entire `outputs/` and `data/` trees, archived runs, checkpoints, workbooks, and reports. It defaults to a preview; execution moves candidates to a sibling backup with a restore manifest. The optional `-IncludeLegacyHelpers` selects a small, explicit list of retired helpers and redundant draft documents, not generated results.
