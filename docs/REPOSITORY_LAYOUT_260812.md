# Repository layout and retention policy (260812)

## Canonical paths

| Purpose | Path |
|---|---|
| Final model config | `configs/experiment/pinn/model_260805_10d_frac1.yaml` |
| Reviewed final model document | `docs/MODEL_260805_10D_FRAC1_equation_flow_reviewed.md` |
| Final experiment protocol | `configs/final_experiments/final_protocol_260811.yaml` |
| Training entry point | `scripts/train_process_surrogate.py` |
| All-process runner | `scripts/run_process_kfold_experiments.py` |
| Unseen-process runner | `scripts/run_single_process_full_unseen_experiments.py` |
| Transfer data-efficiency runner | `scripts/run_transfer_data_efficiency_experiments.py` |
| Historical alternate model document | `docs/MODEL_260811_NOVOL_FAST.md` |
| Dataset and split manifests | `data/` |
| Reusable normalizer/scaler cache | `outputs/cache/` |
| Experiment artifacts | `outputs/<date>_<scope>_<model>_<purpose>/` |

## Output naming rule

New production output roots use one format:

```text
YYYYMMDD_<scope>_<model>_<purpose>
```

Examples:

```text
20260812_allproc_m260811_main
20260812_unseen_m260811_transfer
20260812_sensitivity_m260811_depth
```

Temporary runs must contain one of `smoke`, `debug`, `diag`, `profile`, `perf`, or `probe` in the
top-level output name. This identifies their purpose; it does not authorise deleting them.

## Retention policy

Keep (updated for the 2026-10-02 migration):

- production checkpoints and their matching metric/config artifacts;
- final paper aggregation outputs;
- split manifests and source datasets;
- `outputs/cache`, because it reduces training startup time;
- the canonical final config and runner scripts.
- **all** existing `outputs/`, including smoke, debug, diagnostic, profiling, and cache outputs;
- Aspen workbooks, reports, and figures wherever they are stored;
- historical configurations and provenance scripts referenced by completed runs.

Remove from the active directory only after verification, with a recoverable backup:

- Python and pytest caches;
- exact duplicate documents and a reviewed list of retired one-off helpers;
- empty temporary directories and the known invalid one-file path mirror.

Existing `archive/`, logs, and any generated research results are protected. No output-name pattern is sufficient to decide that an artifact is disposable.

Historical YAML files remain in place because they occupy little space and are referenced by older result
metadata and documentation. Moving or renaming them would not speed up training and would break those links.
Use `scripts/cleanup_obsolete_artifacts.ps1` to preview the current cleanup set. `-Execute` moves the selected paths into a timestamped sibling backup and records a restore manifest and file checksums. `-IncludeLegacyHelpers` opts into the explicit reviewed helper/draft list. Nothing is permanently deleted.
