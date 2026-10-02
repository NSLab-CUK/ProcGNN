# Configuration index

Start with these two files:

- [`experiment/pinn/model_260805_10d_frac1.yaml`](experiment/pinn/model_260805_10d_frac1.yaml): canonical Proposed-model configuration used by the final registry.
- [`final_experiments/final_protocol_260811.yaml`](final_experiments/final_protocol_260811.yaml): final experiment families, folds, comparators, and validation monitor.

| Subdirectory | Contents |
|---|---|
| `model/` | Reusable encoder and decoder defaults |
| `data/` | Process/data specification defaults |
| `train/` | Training defaults |
| `overrides/` | Explicit experiment overrides |
| `experiment/` | Versioned model, ablation, and sensitivity configurations |
| `final_experiments/` | Final-paper experiment registry |

Historical YAML files are retained at their original paths: completed checkpoints and result metadata can reference them. A dated file is not necessarily the active configuration. Resolve the selected runner's `--base-config` and includes before interpreting settings.
