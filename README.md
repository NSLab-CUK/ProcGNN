# ProcGNN

Physics-informed graph neural networks for stream-wise prediction in chemical-process flowsheets, with reproducible training, evaluation, paper figures, and GNN-coupled economic optimisation.
<img width="8955" height="4050" alt="모델 아키텍처 FIg1 최종 drawio" src="https://github.com/user-attachments/assets/78474afd-ff72-4d56-909f-17cd979f095d" />

Repository: [NSLab-CUK/ProcGNN](https://github.com/NSLab-CUK/ProcGNN).

The model predicts ten stream properties in ten SMR flowsheets (P01–P10): `Temp`, `Pres`, `Frac_H2O`, `Frac_H2`, `Frac_CH4`, `Frac_CO2`, `Frac_CO`, `Frac_O2`, `Frac_N2`, and `Mass_Flow`. Volume flow is excluded from prediction, loss, checkpoint selection, and final metrics.

## Directory guide

| Directory/file | Purpose |
|---|---|
| [`src/process_graph/`](src/process_graph) | Reusable model, data, and experiment implementation |
| [`scripts/`](scripts/README.md) | Training, evaluation, aggregation, plotting, and audit entry points |
| [`configs/`](configs/README.md) | Model, data, training, and experiment configurations |
| [`docs/`](docs/README.md) | Model documentation, protocols, and release guidance |
| [`tests/`](tests) | Regression and CPU smoke tests |
| [`economic module GA/module/`](economic%20module%20GA/module/README.md) | Constrained GA and Aspen-validation handoff tools |
| [`data/`](data/README.md) | Local data and split manifests; contents are not uploaded |
| `outputs/` | Local checkpoints, logs, tables, figures, and workbooks; not uploaded |
| `requirements.txt` | Installable dependencies |
| `requirements_34.txt` | Recorded server environment; retained as reproducibility evidence |
| `train.py` | Convenience training entry point |

Historical configuration and plotting-script paths are retained because completed results reference them. All existing outputs are protected, including diagnostic runs and `outputs/cache/`. See the [retention policy](docs/REPOSITORY_LAYOUT_260812.md).

## Setup

The reported GPU experiments used Python 3.10 and PyTorch 2.5.1 with CUDA 12.1. Install the PyTorch wheel appropriate for your machine; the server snapshot is not a lockfile for every platform.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

Windows PowerShell: activate with `.venv\Scripts\Activate.ps1` and set `$env:PYTHONPATH = "src"` instead of `source` and `export`.

Supply approved local data according to [data/README.md](data/README.md). Raw datasets, checkpoints, Aspen files, and generated outputs are excluded from Git; exclusion does not delete local files.

## Reproduce experiments

The canonical Proposed configuration is [`configs/experiment/pinn/model_260805_10d_frac1.yaml`](configs/experiment/pinn/model_260805_10d_frac1.yaml). The final registry is [`configs/final_experiments/final_protocol_260811.yaml`](configs/final_experiments/final_protocol_260811.yaml), with five folds and ten processes. Active YAML and registry settings take precedence over historical model descriptions.

Print the plan without training:

```bash
PYTHONPATH=src python scripts/run_final_experiments.py --families all
```

Train the joint Proposed model:

```bash
PYTHONPATH=src python scripts/run_process_kfold_experiments.py \
  --base-config configs/experiment/pinn/model_260805_10d_frac1.yaml \
  --splits-dir data/splits/all_processes_full100k_outer5_grouped_60_20_20 \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --process-ids 1 2 3 4 5 6 7 8 9 10 --joint-all-processes \
  --only-folds 1 2 3 4 5 --max-epochs 30 --device cuda \
  --output-root outputs/reproduction/proposed_joint
```

Aggregate completed results:

```bash
PYTHONPATH=src python scripts/aggregate_final_experiments.py \
  --root outputs/0819final --baseline-root outputs/0819final/baselines
```

The [experiment catalogue](docs/EXPERIMENT_CATALOG.md) describes families, splits, comparators, and evidence requirements. Multi-server launchers require a separately managed baseline repository; set `BASELINE_REPO` to its local path.

## Economic optimisation and Aspen validation

```bash
cd "economic module GA/module"
python test_gnn_economic_smoke.py
python run_ga_optimization.py \
  --processes all --search-method ga --population 20 --generations 100 \
  --seed 42 --device cuda --output-dir ../../outputs/economic_ga/reproduction
```

GA outputs are surrogate-screened candidates, not Aspen-validated optima. Independent simulator recalculation and constraint checks are required before claiming validated improvement. See [economic reproduction](docs/ECONOMIC_GA_REPRODUCTION.md) and the module README.

## Figures, tests, and cleanup

Figure scripts are retained for result provenance. Model-based projections must not be presented as measured transfer results. Source roots and interpretation rules are in the [experiment catalogue](docs/EXPERIMENT_CATALOG.md).

```bash
PYTHONPATH=src python -m pytest -q tests/test_final_paper_protocol_260811.py
PYTHONPATH=src python -m pytest -q tests/test_process_surrogate_smoke.py
```

The cleanup helper is dry-run by default and never selects data or experiment outputs:

```powershell
.\scripts\cleanup_obsolete_artifacts.ps1
.\scripts\cleanup_obsolete_artifacts.ps1 -Execute
```

Execution moves disposable caches into a timestamped backup beside the project; a manifest records restore paths and checksums. See the [release guide](docs/GITHUB_RELEASE_GUIDE.md).

## Publication policy

- Do not commit raw data, Aspen workbooks, checkpoints, caches, or experiment logs.
- Release approved data, checkpoints, and figures through a versioned archive and cite that version.
- Select an institutionally approved software licence before redistribution; none is currently included.
- Add the approved manuscript citation and `CITATION.cff` when the author list, title, and DOI are final.

