# ProcGNN: Physics-Informed Graph Neural Networks for Chemical Process Surrogate Modeling

ProcGNN is a topology-aware framework for predicting stream properties in chemical-process flowsheets, developed at [Network Science Lab @ CUK](https://nslab-cuk.github.io/). Graph message passing, attention, and graph-level aggregation are implemented directly in PyTorch.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](requirements_34.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)](requirements_34.txt)
[![Last commit](https://img.shields.io/github/last-commit/NSLab-CUK/ProcGNN)](https://github.com/NSLab-CUK/ProcGNN/commits/main)
[![Issues](https://img.shields.io/github/issues/NSLab-CUK/ProcGNN)](https://github.com/NSLab-CUK/ProcGNN/issues)
[![Stars](https://img.shields.io/github/stars/NSLab-CUK/ProcGNN)](https://github.com/NSLab-CUK/ProcGNN/stargazers)

## 1. Overview

We aim to build a process-graph surrogate that learns stream-level behavior from both flowsheet connectivity and operating conditions. ProcGNN represents unit operations as nodes and material streams as directed edges. Node and stream embeddings encode equipment roles, operating variables, and stream context, while known feed conditions provide additional conditioning for prediction.

The encoder uses relational bidirectional message passing to capture upstream and downstream interactions through separate attention, update, and directional-fusion modules. A Set2Set readout supplies graph-level context. The stream decoder combines source-node, destination-node, stream, graph, and feed representations, then predicts operating conditions, composition, and mass flow through dedicated output branches. The configured heat-exchanger relation module incorporates paired-stream metadata before stream decoding.

Training combines stream-property supervision with physics-informed regularization for mass, component, and atom conservation. The repository also provides cross-process transfer experiments, sensitivity analyses, SHAP-based interpretation, and an economic module that uses the trained surrogate for constrained GA screening.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5558ea5c-fa0b-47ac-b69a-9c46c9f591e9" width="1000" alt="ProcGNN architecture: process graph embeddings, bidirectional message passing, Set2Set readout, stream-property prediction heads, and physics-informed training objectives" />
</p>
<p align="center"><em>Overall architecture of ProcGNN.</em></p>

The supplied diagram illustrates a four-layer encoder. The current reproduction preset uses five layers; encoder depth is configurable in the model YAML.

### Predicted stream properties

Each stream prediction contains ten properties: temperature, pressure, the mole fractions of H2O, H2, CH4, CO2, CO, O2, and N2, and mass flow rate. The composition branch uses softmax; the mass-flow branch uses a transformed prediction space. Volume flow is excluded from the final ten-property prediction schema.

## 2. Reproducibility

### Datasets

The experiment protocol covers ten steam-methane-reforming flowsheets (P01–P10). Training, validation, and test membership is defined by canonical grouped split manifests, with five folds. Preprocessing must be fitted on training data only, and paired methods must use the same partitions.

Raw simulator-derived datasets and trained checkpoints are managed separately and are not included in this Git repository. Before training or checkpoint-based evaluation, obtain approved local data and follow [data/README.md](data/README.md). The required layout includes:

```text
data/
├── datasets_v3/process_main_merged.csv
├── main_data_Streams/
├── process_specs/raw/
├── reference/v3/
└── splits/
     ├── all_processes_full100k_outer5_grouped_60_20_20/
     └── single_process_full_unseen_60_20_20/
```

### Requirements and environment setup

The reported training environment used Python 3.10, PyTorch 2.5.1, and CUDA 12.1. The core model does not require PyTorch Geometric or DGL. Installable dependencies are listed in [requirements.txt](requirements.txt); [requirements_34.txt](requirements_34.txt) records the original server environment and is not a portable lockfile.

```bash
# Clone the repository
git clone https://github.com/NSLab-CUK/ProcGNN.git
cd ProcGNN

# Create and activate a Python 3.10 environment
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA 12.1 PyTorch build, then the remaining dependencies
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

For other CUDA versions or CPU-only installations, use the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

<details>
<summary>Windows PowerShell setup</summary>

```powershell
git clone https://github.com/NSLab-CUK/ProcGNN.git
Set-Location ProcGNN
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

The training commands below use Bash line continuations. In PowerShell, enter the command on one line or use PowerShell line-continuation syntax.

</details>

### How to train ProcGNN

Run commands from the repository root with the environment activated and `PYTHONPATH=src`.

The canonical model preset is [model_260805_10d_frac1.yaml](configs/experiment/pinn/model_260805_10d_frac1.yaml). Experiment families, folds, and comparison settings are defined in [final_protocol_260811.yaml](configs/final_experiments/final_protocol_260811.yaml).

Print the full experiment command plan without starting training. This writes a plan JSON, but does not execute model runs:

```bash
python scripts/run_final_experiments.py --families all \
   --plan-json outputs/reproduction/command_plan.json
```

Train one joint model across all ten flowsheets for each of the five folds:

```bash
python scripts/run_process_kfold_experiments.py \
   --base-config configs/experiment/pinn/model_260805_10d_frac1.yaml \
   --splits-dir data/splits/all_processes_full100k_outer5_grouped_60_20_20 \
   --merged-csv data/datasets_v3/process_main_merged.csv \
   --process-ids 1 2 3 4 5 6 7 8 9 10 \
   --joint-all-processes --only-folds 1 2 3 4 5 \
   --max-epochs 30 --device cuda \
   --output-root outputs/reproduction/proposed_joint
```

Use a new output root for each reproduction run. Do not overwrite completed experiments.

### Hyperparameters

The following values describe the checked-in reproduction preset, not every historical experiment:

| Setting | Default |
|---|---|
| Encoder | Relational bidirectional GNN |
| Number of GNN layers | 5 |
| Hidden dimension | 384 |
| Graph readout | Set2Set, 3 processing steps |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Initial learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | One process graph |
| Maximum epochs | 30 |
| Gradient-clipping norm | 0.5 |

Model and loss hyperparameters are configured through YAML. Common runner options include `--base-config`, `--process-ids`, `--only-folds`, `--max-epochs`, `--device`, and `--output-root`. Check the selected runner's `--help` before launching an experiment. Preserve the original configuration and split manifests when reproducing a reported result.

### Evaluation and analyses

The repository includes entry points for single-process and joint training, unseen-process transfer, data-efficiency transfer, baselines, sensitivity analysis, SHAP explanations, and computational-efficiency measurements. See the [experiment catalogue](docs/EXPERIMENT_CATALOG.md) and [script index](scripts/README.md) for the corresponding protocols and commands.

Aggregate already completed results:

```bash
python scripts/aggregate_final_experiments.py \
   --root outputs/0819final --baseline-root outputs/0819final/baselines
```

Compare equivalent stream/property metrics with matching folds and data partitions. Figures explicitly marked as model-based projections must not be presented as measured transfer results.

### Economic optimization

The [economic module](economic%20module%20GA/module/README.md) connects stream predictions to equipment sizing, techno-economic indicators, and constrained GA search:

```bash
cd "economic module GA/module"
python test_gnn_economic_smoke.py
python run_ga_optimization.py \
   --processes all --search-method ga --population 20 --generations 100 \
   --seed 42 --device cuda \
   --output-dir ../../outputs/economic_ga/reproduction
```

These commands require the approved data and trained checkpoint. GA outputs are surrogate-screened candidates, not Aspen-validated optima. Independent simulator recalculation and constraint checks are required before claiming validated improvement. See [ECONOMIC_GA_REPRODUCTION.md](docs/ECONOMIC_GA_REPRODUCTION.md).

### Tests

Run the lightweight model and protocol checks from the repository root:

```bash
python -m pytest -q tests/test_process_surrogate_smoke.py \
   tests/test_final_paper_protocol_260811.py
```

Install `pytest` separately if it is not already available. The model smoke tests use small CPU graphs; the full protocol checks also inspect approved local reference files.

<details>
<summary>Repository layout</summary>

| Path | Contents |
|---|---|
| `src/process_graph/` | Model, data, and experiment implementation |
| `configs/` | Model presets and experiment configurations |
| `scripts/` | Training, evaluation, plotting, and aggregation entry points |
| `docs/` | Model derivations, protocols, and documentation |
| `economic module GA/module/` | GNN-coupled economics, GA, and Aspen handoff |
| `tests/` | Regression and smoke tests |
| `data/` | Approved local datasets and split manifests; not uploaded |
| `outputs/` | Local checkpoints, logs, figures, and tables; not uploaded |

Historical configuration and provenance-script paths are retained because completed artifacts reference them. Existing results, reports, and workbooks are protected. See the [retention policy](docs/REPOSITORY_LAYOUT_260812.md) and [release guide](docs/GITHUB_RELEASE_GUIDE.md).

</details>

## 3. Reference and Documentation

- [Reviewed model equations and implementation](docs/MODEL_260805_10D_FRAC1_equation_flow_reviewed.md)
- [Final-paper experiment protocol](docs/FINAL_PAPER_PROTOCOL_260811.md)
- [Experiment catalogue](docs/EXPERIMENT_CATALOG.md)
- [Documentation index](docs/README.md)

Publication and preprint links will be added when the bibliographic details are finalized.

## 4. Citing ProcGNN

If ProcGNN is useful in your research, please cite the associated paper once its publication details are available. Until then, reference this [source-code repository](https://github.com/NSLab-CUK/ProcGNN) and record the exact commit used for your experiments. A paper-specific BibTeX entry will be added after the title, author list, and publication identifier are confirmed.

## 5. Contributors

Developed at [Network Science Lab @ CUK](https://nslab-cuk.github.io/). Repository maintainer: [JunheeCho3337](https://github.com/JunheeCho3337).

For questions or reproducibility issues, please [open a GitHub issue](https://github.com/NSLab-CUK/ProcGNN/issues).

No software license is currently included in the repository. A license will be added after institutional approval.

