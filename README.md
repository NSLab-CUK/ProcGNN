# Physics-Informed Graph Neural Networks for Chemical Process Surrogate Modeling (ProcGNN)

ProcGNN is a process-graph learning framework for stream-property prediction, developed at [Network Science Lab @ CUK](https://nslab-cuk.github.io/) and implemented directly in PyTorch.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](requirements_34.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)](requirements_34.txt)
[![Last commit](https://img.shields.io/github/last-commit/NSLab-CUK/ProcGNN)](https://github.com/NSLab-CUK/ProcGNN/commits/main)
[![Stars](https://img.shields.io/github/stars/NSLab-CUK/ProcGNN)](https://github.com/NSLab-CUK/ProcGNN/stargazers)
[![Issues](https://img.shields.io/github/issues/NSLab-CUK/ProcGNN)](https://github.com/NSLab-CUK/ProcGNN/issues)

## 1. Overview

We present ProcGNN, a physics-informed graph neural network for stream-level chemical process surrogate modeling. Implemented directly in PyTorch, ProcGNN uses shared unit and stream representations to predict temperature, pressure, mass flow rate, and component mole fractions across different process flowsheets.
<p align="center">
  <img src="https://github.com/user-attachments/assets/5558ea5c-fa0b-47ac-b69a-9c46c9f591e9" width="1000" alt="Overall ProcGNN architecture: process graph embeddings, bidirectional GNN layers, Set2Set readout, stream-property prediction heads, and conservation-based training" />
</p>
<p align="center"><em>The overall architecture of ProcGNN.</em></p>
Rigorous process simulation is widely used for process design and optimization, but repeated simulations can be computationally expensive. Conventional surrogate models are usually developed for a fixed process structure and a predefined set of input and output variables. Reusing them becomes difficult when unit operations, material-stream connections, or prediction targets change. ProcGNN addresses this problem by representing unit operations as nodes and material streams as directed edges. A shared prediction function is applied to each stream, allowing the prediction set to follow the streams represented in the supplied flowsheet.

ProcGNN combines three main components. First, it encodes unit-operation features, stream features, and feed conditions using common definitions across flowsheets. Feed-ratio features represent the air-to-methane and water-to-methane relationships, while a relation gate incorporates heat-transfer interactions between related hot- and cold-side streams. After four bidirectional GNN layers, each stream prediction combines its source and destination unit representations, stream features, a Set2Set graph representation, and directly encoded feed conditions. This gives each prediction access to both information from connected units and information from the entire flowsheet.

Second, flow-aware message passing treats the two propagation directions separately. Forward propagation aggregates source-unit information at each destination through incoming streams; backward propagation aggregates destination-unit information at each source through outgoing streams. In both directions, attention is normalized over the outgoing streams of the same source unit, allowing the model to compare branches that lead to different destinations. Differential encoding provides each update with a learned feature of the difference between the aggregated message and the unit's previous representation. Residual fusion also retains the previous and initial unit representations as messages propagate through the graph.

Third, physics-informed training penalizes conservation violations using predicted inlet and outlet stream properties. Mass balances apply to valid units with represented inlet and outlet streams, component balances apply to non-reactive units, and atom balances apply to reactive units. These unit-level rules can be reused across flowsheets wherever the corresponding units and streams are represented. The conservation penalties are combined with supervised stream-property losses after an initial supervised training period. Separate prediction heads estimate temperature and pressure, composition, and mass flow rate; softmax ensures that the predicted component mole fractions are nonnegative and sum to one.

Experiments cover ten steam methane reforming (SMR) flowsheets under single-process prediction, multi-process prediction, zero-shot transfer, and limited-data fine-tuning. Compared with the strongest competing model in each setting, ProcGNN reduces average sMAPE from 0.1140 to 0.0681 in single-process prediction and from 0.6773 to 0.2196 in multi-process prediction, corresponding to relative error reductions of 40.2% and 67.6%, respectively. In zero-shot transfer to an unseen flowsheet, it achieves the lowest sMAPE for eight of the ten stream properties. Ablations further show that the architectural components and the three conservation terms contribute to prediction accuracy and conservation satisfaction.
## 2. Reproducibility

### Datasets

Experiments cover ten steam-methane-reforming flowsheets (P01–P10) and ten stream properties. The evaluation protocol uses five canonical grouped folds. Preprocessing is fitted on training data only, and paired methods use the same split manifests.

Raw simulator-derived data and trained checkpoints are managed separately and are not distributed in this Git repository. Obtain approved local data and follow [data/README.md](data/README.md) before training or checkpoint-based evaluation. The model expects merged process data, stream tables, graph specifications, heat-exchanger pairing metadata, and the original split manifests.

### Requirements and Environment Setup

The reported training environment used **Python 3.10, PyTorch 2.5.1, and CUDA 12.1**. Message passing, attention, and graph aggregation use native PyTorch operations; the core model does not require PyTorch Geometric or DGL. Dependencies are listed in [requirements.txt](requirements.txt). The original server package snapshot is available in [requirements_34.txt](requirements_34.txt), but is not a cross-platform lockfile.

```bash
# Download the source code
git clone https://github.com/NSLab-CUK/ProcGNN.git
cd ProcGNN

# Create and activate a Python environment
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch and the remaining dependencies
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

For other CUDA builds or CPU-only installations, see the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

<details>
<summary>Windows PowerShell</summary>

Use `py -3.10 -m venv .venv`, activate with `.\.venv\Scripts\Activate.ps1`, and set `$env:PYTHONPATH = "src"`. The commands below use Bash line continuations; enter them on one line or use PowerShell continuation syntax on Windows.

</details>

### How to Run ProcGNN

Run from the repository root with the environment activated. The default model configuration is [model_260805_10d_frac1.yaml](configs/experiment/pinn/model_260805_10d_frac1.yaml), and the experiment registry is [final_protocol_260811.yaml](configs/final_experiments/final_protocol_260811.yaml).

Train a joint model across all ten flowsheets, repeated over five folds:

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

To inspect the complete experiment plan without starting training:

```bash
python scripts/run_final_experiments.py --families all \
  --plan-json outputs/reproduction/command_plan.json
```

The plan command writes a JSON file but does not execute training. Always use a new output root and preserve the original data partitions when reproducing a reported result.

### Hyperparameters

Model and loss settings are defined in YAML. The current reproduction preset uses:

| Hyperparameter | Value |
|---|---|
| GNN layers / hidden dimension | 5 / 384 |
| Graph readout | Set2Set, 3 processing steps |
| Dropout | 0.1 |
| Optimizer / initial learning rate | AdamW / 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size / maximum epochs | One process graph / 30 |
| Gradient-clipping norm | 0.5 |

The architecture figure illustrates a four-layer variant; the checked-in reproduction preset uses five layers. Encoder depth is controlled by `overrides.model.num_layers`.

Useful training options include:

- `--base-config`: YAML configuration for the experiment.
- `--process-ids`: flowsheets to include, for example `1 2 3`.
- `--only-folds`: fold indices to run, for example `1 2 3 4 5`.
- `--max-epochs`: maximum training epochs, for example `30`.
- `--device`: execution device, for example `cuda`.
- `--output-root`: destination for the run artifacts.

Use the selected script's `--help` for its complete argument list.

### Additional Experiments and Checks

The [experiment catalogue](docs/EXPERIMENT_CATALOG.md) and [script index](scripts/README.md) describe single-process training, baselines, unseen-process transfer, data-efficiency transfer, sensitivity analysis, SHAP explanations, and computational-efficiency measurements. Figures identified as model-based projections are not measured transfer results.

The [economic optimization module](economic%20module%20GA/module/README.md) supports constrained GA screening and Aspen handoff. Surrogate-screened candidates must be independently recalculated and constraint-checked in Aspen before they are reported as validated optima.

For a lightweight installation check using small CPU graphs:

```bash
python -m pip install pytest
python -m pytest -q tests/test_process_surrogate_smoke.py
```

See the [documentation index](docs/README.md) for further model details, data requirements, and reproducibility protocols.

## 3. Reference

- [Model equations and implementation](docs/MODEL_260805_10D_FRAC1_equation_flow_reviewed.md)
- [Final experiment protocol](docs/FINAL_PAPER_PROTOCOL_260811.md)
- [Experiment catalogue](docs/EXPERIMENT_CATALOG.md)

## 4. Citing ProcGNN

If ProcGNN is useful in your work, please reference the source-code repository and record the exact commit used for your experiments. The entry below cites the software repository, not a research paper:

```bibtex
@misc{procgnn_code,
  title        = {ProcGNN},
  howpublished = {\url{https://github.com/NSLab-CUK/ProcGNN}},
  note         = {Source code repository}
}
```

A paper-specific citation and publication link will be added once the title, author list, and DOI or preprint identifier are confirmed.

## 5. Contributors

Developed at [Network Science Lab @ CUK](https://nslab-cuk.github.io/). Repository maintainer: [JunheeCho3337](https://github.com/JunheeCho3337).

Please [open an issue](https://github.com/NSLab-CUK/ProcGNN/issues) for questions or reproducibility reports.

No software license is currently included in the repository.

