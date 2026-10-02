# Unseen Process Transfer Learning Pipeline

This document records the first implementation of the held-out-process transfer
learning workflow for the process GNN surrogate.

## Goal

For one held-out target process:

- Pretrain on the remaining source processes.
- Fine-tune on a small target-process adaptation subset.
- Compare against scratch training on the exact same target subset.
- Evaluate all fine-tune and scratch runs on one fixed target test manifest.

The implementation supports one held-out process at a time, with either the
legacy single split or a fold-aware split such as 5-fold transfer evaluation.
Repeating over all ten held-out processes and aggregating process-macro results
is intentionally left for a later stage.

## New Files

| File | Purpose |
| --- | --- |
| `scripts/create_process_transfer_splits.py` | Builds source train/val manifests, fixed target test manifest, target adaptation pool, and nested ratio train/val manifests. |
| `scripts/run_process_transfer_experiments.py` | Runs pretrain, fine-tune, or scratch stages by writing runtime overrides and calling `scripts/train_process_surrogate.py`. |
| `scripts/create_process_unseen_splits.py` | Builds 5-fold source train/val and held-out test manifests for zero-shot evaluation from per-process K-fold manifests. |
| `scripts/eval_process_surrogate_edge_all.py` | Runs held-out manifest evaluation from a saved checkpoint and writes zero-shot evaluation metadata. |
| `scripts/run_unseen_process_experiments.py` | Runs fold-wise source pretraining, zero-shot evaluation, and fold aggregation. |
| `scripts/create_process_unseen_transfer_splits.py` | Builds full held-out process split trees for zero-shot and transfer ratios from per-process K-fold manifests. |
| `scripts/run_unseen_process_transfer_experiments.py` | Unified runner for splits, source pretraining, zero-shot evaluation, fine-tuning, scratch, and aggregation across selected held-out processes/folds/ratios. |
| `docs/UNSEEN_PROCESS_TRANSFER.md` | This handoff document. |

## Training Script Changes

`scripts/train_process_surrogate.py` now supports model-only transfer
initialization:

```text
--pretrained-checkpoint PATH
--pretrained-load-mode model_only
--pretrained-strict-load / --no-pretrained-strict-load
--finetune-mode full | head_only
```

This is separate from resume training.

- `model_only` loads only `model_state_dict` or `model`.
- Optimizer, scheduler, epoch, best metric, early stopping state, and gradient
  scaler state are newly initialized.
- `full` trains all model parameters.
- `head_only` freezes `model.encoder` and trains only prediction heads
  (`edge_decoder`, `edge_stream_head`, and/or `heads.*`).
- `transfer_initialization.json` is written in each run directory.
- Checkpoints include a `transfer_initialization` metadata block.

`src/process_graph/experiment/train_utils.py` now builds optimizers from
parameters with `requires_grad=True`, so head-only fine-tuning does not optimize
frozen encoder parameters. Normal training is unchanged because all parameters
remain trainable.

## Split Structure

Single-split legacy example:

```text
data/splits/transfer/heldout_P10/
|-- metadata.json
|-- ratio_summary.csv
|-- source_train.csv
|-- source_val.csv
|-- target_adaptation_pool.csv
|-- target_test.csv
|-- ratio_01/
|   |-- train.csv
|   `-- val.csv
|-- ratio_05/
|   |-- train.csv
|   `-- val.csv
|-- ratio_10/
|   |-- train.csv
|   `-- val.csv
|-- ratio_20/
|   |-- train.csv
|   `-- val.csv
`-- ratio_50/
    |-- train.csv
    `-- val.csv
```

Rows are grouped by `ID` before splitting, so rows from the same sample do not
cross train/val/test boundaries. Ratio subsets are nested: `1%` is contained in
`5%`, which is contained in `10%`, and so on.

For 5-fold transfer, pass `--k-folds 5`. Source folds are stratified by
process, so each fold keeps the source-process sample ratio balanced instead of
mixing all source samples globally. The root then contains fold subdirectories,
and each fold has the same manifest layout as the single-split case:

```text
data/splits/transfer/heldout_P10_5fold/
|-- metadata.json
|-- split_summary.csv
|-- fold_01/
|   |-- metadata.json
|   |-- ratio_summary.csv
|   |-- source_train.csv
|   |-- source_val.csv
|   |-- target_adaptation_pool.csv
|   |-- target_test.csv
|   `-- ratio_01/
|       |-- train.csv
|       `-- val.csv
`-- fold_05/
    `-- ...
```

Within each fold, the held-out process test sample IDs are disjoint from that
fold's adaptation pool. Source process validation sample IDs are also fold
specific and process-balanced. Ratio subsets remain nested inside each fold.

## Leakage Policy

The implemented default is:

- Pretraining normalization is fit only from the source train manifest.
- Fine-tuning normalization is fit only from the target adaptation train
  manifest for the selected ratio.
- Scratch normalization is fit from the same target adaptation train manifest as
  the matching fine-tune run.
- Target validation and target test rows are not used for fitting normalizers.
- Augmentation, if enabled by the base config, can only see the runtime train
  manifest because augmentation is applied after the train dataset is built from
  that manifest.

The first implementation does not implement source-fixed normalizer reuse from a
checkpoint. If that experiment is needed, add an explicit
`--normalization-policy source_fixed` path after checkpoint-side scaler metadata
is available.

## Example Commands

Create held-out Process 10 splits:

```bash
PYTHONPATH=src python scripts/create_process_transfer_splits.py \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --ratios 0.01 0.05 0.10 0.20 0.50 \
  --seed 42 \
  --target-test-ratio 0.20 \
  --target-val-ratio 0.20 \
  --output-root data/splits/transfer/heldout_P10 \
  --overwrite
```

Create held-out Process 10 5-fold splits:

```bash
PYTHONPATH=src python scripts/create_process_transfer_splits.py \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --ratios 0.01 0.05 0.10 0.20 0.50 \
  --seed 42 \
  --target-test-ratio 0.20 \
  --target-val-ratio 0.20 \
  --k-folds 5 \
  --output-root data/splits/transfer/heldout_P10_5fold \
  --overwrite
```

Pretrain on source processes:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/proposed/p5_aug12k_e3mid.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --split-root data/splits/transfer/heldout_P10 \
  --mode pretrain \
  --max-epochs 5 \
  --output-root outputs/transfer \
  --force-reevaluate \
  --skip-startup-debug
```

For 5-fold runs, use the fold-aware split root and add `--k-folds 5`. Add
`--only-folds 1` during smoke tests, or omit it to run all five folds:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/proposed/p5_aug12k_e3mid.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --split-root data/splits/transfer/heldout_P10_5fold \
  --mode pretrain \
  --k-folds 5 \
  --max-epochs 5 \
  --output-root outputs/transfer_5fold \
  --force-reevaluate \
  --skip-startup-debug
```

Fine-tune full model on 1% target adaptation data:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/proposed/p5_aug12k_e3mid.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --ratios 0.01 \
  --split-root data/splits/transfer/heldout_P10 \
  --mode finetune \
  --finetune-mode full \
  --max-epochs 5 \
  --output-root outputs/transfer \
  --force-reevaluate \
  --skip-startup-debug
```

Scratch train on the same 1% target adaptation data:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/proposed/p5_aug12k_e3mid.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --ratios 0.01 \
  --split-root data/splits/transfer/heldout_P10 \
  --mode scratch \
  --max-epochs 5 \
  --output-root outputs/transfer \
  --force-reevaluate \
  --skip-startup-debug
```

The runner will auto-discover the latest source pretrain `best.pt` for fine-tune
when `--pretrained-checkpoint` is omitted.

## 5-Fold Zero-Shot Evaluation Design

This section records the intended zero-shot evaluation workflow. It is distinct
from the fine-tuning transfer workflow above.

For one held-out process, such as P10:

```text
Source processes: P01-P09
Held-out process: P10
```

Each fold trains a fresh source-only model and evaluates that fold's best
checkpoint directly on the held-out process test manifest:

```text
fold_01 source P01-P09 train/val -> fold_01 P10 test zero-shot
fold_02 source P01-P09 train/val -> fold_02 P10 test zero-shot
...
fold_05 source P01-P09 train/val -> fold_05 P10 test zero-shot
```

The rules are:

- Each fold starts from random model initialization.
- Fold checkpoints are independent; do not reuse or continue another fold.
- The held-out process is excluded from source train and source validation.
- The held-out test manifest is not used for normalizer fitting, checkpoint
  selection, early stopping, or monitor metric selection.
- The best checkpoint is selected only from source validation metrics, usually
  `val_target_mean_r2` with `monitor_mode=max`.
- Zero-shot evaluation runs with `model.eval()` and `torch.no_grad()` or
  `torch.inference_mode()`.
- Zero-shot evaluation must not create an optimizer or scheduler, run backward,
  update checkpoints, or fit a normalizer from held-out test rows.

### Zero-Shot Split Layout

The desired zero-shot split layout is:

```text
data/splits/unseen/heldout_P10/
|-- metadata.json
|-- fold_01/
|   |-- source_train.csv
|   |-- source_val.csv
|   |-- heldout_test.csv
|   `-- metadata.json
|-- fold_02/
|-- fold_03/
|-- fold_04/
`-- fold_05/
```

Each manifest should keep the columns expected by
`ProcessGraphTabularDataset`:

```text
process_id
sample_id
merged_row_index
```

If the source manifest has additional columns, preserve them when practical.

Each fold must validate:

- `source_train.csv` contains no held-out process rows.
- `source_val.csv` contains no held-out process rows.
- `heldout_test.csv` contains only the held-out process.
- Source train, source validation, and held-out test row indices are disjoint.
- Every `merged_row_index` is valid.
- No manifest is empty.
- All requested source processes are present.

For the 5-fold set, also check whether held-out test rows are disjoint across
folds. If the upstream K-fold definition intentionally allows overlap, record
that policy in metadata instead of silently assuming disjoint folds.

### Normalization Policy

For zero-shot, each fold's normalization statistics must come from only that
fold's source train manifest:

```text
fold_01 normalizer: fold_01/source_train.csv
fold_02 normalizer: fold_02/source_train.csv
...
fold_05 normalizer: fold_05/source_train.csv
```

Do not include source validation, held-out train/adaptation rows, held-out
validation rows, or held-out test rows in normalizer fitting.

Standalone evaluation should prefer reusing fold pretraining normalizer/scaler
artifacts if available. If those artifacts are not stored, it may recompute the
same statistics from that fold's `source_train.csv`. The evaluation metadata
must record the normalization source and the source train manifest path.

### Zero-Shot Output Layout

Recommended output structure:

```text
outputs/unseen/heldout_P10/
|-- experiment_metadata.json
|-- fold_01/
|   |-- pretrain/
|   |   |-- runtime_overrides.json
|   |   `-- training artifacts including best.pt
|   `-- zero_shot/
|       |-- evaluation_metadata.json
|       |-- target-edge metric artifacts
|       `-- all-edge metric artifacts
|-- fold_02/
|-- fold_03/
|-- fold_04/
|-- fold_05/
`-- aggregate/
    |-- fold_metrics.csv
    |-- metric_summary.csv
    |-- target_property_summary.csv
    |-- all_edge_property_summary.csv
    `-- aggregate_metadata.json
```

Every `evaluation_metadata.json` should include:

```text
fold
heldout_process
unique_process_ids
test sample count
checkpoint path
source train manifest
source val manifest
heldout test manifest
normalization source
```

The evaluation should fail if `unique_process_ids` contains any process other
than the held-out process.

### Aggregation Policy

The primary 5-fold result is the macro average over folds:

```text
R2_macro_over_folds = mean(R2_fold_01, ..., R2_fold_05)
```

Report metrics as mean plus standard deviation. Keep pooled prediction R2 as a
separate optional diagnostic if predictions from all folds are concatenated.
Do not mix fold-macro R2 and pooled R2 in the same column.

Minimum aggregate files:

```text
fold_metrics.csv
metric_summary.csv
target_property_summary.csv
all_edge_property_summary.csv
aggregate_metadata.json
```

`fold_metrics.csv` stores one row per fold. `metric_summary.csv` stores
`mean`, `std`, `min`, `max`, and `valid_folds` for summary metrics such as:

```text
Target mean R2
Flatten R2
Edge macro R2
All-edge mean R2
```

Target and all-edge property summaries should include every property exported by
the metric code. At minimum, verify these rows:

```text
Target: Frac_CH4, Frac_CO, Frac_CO2, Frac_H2, Frac_H2O, Mass_Flow, Mole_Flow, Vol_Flow
All-edge: Frac_H2, Frac_O2, Mass_Flow, Mole_Flow, Vol_Flow, Enthalpy
```

By default, all five folds must succeed before aggregate results are written.
If partial aggregation is explicitly allowed, metadata must list successful
folds, failed folds, failure reasons, and `valid_folds`.

### Reproducibility

Split generation should use the existing fold manifests and should not reshuffle
fold assignments. Training seed policy should be explicit:

```text
fold_seed_mode=fixed   -> every fold uses base seed, e.g. 42
fold_seed_mode=offset  -> fold seed = base_seed + fold_index
```

Use `fixed` as the default because it is easier to compare fold behavior.

### Intended Commands

Create or verify 5-fold unseen manifests:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/create_process_unseen_splits.py \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --folds 1 2 3 4 5 \
  --input-split-root data/splits/process_kfold \
  --output-root data/splits/unseen/heldout_P10 \
  --overwrite
```

Run fold 1 only:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --folds 1 \
  --split-root data/splits/unseen/heldout_P10 \
  --max-epochs 5 \
  --monitor-metric val_target_mean_r2 \
  --monitor-mode max \
  --seed 42 \
  --output-root outputs/unseen/heldout_P10 \
  --force-reevaluate \
  --skip-startup-debug
```

Run all five folds:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --folds 1 2 3 4 5 \
  --split-root data/splits/unseen/heldout_P10 \
  --max-epochs 5 \
  --monitor-metric val_target_mean_r2 \
  --monitor-mode max \
  --seed 42 \
  --output-root outputs/unseen/heldout_P10 \
  --force-reevaluate \
  --skip-startup-debug
```

Aggregate existing fold outputs only:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --folds 1 2 3 4 5 \
  --split-root data/splits/unseen/heldout_P10 \
  --output-root outputs/unseen/heldout_P10 \
  --aggregate-only
```

Evaluate one fold from an existing checkpoint:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-process 10 \
  --source-process-ids 1 2 3 4 5 6 7 8 9 \
  --folds 1 \
  --split-root data/splits/unseen/heldout_P10 \
  --output-root outputs/unseen/heldout_P10 \
  --eval-only \
  --checkpoint outputs/unseen/heldout_P10/fold_01/pretrain/checkpoints/best.pt \
  --force-reevaluate \
  --skip-startup-debug
```

## Full Unseen Zero-Shot And Transfer Pipeline

The complete pipeline is exposed through:

```text
scripts/create_process_unseen_transfer_splits.py
scripts/run_unseen_process_transfer_experiments.py
```

It supports the final experimental grid:

```text
10 held-out processes x 5 folds x zero-shot
10 held-out processes x 5 folds x 5 ratios x {finetune, scratch}
```

The split generator creates, for each held-out process and fold:

```text
source_train.csv
source_val.csv
target_adaptation_pool.csv
target_test.csv
ratio_01/train.csv
ratio_01/val.csv
...
ratio_50/train.csv
ratio_50/val.csv
```

The target adaptation pool is:

```text
held-out fold train + held-out fold val
```

Then a deterministic pool split and fixed permutations are used so ratio subsets
are nested:

```text
ratio_01/train subset ratio_05/train subset ...
ratio_01/val   subset ratio_05/val   subset ...
```

The target test manifest is always the held-out process fold test manifest and
is not used for source pretraining, checkpoint selection, ratio adaptation,
scratch training, augmentation, or normalizer fitting.

### Unified Runner Modes

`scripts/run_unseen_process_transfer_experiments.py` supports:

```text
splits
pretrain
zero_shot
finetune
scratch
aggregate
all
```

`all` runs:

```text
split creation
source pretraining
zero-shot evaluation
fine-tuning
scratch
aggregation
```

For real execution, `all` first runs pretraining and then launches later stages
after the fold-specific `best.pt` checkpoints exist.

The runner writes:

```text
run_metadata.json
status.json
runtime_overrides.json
command.txt
stdout.log
stderr.log
```

`--resume-existing` skips completed runs with required artifacts. `--force-reevaluate`
overrides that behavior.

### Full Pipeline Commands

Create all splits:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 1 2 3 4 5 6 7 8 9 10 \
  --folds 1 2 3 4 5 \
  --ratios 0.01 0.05 0.10 0.20 0.50 \
  --split-root data/splits/unseen_transfer \
  --output-root outputs/unseen_transfer \
  --mode splits \
  --force-reevaluate \
  --skip-startup-debug
```

P10 fold 1 source pretraining only:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 10 \
  --folds 1 \
  --ratios 0.01 \
  --split-root data/splits/unseen_transfer \
  --max-epochs-pretrain 5 \
  --output-root outputs/unseen_transfer \
  --mode pretrain \
  --resume-existing \
  --skip-startup-debug
```

P10 fold 1 zero-shot evaluation:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 10 \
  --folds 1 \
  --ratios 0.01 \
  --split-root data/splits/unseen_transfer \
  --output-root outputs/unseen_transfer \
  --mode zero_shot \
  --resume-existing \
  --skip-startup-debug
```

P10 fold 1 ratio 1 percent fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 10 \
  --folds 1 \
  --ratios 0.01 \
  --methods finetune \
  --split-root data/splits/unseen_transfer \
  --max-epochs-transfer 5 \
  --finetune-mode full \
  --output-root outputs/unseen_transfer \
  --mode finetune \
  --resume-existing \
  --skip-startup-debug
```

P10 fold 1 ratio 1 percent scratch:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 10 \
  --folds 1 \
  --ratios 0.01 \
  --methods scratch \
  --split-root data/splits/unseen_transfer \
  --max-epochs-transfer 5 \
  --output-root outputs/unseen_transfer \
  --mode scratch \
  --resume-existing \
  --skip-startup-debug
```

P10 full 5-fold zero-shot and transfer:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 10 \
  --folds 1 2 3 4 5 \
  --ratios 0.01 0.05 0.10 0.20 0.50 \
  --methods finetune scratch \
  --split-root data/splits/unseen_transfer \
  --max-epochs-pretrain 5 \
  --max-epochs-transfer 5 \
  --monitor-metric val_target_mean_r2 \
  --monitor-mode max \
  --seed 42 \
  --fold-seed-mode fixed \
  --normalization-policy source_fixed \
  --finetune-mode full \
  --gpu-ids 0 \
  --max-parallel 1 \
  --output-root outputs/unseen_transfer \
  --mode all \
  --resume-existing \
  --skip-startup-debug
```

Server 1, held-out P01-P05:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 1 2 3 4 5 \
  --folds 1 2 3 4 5 \
  --ratios 0.01 0.05 0.10 0.20 0.50 \
  --methods finetune scratch \
  --split-root data/splits/unseen_transfer \
  --max-epochs-pretrain 5 \
  --max-epochs-transfer 5 \
  --gpu-ids 0 1 2 3 \
  --max-parallel 4 \
  --output-root outputs/unseen_transfer \
  --mode all \
  --resume-existing \
  --skip-startup-debug
```

Server 2, held-out P06-P10:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 6 7 8 9 10 \
  --folds 1 2 3 4 5 \
  --ratios 0.01 0.05 0.10 0.20 0.50 \
  --methods finetune scratch \
  --split-root data/splits/unseen_transfer \
  --max-epochs-pretrain 5 \
  --max-epochs-transfer 5 \
  --gpu-ids 0 1 2 3 \
  --max-parallel 4 \
  --output-root outputs/unseen_transfer \
  --mode all \
  --resume-existing \
  --skip-startup-debug
```

Aggregation only:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 1 2 3 4 5 6 7 8 9 10 \
  --folds 1 2 3 4 5 \
  --ratios 0.01 0.05 0.10 0.20 0.50 \
  --split-root data/splits/unseen_transfer \
  --output-root outputs/unseen_transfer \
  --mode aggregate \
  --allow-partial-aggregation
```

Failed-run rerun:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_unseen_process_transfer_experiments.py \
  --base-config configs/experiment/pinn/pinn_weighting/pinnw7_mass_atom_volume.yaml \
  --heldout-processes 10 \
  --folds 1 \
  --ratios 0.01 \
  --methods finetune scratch \
  --split-root data/splits/unseen_transfer \
  --output-root outputs/unseen_transfer \
  --mode all \
  --resume-existing \
  --force-reevaluate \
  --skip-startup-debug
```

## Output Layout

```text
outputs/transfer/
`-- heldout_P10/
    |-- pretrain/
    |   `-- source_1_2_3_4_5_6_7_8_9/
    |-- ratio_01/
    |   |-- finetune_full/
    |   |-- finetune_head_only/
    |   `-- scratch/
    |-- ratio_05/
    |   |-- finetune_full/
    |   |-- finetune_head_only/
    |   `-- scratch/
    `-- ...
```

With `--k-folds 5`, the runner writes the same stage layout under each fold:

```text
outputs/transfer_5fold/
`-- heldout_P10/
    |-- fold_01/
    |   |-- pretrain/
    |   |   `-- source_1_2_3_4_5_6_7_8_9/
    |   `-- ratio_01/
    |       |-- finetune_full/
    |       `-- scratch/
    `-- fold_05/
        `-- ...
```

Each stage directory contains `runtime_overrides.json` and
`transfer_run_metadata.json`. The actual training run is still timestamped by
`train_process_surrogate.py` under that stage directory.

## Validation Checklist

Before a long run:

1. Run split generation and inspect `metadata.json` and `ratio_summary.csv`.
2. Confirm `target_test.csv` is disjoint from every ratio train/val manifest.
3. Run transfer runner with `--dry-run` to inspect commands.
4. Run a short `--max-epochs 1` smoke pretrain.
5. Run a short fine-tune with `--pretrained-checkpoint` or auto-discovered
   source `best.pt`.
6. Inspect `transfer_initialization.json` in the run directory.
7. Confirm fine-tune starts at epoch 1 and creates a new optimizer state.
8. Run the matching scratch command and confirm it uses the same ratio manifests.

## Current Limitations

- No automatic loop over all held-out processes.
- Full held-out-process loop is available in
  `run_unseen_process_transfer_experiments.py`, but process-level macro and
  transfer-gain summaries are currently minimal and should be expanded before
  paper tables.
- No multi-seed orchestration beyond one seed policy per invocation.
- Zero-shot evaluation is implemented through a wrapper around
  `train_process_surrogate.py --max-epochs 0`; it avoids training epochs and
  backward passes, but the backend still constructs the standard training
  objects needed by the existing metric path.
- `--normalization-policy source_fixed` is recorded in metadata, but the current
  backend still fits scalers from each runtime train manifest unless deeper
  normalizer-state loading is implemented in `train_process_surrogate.py`.
- No 100% target upper-bound mode yet.

Unseen process transfer 1st implementation is complete. Existing standard
training paths remain supported.
