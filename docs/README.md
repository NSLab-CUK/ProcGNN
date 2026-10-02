# Documentation index

## Current reproducibility entry points

| Topic | Document |
|---|---|
| Experiment families and result interpretation | [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md) |
| Final experiment protocol | [FINAL_PAPER_PROTOCOL_260811.md](FINAL_PAPER_PROTOCOL_260811.md) |
| Active model equations and implementation review | [MODEL_260805_10D_FRAC1_equation_flow_reviewed.md](MODEL_260805_10D_FRAC1_equation_flow_reviewed.md) |
| General final-model description | [MODEL_final.md](MODEL_final.md) |
| Local directory and retention policy | [REPOSITORY_LAYOUT_260812.md](REPOSITORY_LAYOUT_260812.md) |
| GitHub release and migration policy | [GITHUB_RELEASE_GUIDE.md](GITHUB_RELEASE_GUIDE.md) |
| Economic GA reproduction and Aspen claim boundaries | [ECONOMIC_GA_REPRODUCTION.md](ECONOMIC_GA_REPRODUCTION.md) |
| Data-efficiency transfer protocol | [TRANSFER_DATA_EFFICIENCY_260811.md](TRANSFER_DATA_EFFICIENCY_260811.md) |

For exact model/training settings, verify the configuration selected by the runner. The canonical final registry points to `configs/experiment/pinn/model_260805_10d_frac1.yaml`; documentation written for another dated model is historical evidence, not an override.

## Historical evidence and generated assets

Dated model notes, architecture specifications, audits, and result summaries are retained where they are referenced by existing artifacts. `figures/`, local `paper_tables/`, and all `outputs/` remain untouched by cleanup. The exact duplicate `final_model.md` and unreferenced draft notes can be archived recoverably by the cleanup helper; `MODEL_final.md` remains canonical.
