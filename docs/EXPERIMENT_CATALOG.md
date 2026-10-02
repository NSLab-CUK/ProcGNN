# Experiment catalogue and reproducibility map

## Canonical final protocol

The final-paper registry is `configs/final_experiments/final_protocol_260811.yaml`. It fixes the following conditions unless a family explicitly changes them:

- Processes: P01--P10.
- Targets: temperature, pressure, seven gas mole fractions, and mass flow (10 properties total; volume flow excluded).
- Splits: five-fold grouped 60/20/20 splits. The joint root is `data/splits/all_processes_full100k_outer5_grouped_60_20_20`; the unseen-process root is `data/splits/single_process_full_unseen_60_20_20`.
- Preprocessing: fitted from training rows only.
- Proposed monitor: `val_target_edge_property_mean_r2`, maximised. It is a target-edge, property-wise aggregate and must not be conflated with broader all-edge or flattened R-squared diagnostics.
- Test data: used only for final evaluation after checkpoint selection.

## Main experiment families

| Family | Design | Required output evidence | Main scripts |
|---|---|---|---|
| Proposed single-process | Scratch training for P01--P10, five folds each | per-fold metrics, config, and checkpoint provenance | `run_process_kfold_experiments.py` |
| Single-process baselines | 13 models × 10 processes × 5 folds | matched split, train-only preprocessing, target-edge metrics | `run_final_baseline_experiments.py` |
| Multi-process comparison | B6, GCN, GIN, GAT, Proposed; five joint folds | one shared model per fold across all ten processes | baseline runner + GNN runner |
| Zero-shot | Train on nine processes and evaluate the held-out process | 10 source checkpoints and 50 held-out evaluations | `run_single_process_full_unseen_experiments.py` |
| Data efficiency | Transfer at 2, 4, 6, 8, 10--90% data ratios | subset manifests, hashes, fixed horizon, and fold result files | `run_transfer_data_efficiency_experiments.py` |
| Efficiency | Five joint models × five folds | training wall time, peak GPU allocation, inference benchmark | `aggregate_final_efficiency.py` |
| SHAP | Proposed target predictions, P01--P10, fold 1 | explainer metadata, background/sample IDs, aggregation tables | `run_final_shap_experiments.py` |
| Depth sensitivity | depths 1--7, five folds | default-reuse provenance and sensitivity metrics | `run_final_sensitivity_experiments.py` |
| Physics-weight sensitivity | mass/component/atom at 0.5×, 1×, 2× | changed config plus shared-default provenance | `run_final_sensitivity_experiments.py` |

The final registry records 1,445 model fits after its explicit reuse policy. Zero-shot evaluations, SHAP analyses, and efficiency measurements do not necessarily create new training fits.

## Economic GA experiment

The submodule `economic module GA/경제성 평가 모듈/` connects a trained GNN surrogate to process economics and constrained GA search.

| Stage | What is measured | Evidence |
|---|---|---|
| GNN economic smoke test | Ten processes can be converted to valid stream/economic inputs | `test_gnn_economic_smoke.py` output |
| GA search | Best feasible surrogate-screened LCOH and constraint status | `ga_history.csv`, `ga_summary.json` |
| Search comparison | GA versus equal-budget random search over repeated seeds | paper summary CSV/JSON |
| Simulator validation | Aspen recalculation of exported operating conditions | validation handoff folder and returned workbook |

The GA protocol distinguishes surrogate-screened candidates from Aspen-validated claims. Do not report a surrogate-only candidate as a simulator-confirmed optimum. See `docs/ECONOMIC_GA_PAPER_PROTOCOL_0914.md`.

## Current paper-figure entry points

| Figure family | Script | Current paper-ready output root |
|---|---|---|
| Transfer 0--10% | `plot_transfer_0_10_model_based_projection.py` | `outputs/0819final/Paper_Tables_Final3_figures/final/transfer_0_10_model_based_projection_v5_visible_error_bars_20260916/` |
| Absolute SHAP | `replot_absolute_shap_summary.py` | `outputs/0819final/Paper_Tables_Final3_figures/final/shap_absolute_abc_no_legend_compact_feed_axis40_v12_20260918/` |
| Sensitivity | `plot_sensitivity_individual_7cases.py` | `outputs/0819final/Paper_Tables_Final3_figures/final/sensitivity_7cases_center3_measured_outer4_full_sd_overlap_v4_20260918/` |
| Efficiency | `plot_efficiency_broken_y_zoom.py` | `outputs/0819final/Paper_Tables_Final3_figures/efficiency_analysis_individual_figure3_pairscaled_train080_v23_20260918/` |
| Constraint ablation | `plot_constraint_satisfaction_ablations_bars.py` | `outputs/0819final/Paper_Tables_Final3_figures/final/constraint_satisfaction_ablation_grouped_bars_v2_matched_aspect_20260918/` |

Output roots are excluded from Git. Each contains output-specific README files and, for table-driven figures, the exact input workbook used for plotting.

## Result interpretation guardrails

- Report `target_edge_property_mean_r2` before target-row or all-edge aggregates when evaluating the final GNN.
- Keep training loss, validation/checkpoint monitor, and final comparison metrics separate.
- Do not compare a target-edge result to a flattened all-edge score.
- Do not treat estimated all-stream baseline scenarios as directly measured benchmarks unless the raw benchmark is released.
- Preserve process, fold, and subset provenance in every table and figure.
