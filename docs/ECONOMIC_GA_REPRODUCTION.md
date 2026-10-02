# Economic GA reproduction protocol

## Scope

This protocol evaluates constrained GNN-coupled economic optimisation for all ten SMR flowsheets (P01--P10). It compares genetic algorithm (GA) search against equal-budget random search, then sends only selected candidates for independent Aspen revalidation.

The resulting stages have different evidential status:

1. **Surrogate-screened optimisation:** GNN-predicted stream properties are passed through the economic and constraint pipeline.
2. **Aspen validation:** selected operating conditions are independently recalculated in Aspen. Only this stage supports a simulator-validated optimisation claim.

## Fixed conditions

- GNN configuration: `configs/experiment/pinn/model_260805_10d_frac1.yaml` plus the paired runtime overrides and checkpoint recorded by each run.
- Process template: ID 5000 where available; otherwise the module-selected central template row.
- Search variables: non-sink operating variables bounded by the 1st--99th percentile of observed data. Feed-flow scales remain fixed unless `--include-feed-flow` is explicitly used.
- Production constraint: 90--110% of the process-specific GNN template production.
- Process constraints: S/C ratio, reformer temperature, product purity, pinch, and stream sanity, as implemented in `objective.py`.
- Economics: GNN-predicted streams only, itemised OPEX, and the prices bundled with the economic module.
- Input safety: `--unsafe-input-proxies` is forbidden for paper optimisation.

## Search arms and budget

| Arm | Search methods | CO2 constraint | Purpose |
|---|---|---|---|
| `LCOH_capacity` | GA and random | none | Main feasible-LCOH comparison |
| `CO2_90pct` | GA and random | ≤90% of process-specific template specific CO2 | Environmental trade-off |

Paper-profile settings are five seeds (`101, 202, 303, 404, 505`), population 24, and 30 generations. Each process/method/seed/arm evaluates `24 × (30 + 1) = 744` candidates. The complete paper profile therefore plans 148,800 GNN-economic candidate evaluations across ten processes, two methods, two arms, and five seeds.

For development, the runner supplies two smaller profiles:

| Profile | Seeds | Population | Generations |
|---|---:|---:|---:|
| `smoke` | 101, 202 | 4 | 1 |
| `pilot` | 101, 202, 303 | 12 | 8 |
| `paper` | 101, 202, 303, 404, 505 | 24 | 30 |

## Commands

Run from the repository root with an activated Python environment and a verified CUDA PyTorch installation.

```powershell
Set-Location -LiteralPath "economic module GA/경제성 평가 모듈"
python test_gnn_economic_smoke.py
./run_paper_experiments.ps1 -Profile smoke -Arms primary -Device cpu
./run_paper_experiments.ps1 -Profile pilot -Arms all -Device cuda
./run_paper_experiments.ps1 -Profile paper -Arms all -Device cuda
```

Restrict a study to one process only for debugging:

```powershell
./run_paper_experiments.ps1 -Profile paper -Arms all -Processes P03 -Device cuda
```

Every invocation creates a new timestamped `outputs/paper_economic_ga_<timestamp>/` root. The runner refuses to reuse an existing output root, preventing accidental mixing of arms or seeds.

## Required outputs

| Output | Meaning |
|---|---|
| `<arm>/<method>/seed_<seed>/ga_history.csv` | Complete candidate-level search history and violations |
| `<arm>/<method>/seed_<seed>/ga_summary.json` | Best feasible solution and GNN/scaler/proxy provenance |
| `paper_summary/paper_aggregate.csv` | Cross-process, cross-seed summary |
| `paper_summary/paper_paired_ga_vs_random.csv` | Equal-budget GA/random comparison |
| `paper_summary/aspen_recheck_candidates.csv` | Candidate queue for simulator validation |
| `paper_summary/paper_report.json` | Study-level machine-readable report |

Prepare the validation handoff only after the full study summary is complete:

```bash
python prepare_aspen_validation_handoff.py --root <PAPER_OUTPUT_ROOT>
```

## Reporting rules

- Select only feasible candidates; never substitute a penalty-minimising infeasible point for a feasible optimum.
- Pair GA and random search by process, arm, seed, population, and generation budget.
- Report GNN-screened and Aspen-recalculated values separately.
- Retain the command line, config/checkpoint checksum, scaler provenance, proxy audit, and candidate JSON for every Aspen handoff.
- If Aspen violates any required constraint or materially changes ranking, exclude the candidate from the final validated optimum set.
