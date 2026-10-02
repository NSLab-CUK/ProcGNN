# GNN-coupled economic optimisation module

This module converts predicted stream properties from the 10D process-graph surrogate into techno-economic indicators for P01--P10, then searches feasible operating conditions with a genetic algorithm (GA).

## Scope

- Objective: minimise levelised cost of hydrogen (LCOH).
- Constraints: production window, S/C ratio, reformer outlet temperature, product purity, pinch/stream-sanity constraints, and optional specific-CO2 cap.
- Processes: P01--P10.
- Surrogate inputs: predicted temperature, pressure, seven mole fractions, and mass flow for every stream required by the costing pipeline.

The GA is a **surrogate-screening** procedure. Its candidates must be recalculated in Aspen before they are reported as validated optima.

## Files

| File | Role |
|---|---|
| `objective.py` | Public `evaluate(streams, process_id)` objective/constraint interface |
| `gnn_economic.py` | GNN checkpoint loading, operating-variable perturbation, and stream prediction |
| `run_ga_optimization.py` | GA or equal-budget random-search runner |
| `summarize_paper_ga.py` | Aggregates runs and prepares Aspen candidate queues |
| `prepare_aspen_validation_handoff.py` | Writes candidate packages for independent Aspen revalidation |
| `costing_pipeline.py` | Equipment sizing, capital cost, operating cost, and LCOH calculation |
| `indicators.py` | Production, conversion, energy, emission, and feasibility indicators |
| `test_gnn_economic_smoke.py` | P01--P10 integration smoke test |

## Setup

Run commands from this directory. The parent project must contain the final model configuration, trained checkpoint, stream data under `data/main_data_Streams`, and canonical graph metadata.

```bash
cd "economic module GA/경제성 평가 모듈"
python test_gnn_economic_smoke.py
```

The smoke test is the required preflight. It verifies that all ten processes produce stream predictions and valid economic outputs before launching GA.

## Exploratory GA run

```bash
python run_ga_optimization.py \
  --processes all --search-method ga --population 20 --generations 100 \
  --seed 42 --device cuda \
  --output-dir ../../outputs/economic_ga/exploratory_g100
```

Key options:

- `--processes all` or a comma-separated selection such as `P03,P05`.
- `--search-method ga` or `random`; use matched population, generation, and seed settings for a fair search-budget comparison.
- `--include-feed-flow` enables direct CH4, water, air, and fuel-flow scales.
- `--max-specific-co2` provides an absolute CO2 cap. The paper protocol uses a process-specific relative cap for its environmental arm.
- `--unsafe-input-proxies` is excluded from paper optimisation because it permits result/proxy features that invalidate a forward-looking search.

Each run records population history, feasibility violations, configuration/checkpoint provenance, scaler use, and proxy audit in its output directory.

## Paper protocol

The paper protocol uses all ten processes, repeated seeds, equal-budget GA and random-search arms, feasibility-only selection, and Aspen revalidation of the top candidates.

```powershell
./run_paper_experiments.ps1 -Profile paper -Arms all -Device cuda
```

Then aggregate and prepare the Aspen queue:

```bash
python summarize_paper_ga.py --root <PAPER_OUTPUT_ROOT>
python prepare_aspen_validation_handoff.py --root <PAPER_OUTPUT_ROOT>
```

See `../../docs/ECONOMIC_GA_REPRODUCTION.md` for fixed settings, candidate-selection rules, and the distinction between surrogate and Aspen results.

## Output contract

| Artifact | Purpose |
|---|---|
| `ga_history.csv` | Every evaluated candidate and constraint result |
| `ga_summary.json` | Best feasible solution plus model/scaler/proxy provenance |
| `paper_summary/` | Cross-seed and cross-process comparison tables |
| `aspen_recheck_candidates.csv` | Candidate queue for independent Aspen validation |
| `aspen_validation_handoff/` | Self-contained handoff package and instructions |

Do not overwrite a completed paper run. Use a new timestamped output root and retain the source configuration and checkpoint checksum with every result.
