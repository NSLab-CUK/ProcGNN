# Edge-All Target Stream Weighting Handoff

## Goal

The current objective is edge-all stream vector prediction, not amount target prediction.
The model still predicts every canonical stream edge with the 12 stream features in
`STREAM_EDGE_FEATURE_SLOTS` order:

`Temp, Pres, Vol_Flow, Mole_Flow, Mass_Flow, Frac_H2O, Frac_H2, Frac_CH4, Frac_CO2, Frac_CO, Frac_O2, Frac_N2`.

The grouped-property `EdgeDecoder` remains unchanged. No target-specific H2, CO2, H2O, or amount head was added.

## Loss Policy

`data/reference/v4/target_stream_targets.csv` is now used as the source of truth for important target stream edges.
For each process, target stream edges are resolved by unique `(process_id, canonical_edge_id)`.

The base `edge_all` loss applies an edge-level weight:

```text
sum_e sum_f mask[e,f] * edge_weight[e] * SmoothL1(pred[e,f], true[e,f])
/ sum_e sum_f mask[e,f] * edge_weight[e]
```

The same edge weight applies to all supervised features on the target stream edge. It is not a Frac-only weight.

The default config keys are:

```yaml
use_target_stream_loss_weighting: true
target_stream_loss_weight: 5.0
target_stream_weighting_mode: unique_target_stream_edge_by_process
target_stream_weight_conflict_policy: max
allow_yaml_only_target_streams: false
```

Final training loss is:

```text
loss_total_final = weighted_edge_all + weighted_all_edge_r2
```

`loss_primary_frac`, `loss_target_row_frac`, and `loss_v4_targets_frac` are legacy/deprecated compatibility keys and do not contribute to the final loss under `target_feature_loss_scope: target_stream_edge_weighted_edge_all`.

## Matching

Target stream edge matching is process-local:

1. Use `canonical_answer_edge_id` when present.
2. Otherwise fallback to `required_stream_key`, `main_data_stream_key`, or target stream name in the same process.
3. YAML override rows may use `canonical_answer_edge_id`, `fallback_stream_key`, or `target_stream`.
4. `target_species` is metadata only and is never used for edge routing.
5. Failed matches are skipped with a `skip_reason`; no substitute edge is invented.

If the same target stream edge appears in multiple target rows, the edge is weighted once. If YAML gives multiple weights for one edge, `target_stream_weight_conflict_policy: max` keeps the largest weight.

Process6 `OUT_EXHAUST` is handled through `P06_E013` from the CSV, with `fallback_stream_key: "17"` in the Process6 YAML for clarity. It is never rerouted to another CO2 edge.

## YAML Overrides

Process-specific configs can set:

```yaml
target_stream_loss_weights:
  - target_id: P06_T004
    canonical_answer_edge_id: P06_E013
    target_stream: OUT_EXHAUST
    target_species: CO2
    target_feature: "Stream17_Mole_Flow*Stream17_Frac_CO2"
    fallback_stream_key: "17"
    weight: 4.0
```

With `allow_yaml_only_target_streams: false`, YAML can override CSV-defined target stream edges but cannot create new target streams. With `true`, YAML-only target streams are allowed only if they match a canonical edge in the same process.

## Metrics

Overall edge-all metrics remain enabled:

- edge_all MAE/RMSE/R2
- property-level MAE/RMSE/R2
- process-level MAE/RMSE/R2
- stream-role metrics

Target stream feature metrics are saved to:

```text
target_stream_feature_metrics.json
target_stream_feature_metrics.csv
target_stream_loss_weighting_diagnostics.csv
```

The JSON is split-aware and reports target stream feature prediction on original scale. Missing supervision for a feature is represented as `n=0` and null metric values.

Derived amount metrics such as `Mole_Flow * Frac_H2`, `Mole_Flow * Frac_CO2`, and `Mole_Flow * Frac_H2O` are legacy/diagnostic only. They are not the primary metric.
