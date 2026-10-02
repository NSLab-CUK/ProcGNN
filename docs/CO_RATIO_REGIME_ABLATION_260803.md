# CO Ratio/Regime Ablation (260803)

## Goal

P03 target CO is difficult because the model sees raw feed flows but does not
receive the combustion ratios explicitly, while the previous hard sampler
selects P03 samples from the target CO value itself. This ablation separates
two changes:

1. Explicit feed-ratio input features.
2. AIR/CH4 transition-regime sampling.

The legacy path remains the default. Existing configs without the new keys keep
the same operating dimension and sampler behavior.

## Method A: explicit ratio inputs

When `known_feed_condition.ratio_features.enabled=true`, two operating features
are appended after the three raw feed-flow features:

```text
log_air_ch4_ratio   = log((AIR_Flow   + eps) / (CH4_Flow + eps))
log_water_ch4_ratio = log((WATER_Flow + eps) / (CH4_Flow + eps))
eps                 = 1e-6
```

The ratio is masked when CH4 is missing or not greater than `eps`, or when the
numerator is missing. The existing train-only operating normalizer is applied.

The values are attached to:

- `V_INPUT`: direct global feed context.
- `BURNER` (including names with a `BURNER` semantic token): local combustion
  context.
- Global graph context: indirectly through the existing Set2Set pooling of the
  node embeddings. No process ID, target value, or new target-only output head
  is introduced.

Operating dimension:

```text
legacy F1: 13 base + 3 raw feed = 16
ratio F1:  13 base + 3 raw feed + 2 log ratios = 18
```

## Method B: P03 AIR/CH4 transition sampler

The base 1,000-sample selection remains unchanged. Only the P03 part of the CO
hard-fill rule is replaced:

```text
old: P03_E014 selected by true Frac_CO >= 0.01
new: Process3 selected by 3.0 <= AIR_Flow / CH4_Flow <= 4.2
```

The new rule reads only Main input columns. It does not read `Frac_CO` or any
prediction. Its quota is 200 items within the existing `Frac_CO: 400` hard
quota. The remaining CO quota continues to cover the existing P05/P06/P07/P10
support edges. Total epoch size is unchanged:

```text
base selection: 1,000
hard fill:      1,000
total:          2,000 unique samples
```

Fold-1 train data contains 2,627 P03 candidates in this interval. Across the
train interval, the fraction of P03 `OUT` rows with `Frac_CO >= 0.01` drops
from about 48% at AIR/CH4 3.0-3.4 to about 12% at 3.8-4.2. The interval therefore
covers the transition rather than only the high-CO tail.

## 2x2 experiments

| Config | Ratio input | P03 regime sampler | Purpose |
|---|---:|---:|---|
| `c0_f1_legacy.yaml` | Off | Off | Exact legacy behavior control |
| `c1_f1_ratio.yaml` | On | Off | Ratio-feature effect |
| `c2_f1_regime.yaml` | Off | On | Regime-sampler effect |
| `c3_f1_ratio_regime.yaml` | On | On | Combined effect |

All four configs keep the same F1 model body, head, losses, target weight,
optimizer, scheduler, split, validation sampling, and epoch size. They write to
separate directories under `outputs/0803_co_ratio_regime`.

## Selection decision

The C1 ratio-input model with the existing hard sampler is the selected default.
Its standalone default config is:

```text
configs/experiment/pinn/model_260803_f1_ratio_hard_mass2.yaml
```

The original C0-C3 files remain unchanged for reproducible ablation comparisons.

## Verification

- C0 matches the original F1 YAML after removing only experiment/output names.
- Config audit: operating dimensions are `16, 18, 16, 18` for C0-C3.
- Real sampler audit: 2,000 selected, 2,000 unique, P03 transition quota 200,
  no quota shortfall, no random hard-fill fallback.
- Combined C3 smoke reached one training update and sampled validation with the
  18D input and new sampler active.
