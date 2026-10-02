# Model 260729: E2 Balanced-384 + Focus Sampler

## Final configuration

- Config: `configs/experiment/pinn/head_dimension_260729/e2_balanced384_focus_sampler.yaml`
- Output root: `outputs/0729_e2_balanced384_focus_sampler`
- Parameters: 19,434,123
- Target edge weight: 5
- Train split: 60%
- Validation split: 20%
- Test split: 20%
- Validation during training: 1,000 randomly sampled graphs per epoch

## Epoch sampling

Each epoch contains 2,000 unique graph samples.

| Group | Samples |
|---|---:|
| Base sampler | 1,000 |
| Target Frac_CO hard | 400 |
| Target Mass_Flow high (100k-500k) | 240 |
| Target Mass_Flow extreme (500k+) | 160 |
| Target Frac_CH4 hard | 200 |
| Target Frac_CO2 hard | 0 |
| Total | 2,000 |

The base 1,000 preserve process balancing, uniform sampling, sparse-positive
fraction sampling, property quantile balancing, Mass_Flow-tail coverage, and
within-epoch duplicate prevention. Dedicated CO2 hard sampling is not used.

## Dimension flow

```text
[Node input]
Node role embedding                         32
Unit type embedding                         24
Operating variables                         13
Operating mask                              13
                                              |
Operating [value, mask]                     26
  -> operating encoder: 26 -> 64
                                              |
Node concat: 32 + 24 + 64                  120
  -> node input MLP: 120 -> 384 -> 384
                                              |
Initial node hidden h0                     384
                                              |
                                              v
[Edge input for FlowGNN]
Generic stream-role embedding               24
Stream-ID embedding                         32
Structural attributes                        3
  -> structural encoder: 3 -> 16
                                              |
Edge concat: 24 + 32 + 16                   72
  -> edge input MLP: 72 -> 384 -> 384
                                              |
GNN edge embedding                         384
                                              |
                                              v
[5-layer Bidirectional FlowGNN]
Input node hidden                          384
                                              |
Forward directed attention                 384
Backward directed attention                384
  -> concat: 384 + 384                     768
  -> directional fusion: 768 -> 384
                                              |
Layer residual:
  h <- h_prev + 0.5 * (h_new - h_prev)
Initial-feature injection at every layer:
  h <- h + 0.05 * h0
                                              |
Repeat for 5 layers
                                              |
Local node hidden                          384
                                              |
                 +----------------------------+---------------------+
                 |                                                  |
                 v                                                  v
        Source node 384                                  Destination node 384

All local nodes 384
  -> Set2Set, 3 processing steps
  -> raw graph embedding 768
  -> global projection: 768 -> 384
  -> graph context for each edge 384

[Edge descriptor]
Source local node                           384
Destination local node                      384
Projected global graph context              384
GNN edge embedding                          384
                                              |
Total descriptor: 384 * 4                 1536

No Property Role16 direct concat
No target hidden adapter
                                              |
                                              v
[Shared edge decoder]
1536 -> 768 -> 384

Each stage:
  Linear -> LayerNorm -> GELU -> Dropout(0.1)

Decoder residual is disabled because
intermediate width 768 differs from shared width 384.
                                              |
Shared edge latent                          384
                                              |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
Condition branch      Fraction branch       Mass branch
384 -> 96 -> 2        384 -> 128 -> 7       384 -> 96 -> 1
  Temp, Pres            H2O, H2, CH4,         Mass_Flow
                        CO2, CO, O2, N2
                              |
                        softmax(logits / 0.5)

[Detached Level-2 volume branch]
Temp                                  1
Pres                                  1
Fractions                             7
Mass_Flow                             1
                                      |
Level-1 prediction total             10
  -> detach()
  -> 10 -> 32 -> 1
  -> Vol_Flow

[Direct model output]
Temp + Pres                            2
Fractions                              7
Mass_Flow                              1
Vol_Flow                               1
                                      |
Main stream direct output             11D

[Not predicted]
Mole_Flow: excluded from the head, supervised loss, and final 11D output
Density: no direct prediction head
Enthalpy: no direct prediction head
```

## Training update structure

For each graph sample:

1. Average all valid non-target edge groups and perform one optimizer update.
2. Perform one independent optimizer update for each target edge group.
3. Perform one joint node update using the supervised anchor.

The node update keeps:

- relative mass-balance PINN
- relative component-balance PINN
- relative atom-balance PINN
- A2 joint supervised anchor

Energy PINN, Density loss, Enthalpy loss, and separate Volume PINN loss are
disabled in this model.

## PINN schedule

- Epochs 1-5: node PINN multiplier 0.0
- Epochs 6-7: node PINN multiplier 0.5
- Epoch 8 onward: node PINN multiplier 1.0

The edge supervised loss remains active throughout training.
