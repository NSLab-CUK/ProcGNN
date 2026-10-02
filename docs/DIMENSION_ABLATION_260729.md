# Model 260729 Dimension Ablation

## 1. Scope

This ablation keeps the implemented 11D hierarchical prediction semantics:

```text
Node/edge input
-> 5-layer bidirectional FlowGNN
-> [source node, destination node, global context, GNN edge embedding]
-> shared edge decoder
-> Level 1: Temp, Pres, seven fractions, Mass_Flow
-> Level 2: detached Level-1 10D prediction -> Vol_Flow
```

The following are not reintroduced:

- Density, Enthalpy, directly predicted Mole_Flow
- Energy PINN
- Property Role16 direct concat
- GNN/shared/global features in the Level-2 input

Mass, component, and atom node PINN remain enabled. Fractions remain a
temperature-0.5 softmax. Target weight is 5 in every E0-E8 config.

### A1/A2 inheritance

- A1 (`property_stream_role`, Role16 direct concat) is intentionally not
  inherited. The hierarchical 11D design removes that direct shortcut and
  retains only the generic stream-role embedding in the edge descriptor.
- A2 (`joint_with_supervised_anchor`) is inherited by every E0-E8 config:
  supervised anchor weight `1.0`, node PINN outer weight `0.05`, and target
  weighting disabled for the anchor.
- A2's old Enthalpy term (`lambda_h=0.02`) is not inherited because this model
  does not predict Enthalpy and Energy PINN is disabled.

## 2. Current E0 Audit

### Cardinalities

| Category | Cardinality / observed use | E0 embedding |
|---|---:|---:|
| Node role | 5 | 96 |
| Unit type | 18 | 64 |
| Generic stream role | vocabulary 10, observed 9 | 64 |
| Stream name ID | vocabulary buckets 512 | 96 |

The mapping CSV contains 250 process-edge rows and 52 distinct normalized
stream names. CRC32 modulo 512 maps them to 50 distinct buckets, so two bucket
collisions exist. This is not a raw process-ID input, but it can still act as a
stream-name lookup shortcut.

### Actual forward dimensions

| Stage | E0 runtime shape / width |
|---|---:|
| Operating values | 13 |
| Operating mask | 13 |
| Node categorical width | 96 + 64 = 160 |
| Node continuous width | 13 + 13 = 26 |
| Node fusion input | 186 |
| Initial / GNN node hidden | 512 |
| Edge categorical width | 64 + 96 = 160 |
| Raw structural width | 3 |
| Edge fusion input | 163 |
| GNN edge embedding | 512 |
| FlowGNN layers | 5 |
| Attention | one scalar head per direction |
| Set2Set output | 1024 = 2H |
| Edge descriptor | 2560 = H + H + 2H + H |
| Shared decoder | 2560 -> 384 -> 384 |
| Condition branch | 384 -> 128 -> 2 |
| Fraction branch | 384 -> 128 -> 7 |
| Mass branch | 384 -> 128 -> 1 |
| Volume branch | 10 -> 64 -> 1 |
| Final prediction | 11 |

FlowGNN is not multi-head attention. It has separate forward and backward
attention branches, each producing one scalar attention score per edge. The
new config therefore exposes `attention_num_heads`, but currently validates
that it is exactly 1 instead of pretending that the implementation is
multi-head.

### Existing residual path

Each FlowGNN layer applies:

```text
h <- h_prev + 0.5 * (h_new - h_prev)
h <- h + 0.05 * h0
```

The shared E0 decoder applies:

```text
s0 = Dropout(GELU(LayerNorm(Linear(descriptor))))
s1 = Dropout(GELU(LayerNorm(Linear(s0))))
shared = LayerNorm(s0 + s1)
```

E1 also preserves this decoder residual. E2-E7 use an intermediate dimension
different from the final shared dimension, so they use
`descriptor -> intermediate -> shared` without this residual.

## 3. Problems Found

### Oversized categorical embeddings

Five node-role categories receive 96 dimensions and ten generic stream roles
receive 64 dimensions. The capacity is large relative to category count and
makes identity lookup easier than learning continuous operating variation.

### Modality imbalance

The E0 node encoder concatenates 160 categorical dimensions with 26 operating
dimensions. The E0 edge encoder concatenates 160 categorical dimensions with
only three structural values. There was no modality-specific projection.

### Global dominance

The unprojected 1024D Set2Set vector occupies 40% of the 2560D descriptor and
is repeated for every edge in a graph.

### Abrupt compression

E0 compresses 2560D to 384D in one projection. Its residual refinement is
stable, but it cannot recover modality information lost in the first 6.7x
compression.

### Stream shortcut risk

The ID represents a normalized stream name hashed into 512 buckets. It is not
a process number, but a 96D lookup can still favor stream-name means over
within-stream sample variation.

## 4. Implemented Dimension Controls

The new `model.dimension_design` block controls:

```yaml
dimension_design:
  node:
    role_embedding_dim: 32
    unit_embedding_dim: 24
    operating_hidden_dim: 64
    hidden_dim: 384
  edge:
    role_embedding_dim: 24
    stream_id_embedding_dim: 32
    structural_hidden_dim: 16
    hidden_dim: 384
    use_stream_id: true
    force_unknown_stream_id: false
  gnn:
    hidden_dim: 384
    attention_num_heads: 1
  global:
    project_output: true
    output_dim: 384
  decoder:
    intermediate_dim: 768
    shared_hidden_dim: 384
    residual: false
  level1:
    condition_hidden_dim: 96
    fraction_hidden_dim: 128
    mass_hidden_dim: 96
  level2:
    hidden_dim: 32
    detach_level1_predictions: true
```

New-mode node continuous encoding is:

```text
[13 operating values, 13 masks]
-> Linear
-> configured activation
-> LayerNorm
```

New-mode edge structural encoding is:

```text
3 structural values
-> Linear
-> configured activation
-> LayerNorm
```

Node and GNN hidden widths must match. Unsupported multi-head values and
invalid residual/intermediate combinations raise explicit errors. Explicit
dimension configs bypass the old all-process minimum-hidden override, so E5
really runs at H=256.

An edge hidden width different from node H is supported. The attention scorer
and message path add learned projections when edge width differs, while the
hierarchical descriptor uses the actual edge width.

## 5. E0-E7 Matrix

| Exp | Node emb role/unit | Oper | H | Edge emb role/ID | Struct | Edge H | Global | Descriptor | Decoder | Branch C/F/M | L2 | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 96/64 | raw 26 | 512 | 64/96 | raw 3 | 512 | 1024 | 2560 | 384->384 | 128/128/128 | 64 | 32,503,275 |
| E1 | 32/32 | 64 | 512 | 32/32 | 16 | 512 | 512 | 2048 | 384->384 | 128/128/128 | 64 | 32,727,243 |
| E2 | 32/24 | 64 | 384 | 24/32 | 16 | 384 | 384 | 1536 | 768->384 | 96/128/96 | 32 | 19,434,123 |
| E3 | 32/24 | 64 | 384 | 24/16 | 16 | 384 | 384 | 1536 | 768->384 | 96/128/96 | 32 | 19,419,787 |
| E4 | 32/24 | 64 | 384 | 24/none | 16 | 384 | 384 | 1536 | 768->384 | 96/128/96 | 32 | 19,405,451 |
| E5 | 16/16 | 32 | 256 | 16/16 | 16 | 256 | 256 | 1024 | 512->256 | 64/96/64 | 32 | 8,648,411 |
| E6 | 32/32 | 64 | 512 | 32/32 | 16 | 512 | 512 | 2048 | 1024->512 | 128/192/128 | 64 | 34,499,915 |
| E7 | 32/24 | 64 | 384 | 24/32 | 16 | 384 | 384 | 1536 | 512->256 | 64/96/64 | 32 | 18,809,515 |

E1 has fewer embedding parameters than E0 but slightly more total parameters
because it adds operating, structural, and 1024->512 global projections.

## 6. Fixed Training Conditions

Every config keeps:

- 60/20/20 outer fold-1 manifests
- 30 maximum epochs and patience 5
- learning rate `1e-4`
- target weight 5
- sample-hybrid non-target mean plus per-target updates
- 2% base epoch sampling, hard fill to 2000, hard fraction 40%
- random validation sample of 1000 every epoch
- node-only PINN schedule: epochs 1-5 zero, 6-7 half, 8+ full
- node Mass 0.75, Component `1.5e-7`, Atom 0.2, Energy 0
- fraction log supervision and softmax temperature 0.5
- `val_target_mean_r2` checkpoint monitor

## 7. Diagnostics

Training logs now include:

- parameter counts by node encoder, GNN, edge encoder, pool, global
  projection, shared decoder, and each branch
- training seconds, graph samples/s, optimizer steps/s
- peak allocated GPU memory
- validation seconds
- detailed gradient norms for node role/unit, operating encoder, edge role,
  Stream ID, structural encoder, GNN, global pool/projection, shared decoder,
  and each output branch

`--edge-all-forward-debug` additionally writes representation statistics to
`first_batch_debug.json`:

- mean, standard deviation, mean and p95 row norm
- mean feature-wise standard deviation
- dead and near-zero variance ratios
- entropy effective rank
- initial node, every GNN layer, final local node, edge, global, and shared
  edge latent

To test Stream ID shortcut without retraining, load the same checkpoint in
evaluation mode and set:

```yaml
dimension_design:
  edge:
    force_unknown_stream_id: true
```

This preserves checkpoint shapes but sends every stream through bucket 0.

## 8. Verification

- 21 focused tests pass.
- Eight E0-E7 configs parse and instantiate.
- All eight real-data forward smoke tests pass with prediction, target, and
  mask shape `(21, 11)`.
- E2 end-to-end smoke completes training updates, node PINN routing,
  validation, best/last checkpoint save, checkpoint reload, final target-edge
  evaluation, all-edge evaluation, plots, and metrics export.
- A broader related suite produced 50 passes. Its nine failures are missing
  deleted legacy YAML fixtures, not failures in the new dimension path.

The E2 smoke used only two samples, so its R2 values are intentionally
meaningless and must not be used for model selection.

## 9. Configs

```text
configs/experiment/pinn/head_dimension_260729/e0_current512.yaml
configs/experiment/pinn/head_dimension_260729/e1_embed512.yaml
configs/experiment/pinn/head_dimension_260729/e2_balanced384.yaml
configs/experiment/pinn/head_dimension_260729/e3_balanced384_sid16.yaml
configs/experiment/pinn/head_dimension_260729/e4_balanced384_no_sid.yaml
configs/experiment/pinn/head_dimension_260729/e5_compact256.yaml
configs/experiment/pinn/head_dimension_260729/e6_large512.yaml
configs/experiment/pinn/head_dimension_260729/e7_balanced384_narrow.yaml
configs/experiment/pinn/head_dimension_260729/e8_compact256_midhead_focus.yaml
```

### E8 follow-up

E8 preserves the E5 256D node/edge/GNN/global body and changes only the
decoder/head and active hard-sample allocation:

| Block | E5 | E8 |
|---|---:|---:|
| Edge descriptor | 1024 | 1024 |
| Decoder | 1024 -> 512 -> 256 | 1024 -> 768 -> 384 |
| Condition branch | 64 | 96 |
| Fraction branch | 96 | 128 |
| Mass branch | 64 | 96 |
| Volume branch | 32 | 32 |

The epoch still contains 2% base sampling plus hard fill to 2000 unique
samples. The base 1200 keeps the E5 uniform, sparse-positive, quantile-balanced,
and Mass_Flow-tail mixture. The 800 hard-fill slots are blended as follows:

- 300 Target Frac_CO samples, including the dominant `P03_E014` and retained
  E5 CO coverage on `P07_E017`
- 350 Target Mass_Flow tail samples across `P04_E009`, `P06_E013`,
  `P01_E015`, `P09_E022`, and `P10_E015`
- 150 retained E5 Target Frac_CH4 samples
- no dedicated Target Frac_CO2 hard-fill samples; CO2 remains covered by the
  base sparse-positive and quantile-balanced sampling

This directs 81.25% of the extra quota toward CO and Mass_Flow, while retaining
18.75% for CH4 regression protection. The wider fraction head is intended to
reduce the all-edge H2O conditional-mean compression; target H2O and CO2 are
not hard-sampled because their target-edge performance was already strong.

## 10. Selection Rule

No final winner is claimed before full runs finish. Compare the same best
checkpoint policy using:

1. target edge-internal pooled and equal-edge macro R2
2. all-edge property R2
3. target aggregate R2
4. CH4, CO, CO2, Mass_Flow, and Vol_Flow separately
5. parameters, peak memory, and samples/s
6. Stream ID normal-vs-unknown inference sensitivity

E2 is the primary balanced candidate, E5 tests whether the present model is
substantially oversized, and E6 tests whether 512D GNN capacity remains useful
after removing embedding/global imbalance.
