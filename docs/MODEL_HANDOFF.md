# Model Handoff

The canonical model-structure document for the current project state is:

```text
docs/MODEL_260805.md
```

This is the finalized R1-HX-F64 model. `docs/MODEL_260716.md` remains as a
historical model document.

Older PI grouped-head defaults, single-process held-out transfer notes, and
pre-260716 ablation descriptions have been removed from this handoff to avoid
conflicting guidance.

Use `docs/MODEL_260716.md` for the current final model recipe:

- `configs/experiment/pinn/model_260716_pinnbal_*_f01.yaml`
- full100k outer5 fold-1 grouped split basis
- sample-hybrid target-edge update mode
- 12D direct edge-all PI grouped-property output plus `rho_pred`/`h_pred`
- 3-flow prediction head: `Vol_Flow`, `Mole_Flow`, `Mass_Flow`
- softmax fraction head with log-space fraction loss
- scaled-log Mass_Flow transform with scale `2.0`
- target-edge supervised weight `40.0`
- optional Target Branch Hidden Adapter ablation in
  `model_260716_full100k_outer5_epoch2pct_val1000_targetfocus_samplehybrid_targethiddenadapter.yaml`
- stabilized node-level PINN recipe with true-consistency and prediction-residual guards
- epoch-balanced 2% train sampling with scheduled predefined hard-sample
  oversampling and 1000-sample validation sampling
- process-pair 5-fold unseen zero-shot / transfer / scratch workflow

Historical implementation notes remain in the git history and in older handoff
documents, but this file intentionally points to the current canonical model
document only.
