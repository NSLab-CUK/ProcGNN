# Stream-Key Join Fix (260731)

## Scope

This change fixes graph-to-`Process_Streams.csv` joins without changing model
architecture, topology, sampling, loss, PINN, target weights, or metrics.

## Canonicalization

All runtime joins now use one shared rule:

- Missing values become an empty key.
- Unicode whitespace is collapsed and trimmed.
- Integer-like numeric keys lose leading zeros.
- `01`, `001`, `1`, `1.0`, and `" 01 "` all become `1`.
- Alphanumeric keys such as `PROD`, `R1-FEED`, and `P03_E014` are preserved.
- Process identifiers such as `3`, `03`, `P3`, `P03`, and `Process3` become
  `P03`.

The same rule is used by graph construction, fast scaler fitting, target
metrics, and sampling metadata.

## Safety Behavior

- More than one CSV row mapping to the same canonical key in one sample raises
  a collision error.
- A non-empty canonical graph stream key is required supervision. A missing
  required CSV row raises an error containing process, sample, edge, raw key,
  canonical key, available keys, and CSV path.
- Empty canonical keys remain optional. This preserves the intentionally
  unmapped P01 edge and F2 context edges with zero masks.
- Stream-key canonicalization is part of the scaler-cache fingerprint. Old
  scaler caches cannot be reused after this fix.

## Full Audit

Audit command:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 \
python scripts/audit_stream_key_joins.py \
  --output-dir outputs/stream_key_join_audit_260731
```

Results:

| Item | Result |
|---|---:|
| Required joins expected | 2,449,970 |
| Required joins after canonicalization | 2,449,968 |
| Target joins | 189,998 / 189,998 |
| Canonical-key collisions | 0 |
| P03 joins recovered from formatting mismatch | 90,000 |
| P03 requested 9-edge coverage | 90,000 / 90,000 |
| P03 requested 9-edge property-value coverage | 1,260,000 / 1,260,000 |

The P03 9-edge property check covers Temp, Pres, Vol_Flow, Mole_Flow,
Mass_Flow, seven fractions, Enthalpy, and Density.

## Remaining Source-Data Defect

The audit found one issue that is not a key-format mismatch:

- Process: `P03`
- Sample: `7593`
- Missing source rows: stream `11` (`P03_E015`) and stream `10` (`P03_E017`)
- Source file: `data/main_data_Streams/3.Process_Streams.csv`

The corresponding two CSV lines are completely blank. No trustworthy
ground-truth values exist in this checkout, so the implementation does not
guess, interpolate, copy another stream, or silently insert zeros. Accessing
that sample now fails with a detailed error.

To obtain strict 100% required coverage, restore those two rows from the
authoritative process export and rerun the audit with `--strict`.

## Audit Artifacts

The audit writes:

- `process_join_summary.csv`
- `missing_required_edges.csv`
- `stream_key_raw_to_canonical.csv`
- `canonical_key_collisions.csv`
- `sample_join_failures.csv`
- `supervision_count_before_after.csv`
- `audit_summary.json`

under `outputs/stream_key_join_audit_260731`.

## Verification

- Stream-key unit and real-data integration tests: 26 passed.
- Train/validation/test collate, forward, loss, one backward, and metric
  accumulation smoke: passed.
- Known-feed F1/F2 regression tests: 16 passed.
- Sampler regressions: 10 passed, with two unrelated tests deselected because
  their referenced archived YAML files are absent.
- Target-v4 metric regressions: 14 passed.
