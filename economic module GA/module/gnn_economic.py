"""GNN-to-economic evaluation adapter for P01--P10.

The adapter turns one ``Process_Main`` operating-point row into a graph,
runs the existing trained edge-stream GNN, decodes the predicted 10D streams,
and passes only those predictions to :func:`objective.evaluate`.

``Process_Main`` contains product-result columns for several processes.  They
are direct response proxies, not controllable operating conditions.  The
default ``masked_proxy`` policy replaces their graph inputs with a missing
value/mask before inference.  ``unsafe`` exists only to reproduce legacy
behaviour and requires an explicit acknowledgement in the CLI runner.
"""

from __future__ import annotations

import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
for _entry in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from process_graph.data.tabular_dataset import ProcessGraphTabularDataset  # noqa: E402
from process_graph.experiment.config_builders import build_task_specs, model_yaml_to_encoder_config  # noqa: E402
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.experiment.pi_mass_flow import assert_checkpoint_pi_mass_flow_compatible  # noqa: E402
from process_graph.experiment.target_edge_10d_metrics import extract_main_stream_metric_tensors  # noqa: E402
from process_graph.models import ProcessSurrogateModel  # noqa: E402
from scripts.train_process_surrogate import _apply_runtime_overrides, _collate_for_experiment  # noqa: E402

import objective
from process_stream_map import PROCESS_STREAM_MAP


DEFAULT_CONFIG = "configs/experiment/pinn/model_260805_10d_frac1.yaml"
DEFAULT_CHECKPOINT = (
    "outputs/final_paper_260805_10d_frac1/proposed_joint_10d_clean/All/fold_01/"
    "checkpoints/process_kfold_All_F01-20260813-050337/best.pt"
)
DEFAULT_RUNTIME_OVERRIDES = (
    "outputs/final_paper_260805_10d_frac1/proposed_joint_10d_clean/All/fold_01/"
    "runtime_overrides.json"
)

ECONOMIC_10D = (
    "Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2", "Frac_CO",
    "Frac_O2", "Frac_N2", "Mass_Flow",
)
FLOW_TOKENS = ("FLOW", "VOL", "MOLE")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _economic_stream_keys(process_number: int) -> set[str]:
    """All non-null stream keys referenced by the costing topology."""
    keys: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, str):
            keys.add(value)
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(PROCESS_STREAM_MAP[f"P{process_number:02d}"])
    return keys


def _economic_stream_key(process_number: int, predicted_key: str) -> str:
    """Restore per-process numeric stream spelling (e.g. GNN ``06`` -> P03 ``6``)."""
    key = str(predicted_key).strip()
    expected = _economic_stream_keys(process_number)
    if key in expected:
        return key
    if key.isdigit():
        candidates = [item for item in expected if item.isdigit() and int(item) == int(key)]
        if len(candidates) == 1:
            return candidates[0]
    insensitive = [item for item in expected if item.upper() == key.upper()]
    if len(insensitive) == 1:
        return insensitive[0]
    return key


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state_dict") or payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint has no model_state_dict/model mapping.")
    return state


def _load_y_edge_scaler(checkpoint_path: Path, explicit_path: str | Path | None) -> tuple[Path, Mapping[str, Any]]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(_resolve(explicit_path))
    # .../<fold>/checkpoints/<run>/best.pt -> .../<fold>/<run>/y_edge_scaler.pt
    candidates.append(checkpoint_path.parent.parent.parent / checkpoint_path.parent.name / "y_edge_scaler.pt")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or any(payload.get(k) is None for k in ("mean", "std", "columns")):
            raise RuntimeError(f"Invalid y_edge_scaler payload: {candidate}")
        return candidate, payload
    raise FileNotFoundError("Cannot find y_edge_scaler.pt paired with checkpoint: " + ", ".join(map(str, candidates)))


def process_ids(values: str | Iterable[int | str]) -> list[int]:
    """Parse ``all`` or a comma-separated process list into sorted P01--P10 IDs."""
    if isinstance(values, str):
        if values.strip().lower() == "all":
            return list(range(1, 11))
        values = [item.strip() for item in values.split(",") if item.strip()]
    parsed: list[int] = []
    for value in values:
        text = str(value).upper().replace("PROCESS", "").replace("P", "").strip()
        number = int(text)
        if not 1 <= number <= 10:
            raise ValueError(f"Process must be between 1 and 10, got {value!r}.")
        parsed.append(number)
    return sorted(set(parsed))


class GNNEconomicAdapter:
    """Reuse one all-process checkpoint for GNN stream inference and economics."""

    def __init__(
        self,
        *,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        config: str | Path = DEFAULT_CONFIG,
        runtime_overrides: str | Path | None = DEFAULT_RUNTIME_OVERRIDES,
        y_edge_scaler: str | Path | None = None,
        device: str | None = None,
        template_id: int = 5000,
        input_policy: str = "masked_proxy",
    ) -> None:
        self.checkpoint_path = _resolve(checkpoint)
        self.config_path = _resolve(config)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Experiment config not found: {self.config_path}")
        self.input_policy = str(input_policy).strip().lower()
        if self.input_policy not in {"masked_proxy", "unsafe"}:
            raise ValueError("input_policy must be 'masked_proxy' or 'unsafe'.")

        self.experiment = load_experiment_config(self.config_path)
        self.runtime_overrides_path: Path | None = None
        if runtime_overrides:
            candidate = _resolve(runtime_overrides)
            if not candidate.is_file():
                raise FileNotFoundError(f"Runtime overrides file not found: {candidate}")
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError(f"Runtime overrides must be a JSON object: {candidate}")
            _apply_runtime_overrides(self.experiment, payload)
            self.runtime_overrides_path = candidate

        checkpoint_payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint_payload, Mapping):
            raise TypeError(f"Checkpoint payload must be a mapping: {self.checkpoint_path}")
        oper = checkpoint_payload.get("oper_normalizer")
        if not isinstance(oper, Mapping) or oper.get("mean") is None or oper.get("std") is None:
            raise RuntimeError("Checkpoint lacks train-split operating normalizer.")
        self.oper_mean = torch.as_tensor(oper["mean"], dtype=torch.float32)
        self.oper_std = torch.as_tensor(oper["std"], dtype=torch.float32)
        self.oper_std = torch.where(self.oper_std == 0, torch.ones_like(self.oper_std), self.oper_std)
        self.y_edge_scaler_path, self.y_edge_scaler = _load_y_edge_scaler(self.checkpoint_path, y_edge_scaler)
        assert_checkpoint_pi_mass_flow_compatible(
            checkpoint_payload, train_cfg=self.experiment.train, checkpoint_path=self.checkpoint_path
        )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ProcessSurrogateModel(
            encoder_config=model_yaml_to_encoder_config(self.experiment.model, self.experiment.data),
            task_specs=build_task_specs(self.experiment.model, self.experiment.data),
        ).to(self.device)
        self.model.load_state_dict(_checkpoint_state(checkpoint_payload), strict=True)
        self.model.eval()
        self._collate = _collate_for_experiment(self.experiment.data)
        self._work = tempfile.TemporaryDirectory(prefix="gnn_economic_")
        self._datasets: dict[int, ProcessGraphTabularDataset] = {}
        self._main_frames: dict[int, pd.DataFrame] = {}
        self._template_rows: dict[int, pd.Series] = {}
        self._safe_columns: dict[int, list[str]] = {}
        self._proxy_columns: dict[int, list[str]] = {}
        self.template_id = int(template_id)

    def close(self) -> None:
        self._work.cleanup()

    def __enter__(self) -> "GNNEconomicAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _process_label(process_id: int | str) -> tuple[int, str]:
        number = process_ids([process_id])[0]
        return number, f"Process{number}"

    def _prepare_process(self, process_id: int | str) -> int:
        number, label = self._process_label(process_id)
        if number in self._datasets:
            return number
        main_path = PROJECT_ROOT / "data" / "main_data" / f"{number}.Process_Main.csv"
        if not main_path.is_file():
            raise FileNotFoundError(f"Process_Main CSV not found: {main_path}")
        frame = pd.read_csv(main_path)
        if "ID" not in frame.columns or frame.empty:
            raise ValueError(f"{main_path} must contain non-empty ID rows.")
        selected = frame.loc[pd.to_numeric(frame["ID"], errors="coerce") == self.template_id]
        if selected.empty:
            selected = frame.iloc[[len(frame) // 2]]
        template = selected.iloc[0].copy()
        template["process_id"] = label

        staging = Path(self._work.name) / f"{label}.csv"
        pd.DataFrame([template]).to_csv(staging, index=False)
        dataset = ProcessGraphTabularDataset(
            staging,
            self.experiment.data,
            self.experiment.project_root,
            split_filter=None,
            oper_mean=self.oper_mean,
            oper_std=self.oper_std,
        )
        # A per-process Process_Main file naturally lacks target columns owned
        # by other process schemas (and P02 has a legacy CO2 spelling).  The
        # dataset still constructs fixed-task labels before the model forward;
        # add absent label columns as masked NaN values so they cannot enter the
        # graph yet cannot abort inference.
        if dataset._v3_answers is not None:
            target_rows = dataset._v3_answers[dataset._v3_answers["process_id"] == number]
            for raw_target in target_rows.get("target_column", pd.Series(dtype=str)).tolist():
                target_column = str(raw_target).strip()
                match = re.match(r"^[0-9.]+\s*\*\s*(\w+)$", target_column)
                target_column = match.group(1) if match else target_column
                if target_column and target_column not in template.index:
                    template[target_column] = np.nan
        dataset.frame = pd.DataFrame([template])
        spec = dataset._load_spec(label)
        safe, proxy = set(), set()
        for node in spec.nodes.values():
            for cell in node.features.values():
                if getattr(cell, "kind", "") != "reference":
                    continue
                column = str(getattr(cell, "value", "")).strip()
                if not column or column not in frame.columns:
                    continue
                (proxy if node.role == "sink" else safe).add(column)
        self._datasets[number] = dataset
        self._main_frames[number] = frame
        self._template_rows[number] = template
        self._safe_columns[number] = sorted(safe - proxy)
        self._proxy_columns[number] = sorted(proxy)
        return number

    def input_audit(self, process_id: int | str) -> dict[str, object]:
        number = self._prepare_process(process_id)
        return {
            "process_id": f"P{number:02d}",
            "input_policy": self.input_policy,
            "safe_condition_columns": self._safe_columns[number],
            "response_proxy_columns": self._proxy_columns[number],
            "proxy_masked": self.input_policy == "masked_proxy",
        }

    def decision_columns(self, process_id: int | str, *, include_feed_flow: bool = False) -> list[str]:
        """Return variable, non-sink GNN conditioning columns with empirical range."""
        number = self._prepare_process(process_id)
        frame = self._main_frames[number]
        columns: list[str] = []
        for column in self._safe_columns[number]:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty or not math.isfinite(float(values.min())) or float(values.max()) <= float(values.min()):
                continue
            if not include_feed_flow and any(token in column.upper() for token in FLOW_TOKENS):
                continue
            columns.append(column)
        return columns

    def bounds(self, process_id: int | str, columns: Iterable[str], *, low: float = 0.01, high: float = 0.99) -> dict[str, tuple[float, float]]:
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("Quantiles must satisfy 0 <= low < high <= 1.")
        number = self._prepare_process(process_id)
        frame = self._main_frames[number]
        result: dict[str, tuple[float, float]] = {}
        for column in columns:
            if column not in frame.columns:
                raise KeyError(f"P{number:02d}: unknown Process_Main column {column!r}.")
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            lo, hi = float(values.quantile(low)), float(values.quantile(high))
            if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
                raise ValueError(f"P{number:02d}: unusable bounds for {column!r}: {(lo, hi)}")
            result[str(column)] = (lo, hi)
        return result

    def _candidate_row(self, number: int, values: Mapping[str, float] | None) -> pd.DataFrame:
        row = self._template_rows[number].copy()
        for column, value in (values or {}).items():
            if column not in row.index:
                raise KeyError(f"P{number:02d}: unknown Process_Main column {column!r}.")
            value_float = float(value)
            if not math.isfinite(value_float):
                raise ValueError(f"P{number:02d}: non-finite candidate value for {column!r}.")
            row[column] = value_float
        if self.input_policy == "masked_proxy":
            for column in self._proxy_columns[number]:
                row[column] = np.nan
        return pd.DataFrame([row])

    def predict_streams(self, process_id: int | str, values: Mapping[str, float] | None = None) -> dict[str, dict[str, object]]:
        """Return full economic 10D stream dict predicted by the configured GNN."""
        number = self._prepare_process(process_id)
        dataset = self._datasets[number]
        dataset.frame = self._candidate_row(number, values)
        dataset._rows = [0]
        dataset._source_row_indices = [0]
        batch = self._collate([dataset[0]])
        batch_data = {key: value.to(self.device) for key, value in batch.model_kwargs.items()}
        task_inputs = {
            name: {key: value.to(self.device) for key, value in payload.items()}
            for name, payload in batch.task_inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(batch_data, task_inputs=task_inputs)
            pred, _, _, names = extract_main_stream_metric_tensors(
                outputs=outputs,
                targets_raw={key: value.to(self.device) for key, value in batch.targets.items()},
                target_masks={key: value.to(self.device) for key, value in batch.target_masks.items()},
                train_cfg=self.experiment.train,
                data_cfg=self.experiment.data,
                edge_target_columns=batch.edge_target_columns,
                normalizer=self.y_edge_scaler,
            )
        if tuple(names)[:len(ECONOMIC_10D)] != ECONOMIC_10D:
            raise RuntimeError(f"Unexpected decoded GNN stream ordering: {names}")
        if batch.edge_export_meta is None:
            raise RuntimeError("GNN batch has no edge export metadata.")
        physical = pred.detach().cpu().numpy()
        keys = list(batch.edge_export_meta.main_data_stream_key)
        streams: dict[str, dict[str, object]] = {}
        for stream_key, vector in zip(keys, physical):
            key = _economic_stream_key(number, str(stream_key))
            if not key:
                continue
            payload = {
                "T_c": float(vector[0]),
                "P_bar": float(vector[1]),
                "x": [float(v) for v in vector[2:9]],
                "mdot_kgh": float(vector[9]),
            }
            if not (math.isfinite(payload["T_c"]) and math.isfinite(payload["P_bar"]) and math.isfinite(payload["mdot_kgh"]) and np.isfinite(payload["x"]).all()):
                raise RuntimeError(f"P{number:02d}: GNN produced a non-finite stream {key!r}.")
            if key in streams:
                previous = streams[key]
                if not np.allclose(
                    [previous["T_c"], previous["P_bar"], previous["mdot_kgh"], *previous["x"]],
                    [payload["T_c"], payload["P_bar"], payload["mdot_kgh"], *payload["x"]],
                    rtol=1e-5, atol=1e-6,
                ):
                    raise RuntimeError(f"P{number:02d}: duplicate GNN predictions disagree for stream {key!r}.")
                continue
            streams[key] = payload
        return streams

    def evaluate(self, process_id: int | str, values: Mapping[str, float] | None = None, **objective_kwargs: object) -> dict[str, object]:
        number, label = self._process_label(process_id)
        streams = self.predict_streams(number, values)
        result = objective.evaluate(streams, f"P{number:02d}", **objective_kwargs)
        return {
            **result,
            "candidate": {str(k): float(v) for k, v in (values or {}).items()},
            "input_audit": self.input_audit(number),
            "stream_count": len(streams),
            "gnn": {
                "checkpoint": str(self.checkpoint_path),
                "config": str(self.config_path),
                "runtime_overrides": str(self.runtime_overrides_path or ""),
                "y_edge_scaler": str(self.y_edge_scaler_path),
                "device": str(self.device),
            },
        }
