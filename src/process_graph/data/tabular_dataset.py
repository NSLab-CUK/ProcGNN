from __future__ import annotations

import math
import re
import sys
import warnings
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

from ..debug_ndjson import write_training_debug_event
from ..constants import (
    HX_ROLE_TO_IDX,
    OPER_FEATURE_SLOTS,
    ROLE_TO_IDX,
    STREAM_EDGE_FEATURE_SLOTS,
    STREAM_ROLE_TO_IDX,
    STREAM_ROLE_VOCAB,
    property_stream_role_id,
    UNIT_TO_IDX,
)
from ..experiment.schema import DECODER_CATEGORIES, DataConfig
from ..experiment.pi_mass_flow import transform_mass_flow_scalar
from ..known_feed import (
    KNOWN_FEED_NAMES,
    KNOWN_FEED_FEATURE_NAMES,
    IndependentFeedTopologyBinding,
    KnownFeedBinding,
    build_independent_feed_topology_bindings,
    known_feed_bindings,
    known_feed_feature_names,
    known_feed_ratio_values,
    operating_feature_names,
    parse_known_feed_condition,
)
from ..hx_pair import parse_hx_pair_relation
from ..feed_head import parse_feed_head_conditioning
from ..parser import parse_process_file
from ..resolver import build_graph_sample, build_incoming_node_map, resolve_feature_value
from ..schema import GraphSample, ProcessSpec, ResolveContext
from .stream_keys import (
    CanonicalStreamKeyCollisionError,
    STREAM_KEY_CANONICAL_COLUMN,
    build_canonical_stream_row_index,
    canonicalize_process_id,
    canonicalize_stream_key,
    missing_required_stream_key_error,
)


@dataclass
class YEdgeScalerFitResult:
    """Train-only StandardScaler over y_edge_true with y_edge_mask==1."""

    mean: torch.Tensor
    std: torch.Tensor
    columns: list[str]
    masked_value_counts_per_column: list[int]
    warnings: list[str]
    physical_quantiles: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class YEdgeLog1pColumnScalerFitResult:
    """Train-only StandardScaler over log1p(clamp_min(y_edge_true[column], 0))."""

    mean: torch.Tensor
    std: torch.Tensor
    column: str
    count: int
    warnings: list[str]


@dataclass
class EdgeBatchExportMeta:
    """Per-edge strings aligned with batched edge dimension (for CSV export)."""

    process_id: List[str]
    sample_id: List[str]
    dataset_split: List[str]
    canonical_edge_id: List[str]
    main_data_stream_key: List[str]
    stream_role: List[str]
    is_input_edge: List[float]
    is_output_edge: List[float]
    is_internal_edge: List[float]
    answer_task_names: List[str]
    src_node: List[str] = field(default_factory=list)
    dst_node: List[str] = field(default_factory=list)
    is_context: List[float] = field(default_factory=list)
    is_predictable: List[float] = field(default_factory=list)
    is_supervised: List[float] = field(default_factory=list)
    is_target: List[float] = field(default_factory=list)
    include_in_pinn: List[float] = field(default_factory=list)


@dataclass
class GraphBatch:
    model_kwargs: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]
    target_masks: Dict[str, torch.Tensor]
    task_inputs: Dict[str, Dict[str, torch.Tensor]]
    sample_meta: List[Dict[str, Any]]
    canonical_edge_ids: List[List[str]] | None = None
    # Per-sample dict: fixed-slot key -> global edge row in batched y_edge_pred / y_edge_true.
    answer_edge_pos: List[Dict[str, int]] | None = None
    answer_edge_ids: List[Dict[str, str]] | None = None
    edge_target_columns: List[str] | None = None
    edge_export_meta: EdgeBatchExportMeta | None = None


FIXED_SLOTS: Tuple[str, ...] = ("target", "tailgas")

SPECIES_TO_FRAC_SLOT = {
    "H2": "Frac_H2",
    "CO2": "Frac_CO2",
    "H2O": "Frac_H2O",
}


def _id_key(value: Any) -> str:
    return canonicalize_stream_key(value, source="sample_id")


def _sample_cache_key(process_id: Any, sample_id: Any) -> str:
    pid_text = str(process_id).strip()
    if pid_text and not pid_text.startswith("Process"):
        try:
            pid_text = f"Process{int(float(pid_text))}"
        except (TypeError, ValueError):
            pass
    return f"{pid_text}::{_id_key(sample_id)}"

# Process-aware fixed target columns (validated for datasets_v3).
# Keys: process_id -> {"target": <H2 column>, "tailgas": <CO2 column>}
PROCESS_FIXED_TARGET_COLUMNS: Dict[str, Dict[str, str]] = {
    "Process1": {"target": "PROD_H2_Mole", "tailgas": "FUELGAS_CO2_Mole"},
    "Process2": {"target": "PROD_H2_MoleFlow", "tailgas": "PROD_CO2_MoleFlow"},
    "Process3": {"target": "PROD_H2_Mole", "tailgas": "OUT_CO2_Mole"},
    "Process4": {"target": "PROD_H2_Mole", "tailgas": "EX_CO2_Mole"},
    "Process5": {"target": "PROD_H2_Mole", "tailgas": "Stream12_CO2_Mole"},
    "Process6": {"target": "PROD_H2_Mole", "tailgas": "Stream12_CO2_Mole"},
    "Process7": {"target": "PROD_H2_Mole", "tailgas": "PROD_CO2_Mole"},
    "Process8": {"target": "PROD_H2_Mole", "tailgas": "12_CO2_Mole"},
    "Process9": {"target": "PROD_H2_Mole", "tailgas": "EXHAUS_CO2_Mole"},
    "Process10": {"target": "PROD_H2_Mole", "tailgas": "OUT_CO2_Mole"},
}

# Process-aware fixed stream names for stream-edge readout selection.
# Keys: process_id -> {"target": <stream_name>, "tailgas": <stream_name>}
PROCESS_FIXED_EDGE_STREAM_NAMES: Dict[str, Dict[str, str]] = {
    "Process1": {"target": "PROD", "tailgas": "FUELGAS"},
    "Process2": {"target": "PROD", "tailgas": "PROD"},
    "Process3": {"target": "PROD", "tailgas": "OUT"},
    "Process4": {"target": "PROD", "tailgas": "EX"},
    "Process5": {"target": "PROD", "tailgas": "12"},
    "Process6": {"target": "PROD", "tailgas": "12"},
    "Process7": {"target": "PROD", "tailgas": "PROD"},
    "Process8": {"target": "PROD", "tailgas": "12"},
    "Process9": {"target": "PROD", "tailgas": "EXHAUS"},
    "Process10": {"target": "PROD", "tailgas": "OUT"},
}

# Explicit process/node mapping for Process_Main energy terms. HX Q columns are
# intentionally not connected here: node-level HX total balances use Q=0/W=0.
PROCESS_NODE_ENERGY_MAP: Dict[int, Dict[str, Dict[str, str | None]]] = {
    1: {
        "BURNER": {"q": "Q_BURNER", "w": None},
        "C1": {"q": None, "w": "W_C1"},
        "C2": {"q": None, "w": "W_C2"},
        "C3": {"q": None, "w": "W_C3"},
        "P1": {"q": None, "w": "W_P1"},
        "P2": {"q": None, "w": "W_P2"},
        "P3": {"q": None, "w": "W_P3"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "R3": {"q": "Q_R3", "w": None},
        "R4": {"q": "Q_R4", "w": None},
        "T1": {"q": None, "w": "W_T1"},
        "T2": {"q": None, "w": "W_T2"},
        "T3": {"q": None, "w": "W_T3"},
    },
    2: {
        "C1": {"q": None, "w": "W_C1"},
        "C2": {"q": None, "w": "W_C2"},
        "C3": {"q": None, "w": "W_C3"},
        "C4": {"q": None, "w": "W_C4"},
        "COOL1": {"q": "Q_COOL1", "w": None},
        "COOL2": {"q": "Q_COOL2", "w": None},
        "COOL4": {"q": "Q_COOL4", "w": None},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "R3": {"q": "Q_R3", "w": None},
    },
    3: {
        "BURNER": {"q": "Q_BURNER", "w": None},
        "C1": {"q": None, "w": "W_C1"},
        "C2": {"q": None, "w": "W_C2"},
        "F1": {"q": "Q_F1", "w": None},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "SEP1": {"q": "Q_SEP1", "w": None},
    },
    4: {
        "BURNER": {"q": "Q_BURNER", "w": None},
        "C1": {"q": None, "w": "W_C1"},
        "COOL1": {"q": "Q_COOL1", "w": None},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "R3": {"q": "Q_R3", "w": None},
        "SEP1": {"q": "Q_SEP1", "w": None},
    },
    5: {
        "C1": {"q": None, "w": "W_C1"},
        "COOL1": {"q": "Q_COOL1", "w": None},
        "F1": {"q": "Q_F1", "w": None},
        "HEAT1": {"q": "Q_HEAT1", "w": None},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "SEP1": {"q": "Q_SEP1", "w": None},
    },
    6: {
        "C1": {"q": None, "w": "W_C1"},
        "C2": {"q": None, "w": "W_C2"},
        "C3": {"q": None, "w": "W_C3"},
        "COOL1": {"q": "Q_COOL1", "w": None},
        "F1": {"q": "Q_F1", "w": None},
        "HEAT1": {"q": "Q_HEAT1", "w": None},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "R3_BURNER": {"q": "Q_R3", "w": None},
        "SEP1": {"q": "Q_SEP1", "w": None},
    },
    7: {
        "C1": {"q": None, "w": "W_C1"},
        "COOL1": {"q": "Q_COOL1", "w": None},
        "F1": {"q": "Q_F1", "w": None},
        "HEAT1": {"q": "Q_HEAT1", "w": None},
        "HEAT2": {"q": "Q_HEAT2", "w": None},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "R3": {"q": "Q_R3", "w": None},
    },
    8: {
        "C1": {"q": None, "w": "W_C1"},
        "COOL1": {"q": "Q_Cool1", "w": None},
        "COOL2": {"q": "Q_Cool2", "w": None},
        "COOL3": {"q": "Q_Cool3", "w": None},
        "F1": {"q": "Q_F1", "w": None},
        "HEAT1": {"q": "Q_Heat1", "w": None},
        "HEAT2": {"q": "Q_Heat2", "w": None},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "R3": {"q": "Q_R3", "w": None},
        "SEP1": {"q": "Q_SEP1", "w": None},
    },
    9: {
        "BURNER": {"q": "Q_Burner", "w": None},
        "C1": {"q": None, "w": "W_C1"},
        "C2": {"q": None, "w": "W_C2"},
        "C3": {"q": None, "w": "W_C3"},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "SEP1": {"q": "Q_SEP1", "w": None},
    },
    10: {
        "BURNER": {"q": "Q_BURNER", "w": None},
        "C1": {"q": None, "w": "W_C1"},
        "C2": {"q": None, "w": "W_C2"},
        "COOL1": {"q": "Q_COOL1", "w": None},
        "F1": {"q": "Q_F1", "w": None},
        "P1": {"q": None, "w": "W_P1"},
        "R1": {"q": "Q_R1", "w": None},
        "R2": {"q": "Q_R2", "w": None},
        "SEP1": {"q": "Q_SEP1", "w": None},
    },
}

P03_INCOMPLETE_BALANCE_NODES: set[str] = {"F1", "P1", "C2", "HX5", "MIX1"}


@dataclass
class GraphSampleRecord:
    graph: GraphSample
    slot_targets: Dict[str, float]
    slot_masks: Dict[str, float]
    slot_node_index: Dict[str, int]
    category_node_indices: Dict[str, List[int]]
    category_values: Dict[str, List[float]]
    sample_meta: Dict[str, Any]
    slot_edge_index: Dict[str, int] | None = None


def _to_float_mapping(row: pd.Series, reserved: set[str]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for key, value in row.items():
        if key in reserved:
            continue
        if pd.isna(value):
            continue
        try:
            payload[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


def _tokenize_identifier(text: str) -> List[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(text))
    return [token for token in re.split(r"[^A-Za-z0-9]+", normalized.upper()) if token]


def _canonical_process_number(process_id: str) -> int:
    normalized = canonicalize_process_id(process_id)
    if not normalized:
        raise ValueError(f"Cannot parse process number from process_id={process_id!r}")
    return int(normalized[1:])


def _v3_supervision_scalar(row: pd.Series, target_column: str) -> tuple[float, float]:
    """Resolve supervision from dataset row; literal or '0.8*COL'. Returns (value, mask); mask=0 if missing/NaN."""
    tc = str(target_column).strip()
    if tc in row.index:
        v = row.get(tc)
        if v is not None and not pd.isna(v):
            return float(v), 1.0
        return 0.0, 0.0
    m = re.match(r"^([0-9.]+)\s*\*\s*(\w+)$", tc)
    if m:
        col = m.group(2)
        if col in row.index:
            v = row.get(col)
            if v is not None and not pd.isna(v):
                return float(m.group(1)) * float(v), 1.0
        return 0.0, 0.0
    raise KeyError(
        f"v3 target_column {tc!r} not found in dataset row (expected literal column or pattern like '0.8*PROD_H2_MoleFlow')."
    )


V3_SLOT_TO_TASK_NAME = {"target": "target_h2", "tailgas": "tailgas_co2"}
V3_EDGE_STRUCT_ROLE_COLUMNS: Tuple[str, ...] = (
    "is_input_edge",
    "is_output_edge",
    "is_internal_edge",
)
V3_EDGE_STRUCT_JOIN_COLUMNS: Tuple[str, ...] = (
    "has_main_data_stream_key",
    "edge_feature_mask",
)
V3_EDGE_STRUCT_FEATURE_COLUMNS: Tuple[str, ...] = V3_EDGE_STRUCT_ROLE_COLUMNS + V3_EDGE_STRUCT_JOIN_COLUMNS


def _edge_struct_feature_columns(data_cfg: DataConfig) -> Tuple[str, ...]:
    if bool(getattr(data_cfg, "edge_struct_include_join_flags", True)):
        return V3_EDGE_STRUCT_FEATURE_COLUMNS
    return V3_EDGE_STRUCT_ROLE_COLUMNS


def _edge_struct_values(
    *,
    is_input_edge: Any,
    is_output_edge: Any,
    is_internal_edge: Any,
    has_stream_key: float,
    join_ok: float,
    data_cfg: DataConfig,
) -> list[float]:
    values = [
        _truthy_csv_value(is_input_edge),
        _truthy_csv_value(is_output_edge),
        _truthy_csv_value(is_internal_edge),
    ]
    if bool(getattr(data_cfg, "edge_struct_include_join_flags", True)):
        values.extend([float(has_stream_key), float(join_ok)])
    return values


def _truthy_csv_value(value: Any) -> float:
    text = str(value).strip().lower()
    return 1.0 if text in {"1", "true", "yes", "y"} else 0.0


def _clean_csv_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _edge_row_primary_stream_label(edge_row: pd.Series) -> str:
    sn = _clean_csv_text(edge_row.get("stream_name"))
    if sn:
        return sn
    return _clean_csv_text(edge_row.get("main_data_stream_key"))


class ProcessGraphTabularDataset(torch.utils.data.Dataset):
    """Load `GraphSample` rows from a CSV table + process spec workbooks."""

    def __init__(
        self,
        csv_path: Path,
        data_cfg: DataConfig,
        project_root: Path,
        *,
        split_filter: Optional[str] = None,
        split_manifest: Optional[Path] = None,
        oper_mean: Optional[torch.Tensor] = None,
        oper_std: Optional[torch.Tensor] = None,
        target_mean: Optional[Dict[str, float]] = None,
        target_std: Optional[Dict[str, float]] = None,
        fixed_slot_clip_max: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.project_root = project_root
        self.split_filter = split_filter

        resolved_csv = (project_root / csv_path).resolve()
        if not resolved_csv.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {resolved_csv}")
        self.frame = pd.read_csv(resolved_csv)

        spec_root = (project_root / data_cfg.process_spec_dir).resolve()
        if not spec_root.is_dir():
            raise FileNotFoundError(f"process_spec_dir not found: {spec_root}")

        self._spec_cache: Dict[str, ProcessSpec] = {}
        self._rows: List[int] = []
        self._source_row_indices: List[int] = []
        split_col = data_cfg.split_column
        if split_manifest is not None:
            if not split_manifest.is_file():
                raise FileNotFoundError(f"split manifest not found: {split_manifest}")
            man = pd.read_csv(split_manifest)
            idx_col = None
            for cand in ("merged_row_index", "row_index", "csv_row_index"):
                if cand in man.columns:
                    idx_col = cand
                    break
            if idx_col is None:
                raise ValueError(
                    f"{split_manifest} must contain merged_row_index (or row_index / csv_row_index)."
                )
            raw_idx = [int(x) for x in man[idx_col].tolist()]
            n = len(self.frame)
            self._rows = sorted({i for i in raw_idx if 0 <= i < n})
        else:
            for idx, row in self.frame.iterrows():
                if split_filter is not None and split_col in row:
                    if str(row[split_col]) != split_filter:
                        continue
                self._rows.append(int(idx))

        pcol = data_cfg.process_id_column
        if getattr(data_cfg, "task_mode", "multitask") == "edge_all":
            plist: list[int] | None = None
            if split_filter == "train" and data_cfg.edge_all_train_processes:
                plist = list(data_cfg.edge_all_train_processes)
            elif split_filter == "val" and data_cfg.edge_all_val_processes:
                plist = list(data_cfg.edge_all_val_processes)
            elif split_filter == "test" and data_cfg.edge_all_test_processes:
                plist = list(data_cfg.edge_all_test_processes)
            elif data_cfg.edge_all_processes:
                plist = list(data_cfg.edge_all_processes)
            if plist:
                allow = {canonicalize_process_id(p) for p in plist}
                self._rows = [
                    i
                    for i in self._rows
                    if canonicalize_process_id(self.frame.iloc[i][pcol]) in allow
                ]
            if getattr(data_cfg, "edge_all_split_mode", "seen_process") == "process_holdout":
                ho = getattr(data_cfg, "edge_all_holdout_test_process", None)
                if ho is not None:
                    hp = canonicalize_process_id(ho)
                    if split_filter == "test":
                        self._rows = [
                            i
                            for i in self._rows
                            if canonicalize_process_id(self.frame.iloc[i][pcol]) == hp
                        ]
                    elif split_filter in ("train", "val"):
                        self._rows = [
                            i
                            for i in self._rows
                            if canonicalize_process_id(self.frame.iloc[i][pcol]) != hp
                        ]

        excluded_path_raw = str(
            getattr(data_cfg, "excluded_graph_samples_path", "") or ""
        ).strip()
        if excluded_path_raw:
            excluded_path = Path(excluded_path_raw)
            if not excluded_path.is_absolute():
                excluded_path = (project_root / excluded_path).resolve()
            if not excluded_path.is_file():
                raise FileNotFoundError(
                    f"excluded graph samples file not found: {excluded_path}"
                )
            excluded_frame = pd.read_csv(excluded_path)
            required_columns = {"process_id", "sample_id"}
            missing_columns = sorted(required_columns - set(excluded_frame.columns))
            if missing_columns:
                raise ValueError(
                    f"{excluded_path} missing required columns: {missing_columns}"
                )
            excluded_keys = {
                (canonicalize_process_id(row.process_id), _id_key(row.sample_id))
                for row in excluded_frame.itertuples(index=False)
            }
            before = len(self._rows)
            self._rows = [
                i
                for i in self._rows
                if (
                    canonicalize_process_id(self.frame.iloc[i][pcol]),
                    _id_key(self.frame.iloc[i]["ID"]),
                )
                not in excluded_keys
            ]
            excluded_count = before - len(self._rows)
            if excluded_count:
                print(
                    "[data][excluded-graphs] "
                    f"split={split_filter or 'all'} excluded={excluded_count} "
                    f"remaining={len(self._rows)} path={excluded_path}",
                    flush=True,
                )

        self._source_row_indices = list(self._rows)
        self.frame = self.frame.iloc[self._rows].reset_index(drop=True)
        self._rows = list(range(len(self.frame)))
        if bool(getattr(data_cfg, "downcast_main_frame_numeric", True)):
            protected_cols = {data_cfg.process_id_column, data_cfg.split_column, "ID"}
            for col in self.frame.columns:
                if col in protected_cols:
                    continue
                if pd.api.types.is_float_dtype(self.frame[col]):
                    self.frame[col] = pd.to_numeric(self.frame[col], downcast="float")
                elif pd.api.types.is_integer_dtype(self.frame[col]):
                    self.frame[col] = pd.to_numeric(self.frame[col], downcast="integer")

        self.oper_mean = oper_mean
        self.oper_std = oper_std
        self.target_mean = target_mean or {}
        self.target_std = target_std or {}
        self.fixed_slot_clip_max = fixed_slot_clip_max or {}
        self._debug_slot_log_counts: Dict[str, int] = {slot: 0 for slot in FIXED_SLOTS}
        self._stream_mapping_frame: pd.DataFrame | None = None
        self._stream_process_frame_cache: Dict[str, pd.DataFrame] = {}
        self._main_process_frame_cache: Dict[str, pd.DataFrame] = {}
        self._node_energy_exclude_cache = self._load_node_energy_consistency_cache()
        self._stream_rows_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._stream_allowed_ids_by_process: Dict[str, set[str]] = {}
        self._use_v3_canonical = bool(getattr(data_cfg, "use_canonical_graph_spec_v3", False))
        self._v3_nodes: pd.DataFrame | None = None
        self._v3_edges: pd.DataFrame | None = None
        self._v3_answers: pd.DataFrame | None = None
        self._v3_hx_pairs: pd.DataFrame | None = None
        self._v3_nodes_by_pid: Dict[int, pd.DataFrame] = {}
        self._v3_edges_by_pid: Dict[int, pd.DataFrame] = {}
        self._v3_answers_by_pid: Dict[int, pd.DataFrame] = {}
        self._v3_hx_pairs_by_pid: Dict[int, pd.DataFrame] = {}
        self._v4_targets: pd.DataFrame | None = None
        self._v3_logged_stats: set[str] = set()
        raw_known_feed_cfg = getattr(data_cfg, "known_feed_condition", None)
        self._known_feed_cfg = parse_known_feed_condition(raw_known_feed_cfg)
        self._hx_pair_cfg = parse_hx_pair_relation(
            getattr(data_cfg, "hx_pair_relation", None)
        )
        self._feed_head_cfg = parse_feed_head_conditioning(
            getattr(data_cfg, "feed_head_conditioning", None)
        )
        if self._feed_head_cfg.enabled and not self._known_feed_cfg.enabled:
            raise ValueError(
                "feed_head_conditioning requires known_feed_condition.enabled=true."
            )
        self._known_feed_log_data_availability = bool(
            raw_known_feed_cfg.get("log_data_availability", False)
            if isinstance(raw_known_feed_cfg, Mapping)
            else False
        )
        self._known_feed_log_sample_values = bool(
            raw_known_feed_cfg.get("log_sample_values", False)
            if isinstance(raw_known_feed_cfg, Mapping)
            else False
        )
        self._known_feed_bindings_by_pid: Dict[int, tuple[KnownFeedBinding, ...]] = {}
        self._independent_feed_topology_by_pid: Dict[
            int, tuple[IndependentFeedTopologyBinding, ...]
        ] = {}
        self._v3_input_node_index_by_pid: Dict[int, int | None] = {}
        self._known_feed_sample_log_written = False
        if self._use_v3_canonical:
            if data_cfg.topology_mode != "stream_edge":
                raise CanonicalStreamKeyCollisionError(
                    "data.use_canonical_graph_spec_v3=True requires data.topology_mode='stream_edge' "
                    "(edge-level readout + edge features)."
                )
            self._load_v3_reference_tables()
            if self._known_feed_cfg.enabled:
                self._initialize_known_feed_condition()
        if self.data_cfg.topology_mode == "stream_edge" and "ID" in self.frame.columns:
            pcol = self.data_cfg.process_id_column
            for row_idx in self._rows:
                row = self.frame.iloc[row_idx]
                if pcol not in row or pd.isna(row.get("ID")):
                    continue
                process_num = self._process_number(str(row[pcol]))
                row_key = _id_key(row.get("ID"))
                if not row_key:
                    continue
                self._stream_allowed_ids_by_process.setdefault(process_num, set()).add(row_key)

    def _load_node_energy_consistency_cache(self) -> Dict[str, set[str]]:
        raw_path = str(getattr(self.data_cfg, "node_energy_consistency_cache_path", "") or "").strip()
        if not raw_path:
            return {}
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"node energy consistency cache not found: {path}")
        frame = pd.read_csv(path)
        required = {"process_id", "sample_id", "node_name"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required columns for node energy consistency cache: {missing}")
        if "exclude_energy" in frame.columns:
            bad = frame[pd.to_numeric(frame["exclude_energy"], errors="coerce").fillna(0.0) > 0.5]
        elif "energy_consistent" in frame.columns:
            consistent_text = frame["energy_consistent"].astype(str).str.strip().str.lower()
            bad = frame[~consistent_text.isin({"1", "1.0", "true", "yes", "y"})]
        else:
            raise ValueError(f"{path} must contain exclude_energy or energy_consistent column.")
        cache: Dict[str, set[str]] = {}
        for row in bad.itertuples(index=False):
            key = _sample_cache_key(getattr(row, "process_id"), getattr(row, "sample_id"))
            node_name = str(getattr(row, "node_name")).strip()
            if node_name:
                cache.setdefault(key, set()).add(node_name)
        if cache:
            print(
                f"[node-energy-cache] loaded bad-node masks: samples={len(cache)} rows={len(bad)} path={path}",
                flush=True,
            )
        return cache

    def _load_v3_reference_tables(self) -> None:
        base = (self.project_root / getattr(self.data_cfg, "canonical_graph_spec_v3_dir", "data/reference/v3")).resolve()
        for name in ("canonical_nodes.csv", "canonical_edges.csv", "target_answer_edges.csv"):
            p = base / name
            if not p.is_file():
                raise FileNotFoundError(f"v3 canonical spec file missing: {p}")
        self._v3_nodes = pd.read_csv(base / "canonical_nodes.csv", dtype={"process_id": int})
        self._v3_edges = pd.read_csv(base / "canonical_edges.csv", dtype={"process_id": int})
        self._v3_answers = pd.read_csv(base / "target_answer_edges.csv", dtype={"process_id": int})
        if self._hx_pair_cfg.enabled:
            pair_path = Path(self._hx_pair_cfg.metadata_path)
            if not pair_path.is_absolute():
                pair_path = (self.project_root / pair_path).resolve()
            if not pair_path.is_file():
                raise FileNotFoundError(f"HX pair metadata file missing: {pair_path}")
            self._v3_hx_pairs = pd.read_csv(pair_path, dtype={"process_id": int})
            required = {
                "process_id",
                "hx_node",
                "pair_id",
                "inlet_edge_id",
                "outlet_edge_id",
                "source",
            }
            missing = sorted(required.difference(self._v3_hx_pairs.columns))
            if missing:
                raise ValueError(
                    f"HX pair metadata is missing required columns {missing}: {pair_path}"
                )
        v4_path = (self.project_root / "data/reference/v4/target_stream_targets.csv").resolve()
        if v4_path.is_file():
            self._v4_targets = pd.read_csv(v4_path, dtype={"process_id": int})

    def _v3_nodes_for_pid(self, pid: int) -> pd.DataFrame:
        assert self._v3_nodes is not None
        if pid not in self._v3_nodes_by_pid:
            nodes_df = self._v3_nodes[self._v3_nodes["process_id"] == pid].copy()
            if nodes_df.empty:
                raise ValueError(f"v3 canonical_nodes: no rows for process_id={pid}")
            self._v3_nodes_by_pid[pid] = nodes_df.sort_values("node_id").reset_index(drop=True)
        return self._v3_nodes_by_pid[pid]

    def _canonical_v_input_node_index(self, pid: int) -> int | None:
        if pid in self._v3_input_node_index_by_pid:
            return self._v3_input_node_index_by_pid[pid]
        nodes_df = self._v3_nodes_for_pid(pid).reset_index(drop=True)
        node_id = nodes_df.get("node_id", pd.Series("", index=nodes_df.index)).astype(str).str.upper()
        is_virtual = (
            nodes_df.get("is_virtual", pd.Series(False, index=nodes_df.index))
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes"})
        )
        node_type = nodes_df.get("node_type", pd.Series("", index=nodes_df.index)).astype(str).str.lower()
        canonical = nodes_df[
            node_id.str.endswith("_N_V_INPUT")
            & is_virtual
            & node_type.eq("virtual")
        ]
        if len(canonical) != 1:
            # Compatibility fallback for older reference tables that predate canonical node_id.
            node_name = nodes_df.get("node_name", pd.Series("", index=nodes_df.index)).astype(str)
            canonical = nodes_df[node_name.str.strip().str.upper().eq("V_INPUT") & is_virtual]
        index: int | None = int(canonical.index[0]) if len(canonical) == 1 else None
        if index is None:
            message = (
                f"canonical V_INPUT resolution failed for Process{pid}: "
                f"candidate_count={len(canonical)}"
            )
            if self._known_feed_cfg.missing_v_input == "error":
                raise ValueError(message)
            warnings.warn(message + "; feed values and masks will remain zero.", RuntimeWarning)
        self._v3_input_node_index_by_pid[pid] = index
        return index

    def _known_feed_bindings(self, pid: int, spec: ProcessSpec) -> tuple[KnownFeedBinding, ...]:
        if pid not in self._known_feed_bindings_by_pid:
            self._known_feed_bindings_by_pid[pid] = known_feed_bindings(spec)
        return self._known_feed_bindings_by_pid[pid]

    def _independent_feed_topology_bindings(
        self,
        pid: int,
        spec: ProcessSpec,
    ) -> tuple[IndependentFeedTopologyBinding, ...]:
        if self._known_feed_cfg.mode != "independent_feed_nodes":
            return ()
        if pid not in self._independent_feed_topology_by_pid:
            edges_df = self._v3_edges_for_pid(pid)
            topology = build_independent_feed_topology_bindings(
                process_id=pid,
                bindings=self._known_feed_bindings(pid, spec),
                canonical_edge_rows=edges_df.to_dict(orient="records"),
            )
            base_node_names = set(
                str(value) for value in self._v3_nodes_for_pid(pid)["node_name"].tolist()
            )
            base_edge_ids = set(
                str(value) for value in edges_df["canonical_edge_id"].tolist()
            )
            for binding in topology:
                if binding.feed_node_name in base_node_names:
                    raise ValueError(
                        f"Process{pid}: F2 node {binding.feed_node_name!r} already "
                        "exists in canonical_nodes."
                    )
                if binding.destination_node_name not in base_node_names:
                    raise ValueError(
                        f"Process{pid}: F2 destination "
                        f"{binding.destination_node_name!r} is absent from canonical_nodes."
                    )
                if binding.context_edge_id in base_edge_ids:
                    raise ValueError(
                        f"Process{pid}: F2 edge {binding.context_edge_id!r} collides "
                        "with canonical_edges."
                    )
            self._independent_feed_topology_by_pid[pid] = topology
        return self._independent_feed_topology_by_pid[pid]

    def _target_edge_ids_for_pid(self, pid: int) -> set[str]:
        if self._v4_targets is not None:
            rows = self._v4_targets[self._v4_targets["process_id"] == int(pid)]
            if "canonical_edge_id" in rows.columns:
                return {
                    str(value).strip()
                    for value in rows["canonical_edge_id"].dropna().tolist()
                    if str(value).strip()
                }
        answers = self._v3_answers_for_pid(pid)
        return {
            str(value).strip()
            for value in answers["canonical_answer_edge_id"].dropna().tolist()
            if str(value).strip()
        }

    def _resolve_known_feed_values(
        self,
        *,
        pid: int,
        spec: ProcessSpec,
        data_row: Mapping[str, float],
        incoming_map: Mapping[str, list[str]],
    ) -> tuple[list[float], list[int]]:
        values: list[float] = []
        masks: list[int] = []
        for binding in self._known_feed_bindings(pid, spec):
            if binding.cell_spec is None or binding.source_node_name is None:
                values.append(0.0)
                masks.append(0)
                continue
            value, mask = resolve_feature_value(
                binding.cell_spec,
                ResolveContext(
                    process_spec=spec,
                    data_row=dict(data_row),
                    incoming_map=dict(incoming_map),
                    node_name=binding.source_node_name,
                    slot=binding.operating_slot,
                    passthrough_policy=self.data_cfg.passthrough_policy,
                ),
            )
            values.append(float(value) if mask else 0.0)
            masks.append(int(mask))
        return values, masks

    def _known_feed_node_tail(
        self,
        *,
        node_name: str,
        feed_values: Sequence[float],
        feed_masks: Sequence[int],
        v_input_with_raw_feeds: bool = False,
        independent_feed_index: int | None = None,
    ) -> tuple[list[float], list[int]]:
        """Build the optional raw-feed and ratio operating slots for one node."""
        if not self._known_feed_cfg.enabled:
            return [], []
        tail_names = known_feed_feature_names(self._known_feed_cfg)
        values = [0.0] * len(tail_names)
        masks = [0] * len(tail_names)
        if v_input_with_raw_feeds:
            values[: len(KNOWN_FEED_FEATURE_NAMES)] = list(feed_values)
            masks[: len(KNOWN_FEED_FEATURE_NAMES)] = list(feed_masks)
        elif independent_feed_index is not None and feed_masks[independent_feed_index]:
            values[independent_feed_index] = float(feed_values[independent_feed_index])
            masks[independent_feed_index] = 1

        if self._known_feed_cfg.ratio_features_enabled:
            normalized_name = str(node_name).strip().upper().replace("-", "_")
            is_v_input = normalized_name == "V_INPUT"
            is_burner = "BURNER" in normalized_name.split("_")
            attach_ratios = (
                is_v_input and self._known_feed_cfg.ratio_attach_to_v_input
            ) or (
                is_burner and self._known_feed_cfg.ratio_attach_to_burner
            )
            if attach_ratios:
                ratio_values, ratio_masks = known_feed_ratio_values(
                    feed_values,
                    feed_masks,
                    self._known_feed_cfg,
                )
                ratio_start = len(KNOWN_FEED_FEATURE_NAMES)
                values[ratio_start:] = ratio_values
                masks[ratio_start:] = ratio_masks
        return values, masks

    def _initialize_known_feed_condition(self) -> None:
        process_col = self.data_cfg.process_id_column
        feature_names = list(operating_feature_names(self._known_feed_cfg))
        print(
            f"[known-feed] enabled mode={self._known_feed_cfg.mode} "
            f"features={list(known_feed_feature_names(self._known_feed_cfg))} "
            f"oper_dim={len(feature_names)} "
            "preprocessing=train_only_existing_operating_scaler",
            flush=True,
        )
        if not self._known_feed_log_data_availability:
            return
        for process_id in sorted(
            {str(value) for value in self.frame[process_col].dropna().tolist()},
            key=_canonical_process_number,
        ):
            pid = _canonical_process_number(process_id)
            spec = self._load_spec(process_id)
            bindings = self._known_feed_bindings(pid, spec)
            v_input_index = (
                self._canonical_v_input_node_index(pid)
                if self._known_feed_cfg.mode == "v_input_node"
                else None
            )
            topology = self._independent_feed_topology_bindings(pid, spec)
            process_frame = self.frame[self.frame[process_col].astype(str) == process_id]
            stats: list[str] = []
            for binding in bindings:
                ref = binding.source_reference
                if ref is None or ref not in process_frame.columns:
                    stats.append(
                        f"{binding.feed_name}=source:{binding.source_node_name or '-'} "
                        f"column:{ref or '-'} available:0 missing:{len(process_frame)} zero:0"
                    )
                    continue
                values = pd.to_numeric(process_frame[ref], errors="coerce")
                available = int(values.notna().sum())
                stats.append(
                    f"{binding.feed_name}=source:{binding.source_node_name} column:{ref} "
                    f"available:{available} missing:{int(values.isna().sum())} "
                    f"zero:{int((values == 0).sum())}"
                )
            if self._known_feed_cfg.mode == "v_input_node":
                topology_text = (
                    f"V_INPUT={'1/1' if v_input_index is not None else '0/1'}"
                )
            else:
                topology_text = "connections=" + ",".join(
                    f"{item.feed_node_name}->{item.destination_node_name}"
                    f"[{item.context_edge_id}]"
                    for item in topology
                )
            print(
                f"[known-feed][Process{pid}] {topology_text} | " + " | ".join(stats),
                flush=True,
            )

    def _log_known_feed_sample_once(
        self,
        *,
        process_id: str,
        sample_id: str,
        raw_sample: GraphSample,
        scaled_sample: GraphSample,
    ) -> None:
        if (
            not self._known_feed_cfg.enabled
            or not self._known_feed_log_sample_values
            or self._known_feed_sample_log_written
        ):
            return
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.id != 0:
            return
        pid = _canonical_process_number(process_id)
        start = len(OPER_FEATURE_SLOTS)
        if self._known_feed_cfg.mode == "v_input_node":
            v_input_index = self._canonical_v_input_node_index(pid)
            if v_input_index is None:
                return
            raw_values = raw_sample.x_oper[v_input_index][start:]
            scaled_values = scaled_sample.x_oper[v_input_index][start:]
            masks = raw_sample.x_oper_mask[v_input_index][start:]
            detail = (
                f"V_INPUT_index={v_input_index} raw={raw_values} "
                f"scaled={scaled_values} mask={masks}"
            )
        else:
            rows: list[str] = []
            for node_index, node_name in enumerate(raw_sample.node_names):
                if not str(node_name).startswith("V_FEED_"):
                    continue
                rows.append(
                    f"{node_name}@{node_index}:raw={raw_sample.x_oper[node_index][start:]}"
                    f":scaled={scaled_sample.x_oper[node_index][start:]}"
                    f":mask={raw_sample.x_oper_mask[node_index][start:]}"
                )
            detail = "feed_nodes={" + "; ".join(rows) + "}"
        print(
            f"[known-feed][sample] process=Process{pid} sample={sample_id} "
            f"{detail} x_oper_shape=({len(scaled_sample.x_oper)},"
            f"{len(scaled_sample.x_oper[0]) if scaled_sample.x_oper else 0})",
            flush=True,
        )
        self._known_feed_sample_log_written = True

    def _v3_edges_for_pid(self, pid: int) -> pd.DataFrame:
        assert self._v3_edges is not None
        if pid not in self._v3_edges_by_pid:
            edges_df = self._v3_edges[self._v3_edges["process_id"] == pid].copy()
            if edges_df.empty:
                raise ValueError(f"v3 canonical_edges: no rows for process_id={pid}")
            self._v3_edges_by_pid[pid] = edges_df.sort_values("canonical_edge_id").reset_index(drop=True)
        return self._v3_edges_by_pid[pid]

    def _v3_answers_for_pid(self, pid: int) -> pd.DataFrame:
        assert self._v3_answers is not None
        if pid not in self._v3_answers_by_pid:
            self._v3_answers_by_pid[pid] = self._v3_answers[self._v3_answers["process_id"] == pid].copy()
        return self._v3_answers_by_pid[pid]

    def _v3_hx_pairs_for_pid(self, pid: int) -> pd.DataFrame:
        if not self._hx_pair_cfg.enabled:
            return pd.DataFrame()
        assert self._v3_hx_pairs is not None
        if pid not in self._v3_hx_pairs_by_pid:
            rows = self._v3_hx_pairs[
                self._v3_hx_pairs["process_id"] == pid
            ].copy()
            self._v3_hx_pairs_by_pid[pid] = rows.sort_values(
                ["hx_node", "pair_id"]
            ).reset_index(drop=True)
        return self._v3_hx_pairs_by_pid[pid]

    def _build_hx_pair_metadata(
        self,
        *,
        pid: int,
        edges_df: pd.DataFrame,
        all_edge_ids: Sequence[str],
    ) -> tuple[list[int], list[int], list[float], list[int]]:
        if not self._hx_pair_cfg.enabled:
            return [], [], [], []
        edge_id_to_pos = {str(edge_id): i for i, edge_id in enumerate(all_edge_ids)}
        current: list[int] = []
        paired: list[int] = []
        mask: list[float] = []
        side: list[int] = []
        edge_rows = {
            str(row["canonical_edge_id"]): row
            for _, row in edges_df.iterrows()
        }
        pair_rows = self._v3_hx_pairs_for_pid(pid)
        hx_nodes = set(
            edges_df.loc[
                edges_df["src_node"].astype(str).str.startswith("HX")
                | edges_df["dst_node"].astype(str).str.startswith("HX"),
                "src_node",
            ].astype(str)
        ) | set(
            edges_df.loc[
                edges_df["src_node"].astype(str).str.startswith("HX")
                | edges_df["dst_node"].astype(str).str.startswith("HX"),
                "dst_node",
            ].astype(str)
        )
        hx_nodes = {name for name in hx_nodes if name.startswith("HX")}
        metadata_hx_nodes = set(pair_rows["hx_node"].astype(str))
        missing_hx = sorted(hx_nodes.difference(metadata_hx_nodes))
        if missing_hx:
            raise ValueError(
                f"Process{pid}: HX pair metadata has no rows for units {missing_hx}."
            )

        used_hx_edges: set[tuple[str, str]] = set()
        for row in pair_rows.itertuples(index=False):
            hx_node = str(row.hx_node)
            inlet_id = str(row.inlet_edge_id)
            outlet_id = str(row.outlet_edge_id)
            if inlet_id == outlet_id:
                raise ValueError(f"Process{pid} {hx_node}: HX pair cannot self-reference {inlet_id}.")
            if inlet_id not in edge_rows or outlet_id not in edge_rows:
                raise ValueError(
                    f"Process{pid} {hx_node}: pair references absent edges "
                    f"{inlet_id}->{outlet_id}."
                )
            inlet = edge_rows[inlet_id]
            outlet = edge_rows[outlet_id]
            if str(inlet["dst_node"]) != hx_node or str(outlet["src_node"]) != hx_node:
                raise ValueError(
                    f"Process{pid} {hx_node}: invalid inlet/outlet direction for "
                    f"{inlet_id}->{outlet_id}."
                )
            hx_edge_keys = ((hx_node, inlet_id), (hx_node, outlet_id))
            if any(key in used_hx_edges for key in hx_edge_keys):
                raise ValueError(
                    f"Process{pid} {hx_node}: one-to-one HX-local pair violation involving "
                    f"{inlet_id} or {outlet_id}."
                )
            used_hx_edges.update(hx_edge_keys)
            inlet_pos = edge_id_to_pos[inlet_id]
            outlet_pos = edge_id_to_pos[outlet_id]
            current.extend((inlet_pos, outlet_pos))
            paired.extend((outlet_pos, inlet_pos))
            mask.extend((1.0, 1.0))
            side.extend((1, 2))

        directed = set(zip(current, paired))
        if any((partner, edge) not in directed for edge, partner in directed):
            raise ValueError(f"Process{pid}: asymmetric HX pair relation.")
        return current, paired, mask, side

    def _resolve_v3_stream_row(
        self,
        *,
        edge_row: pd.Series,
        stream_by_name: Mapping[str, pd.Series],
        process_id: str,
        sample_id: Any,
    ) -> tuple[str, str, pd.Series | None]:
        raw_stream_key = _clean_csv_text(edge_row.get("main_data_stream_key"))
        canonical_stream_key = canonicalize_stream_key(
            raw_stream_key,
            process_id=process_id,
            source="canonical_edges.csv",
        )
        if not canonical_stream_key:
            return raw_stream_key, canonical_stream_key, None

        candidates = [canonical_stream_key]
        fallback_key = canonicalize_stream_key(
            _clean_csv_text(edge_row.get("stream_name_norm")),
            process_id=process_id,
            source="canonical_edges.csv:stream_name_norm",
        )
        if fallback_key and fallback_key not in candidates:
            candidates.append(fallback_key)
        for candidate in candidates:
            stream_row = stream_by_name.get(candidate)
            if stream_row is not None:
                return raw_stream_key, canonical_stream_key, stream_row

        process_num = self._process_number(process_id)
        raise missing_required_stream_key_error(
            process_id=process_id,
            sample_id=sample_id,
            canonical_edge_id=edge_row.get("canonical_edge_id"),
            raw_graph_key=raw_stream_key,
            canonical_graph_key=canonical_stream_key,
            available_csv_keys=list(stream_by_name),
            csv_path=self._stream_csv_path(process_num),
        )

    def _enrich_data_row_with_v4_stream_targets(
        self,
        *,
        data_row: Dict[str, float],
        process_id: str,
        stream_by_name: Mapping[str, pd.Series],
    ) -> None:
        """Fill missing Main target columns from Process_Streams formulas when available."""
        if self._v4_targets is None:
            return
        pid = _canonical_process_number(process_id)
        sub = self._v4_targets[self._v4_targets["process_id"] == pid]
        if sub.empty:
            return
        for _, row in sub.iterrows():
            target_col = _clean_csv_text(row.get("target_feature_name"))
            if not target_col or data_row.get(target_col) is not None:
                continue
            species = _clean_csv_text(row.get("target_species"))
            frac_slot = SPECIES_TO_FRAC_SLOT.get(species)
            if not frac_slot:
                continue
            stream_key = _clean_csv_text(row.get("required_stream_key")) or _clean_csv_text(
                row.get("main_data_stream_key")
            )
            stream_key = canonicalize_stream_key(
                stream_key,
                process_id=process_id,
                source="target_stream_targets.csv",
            )
            stream_row = stream_by_name.get(stream_key)
            if stream_row is None:
                continue
            if "Mole_Flow" not in stream_row.index or frac_slot not in stream_row.index:
                continue
            if pd.isna(stream_row["Mole_Flow"]) or pd.isna(stream_row[frac_slot]):
                continue
            try:
                scale = float(row.get("scale_factor", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            data_row[target_col] = scale * float(stream_row["Mole_Flow"]) * float(stream_row[frac_slot])

    def __len__(self) -> int:
        return len(self._rows)

    def _edge_target_columns_for_stats(self) -> list[str]:
        if str(getattr(self.data_cfg, "edge_target_columns_mode", "auto")).lower() == "explicit":
            explicit = getattr(self.data_cfg, "edge_target_columns", None)
            if not explicit:
                raise ValueError("edge_target_columns_mode=explicit requires data.edge_target_columns.")
            return [str(c) for c in explicit]
        return list(STREAM_EDGE_FEATURE_SLOTS)

    def _stats_frame_with_v4_targets(self, *, limit: int | None = None) -> pd.DataFrame:
        max_len = len(self) if limit is None else min(int(limit), len(self))
        frame = self.frame.iloc[self._rows[:max_len]].copy()
        if self._v4_targets is None or frame.empty or "ID" not in frame.columns:
            return frame
        process_col = self.data_cfg.process_id_column
        frame["_ID_STR"] = frame["ID"].map(_id_key)
        for process_id, idx in frame.groupby(process_col, sort=False).groups.items():
            pid = _canonical_process_number(str(process_id))
            sub_targets = self._v4_targets[self._v4_targets["process_id"] == pid]
            if sub_targets.empty:
                continue
            stream_frame = self._stream_frame_for_process_num(str(pid))
            id_keys = frame.loc[idx, "_ID_STR"].astype(str).tolist()
            available = [key for key in id_keys if key in stream_frame.index]
            if not available:
                continue
            proc_stream = stream_frame.loc[available]
            for _, target_row in sub_targets.iterrows():
                target_col = _clean_csv_text(target_row.get("target_feature_name"))
                if not target_col:
                    continue
                species = _clean_csv_text(target_row.get("target_species"))
                frac_slot = SPECIES_TO_FRAC_SLOT.get(species)
                if not frac_slot or "Mole_Flow" not in proc_stream.columns or frac_slot not in proc_stream.columns:
                    continue
                stream_key = canonicalize_stream_key(
                    _clean_csv_text(target_row.get("required_stream_key"))
                    or _clean_csv_text(target_row.get("main_data_stream_key")),
                    process_id=process_id,
                    source="target_stream_targets.csv",
                )
                if not stream_key:
                    continue
                rows = proc_stream[
                    proc_stream[STREAM_KEY_CANONICAL_COLUMN].astype(str) == stream_key
                ]
                if rows.empty:
                    continue
                values = rows[["Mole_Flow", frac_slot]].dropna()
                if values.empty:
                    continue
                try:
                    scale = float(target_row.get("scale_factor", 1.0))
                except (TypeError, ValueError):
                    scale = 1.0
                series = (scale * values["Mole_Flow"].astype(float) * values[frac_slot].astype(float))
                series = series[~series.index.duplicated(keep="first")]
                if target_col not in frame.columns:
                    frame[target_col] = pd.NA
                frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce").astype("float64")
                fill_idx = frame.index.intersection(idx)
                missing = frame.loc[fill_idx, target_col].isna()
                if missing.any():
                    ids = frame.loc[fill_idx[missing], "_ID_STR"].astype(str)
                    frame.loc[fill_idx[missing], target_col] = ids.map(series)
        return frame

    def iter_x_oper_for_normalizer(self, *, limit: int | None = None) -> Iterable[list[list[float]]]:
        """Yield raw x_oper only, avoiding full GraphSampleRecord construction for startup stats."""
        if not (self._use_v3_canonical and self.data_cfg.topology_mode == "stream_edge"):
            raise NotImplementedError("fast x_oper normalizer path currently supports v3 stream_edge datasets only.")
        process_col = self.data_cfg.process_id_column
        reserved = {process_col, self.data_cfg.split_column}
        stats_frame = self._stats_frame_with_v4_targets(limit=limit)
        for _, row in stats_frame.iterrows():
            if process_col not in row:
                raise KeyError(f"Missing required column {process_col!r} in dataset CSV.")
            process_id = str(row[process_col])
            pid = _canonical_process_number(process_id)
            spec = self._load_spec(process_id)
            data_row = _to_float_mapping(row, reserved)

            nodes_df = self._v3_nodes_for_pid(pid)
            node_names = [str(x) for x in nodes_df["node_name"].tolist()]
            topology = self._independent_feed_topology_bindings(pid, spec)
            node_names.extend(item.feed_node_name for item in topology)
            incoming_map = build_incoming_node_map(spec)
            v_input_index = (
                self._canonical_v_input_node_index(pid)
                if self._known_feed_cfg.enabled
                and self._known_feed_cfg.mode == "v_input_node"
                else None
            )
            feed_values, feed_masks = (
                self._resolve_known_feed_values(
                    pid=pid,
                    spec=spec,
                    data_row=data_row,
                    incoming_map=incoming_map,
                )
                if self._known_feed_cfg.enabled
                else ([0.0] * len(KNOWN_FEED_FEATURE_NAMES), [0] * len(KNOWN_FEED_FEATURE_NAMES))
            )
            feed_name_to_index = {
                name: index for index, name in enumerate(KNOWN_FEED_NAMES)
            }
            topology_by_node = {
                item.feed_node_name: item for item in topology
            }
            x_oper: list[list[float]] = []
            for node_index, node_name in enumerate(node_names):
                if node_index == v_input_index:
                    tail_values, tail_masks = self._known_feed_node_tail(
                        node_name=node_name,
                        feed_values=feed_values,
                        feed_masks=feed_masks,
                        v_input_with_raw_feeds=True,
                    )
                    stats_feed_values = [
                        value if mask else math.nan
                        for value, mask in zip(tail_values, tail_masks)
                    ]
                    x_oper.append(
                        [0.0] * len(OPER_FEATURE_SLOTS) + stats_feed_values
                    )
                    continue
                if node_name in topology_by_node:
                    feed_index = feed_name_to_index[
                        topology_by_node[node_name].feed_name
                    ]
                    tail_values, tail_masks = self._known_feed_node_tail(
                        node_name=node_name,
                        feed_values=feed_values,
                        feed_masks=feed_masks,
                        independent_feed_index=feed_index,
                    )
                    stats_feed_values = [
                        value if mask else math.nan
                        for value, mask in zip(tail_values, tail_masks)
                    ]
                    x_oper.append(
                        [0.0] * len(OPER_FEATURE_SLOTS) + stats_feed_values
                    )
                    continue
                if node_name in {"V_INPUT", "V_OUTPUT"}:
                    values = [0.0] * len(OPER_FEATURE_SLOTS)
                    if self._known_feed_cfg.enabled:
                        tail_values, tail_masks = self._known_feed_node_tail(
                            node_name=node_name,
                            feed_values=feed_values,
                            feed_masks=feed_masks,
                        )
                        values += [
                            value if mask else math.nan
                            for value, mask in zip(tail_values, tail_masks)
                        ]
                    x_oper.append(values)
                    continue
                node = spec.nodes.get(node_name)
                values_n: list[float] = []
                for slot in OPER_FEATURE_SLOTS:
                    cell_spec = node.features[slot] if node is not None else None
                    if cell_spec is None:
                        values_n.append(0.0)
                        continue
                    value, _mask = resolve_feature_value(
                        cell_spec,
                        ResolveContext(
                            process_spec=spec,
                            data_row=dict(data_row),
                            incoming_map=incoming_map,
                            node_name=node_name,
                            slot=slot,
                            passthrough_policy=self.data_cfg.passthrough_policy,
                        ),
                    )
                    values_n.append(value)
                if self._known_feed_cfg.enabled:
                    tail_values, tail_masks = self._known_feed_node_tail(
                        node_name=node_name,
                        feed_values=feed_values,
                        feed_masks=feed_masks,
                    )
                    values_n += [
                        value if mask else math.nan
                        for value, mask in zip(tail_values, tail_masks)
                    ]
                x_oper.append(values_n)
            yield x_oper

    def iter_y_edge_for_scaler(
        self, *, limit: int | None = None
    ) -> Iterable[tuple[list[list[float]], list[float], list[str]]]:
        """Yield raw y_edge_true/y_edge_mask only for train-only scaler fitting."""
        if not (self._use_v3_canonical and self.data_cfg.topology_mode == "stream_edge"):
            raise NotImplementedError("fast y_edge scaler path currently supports v3 stream_edge datasets only.")
        max_len = len(self) if limit is None else min(int(limit), len(self))
        process_col = self.data_cfg.process_id_column
        edge_target_columns = self._edge_target_columns_for_stats()
        for index in range(max_len):
            row_idx = self._rows[index]
            row = self.frame.iloc[row_idx]
            if process_col not in row:
                raise KeyError(f"Missing required column {process_col!r} in dataset CSV.")
            process_id = str(row[process_col])
            pid = _canonical_process_number(process_id)
            row_id = row.get("ID")
            if row_id is None or pd.isna(row_id):
                raise KeyError("v3 stream join requires an 'ID' column in the dataset row.")
            stream_rows = self._stream_rows_for_process_id(process_id, row_id)
            stream_by_name = self._canonical_stream_index_for_sample(
                stream_rows,
                process_id=process_id,
                sample_id=row_id,
            )
            y_edge_true: list[list[float]] = []
            y_edge_mask: list[float] = []
            for _, er in self._v3_edges_for_pid(pid).iterrows():
                stream_key, canonical_stream_key, stream_row = self._resolve_v3_stream_row(
                    edge_row=er,
                    stream_by_name=stream_by_name,
                    process_id=process_id,
                    sample_id=row_id,
                )
                ok = stream_row is not None and bool(canonical_stream_key)
                if ok:
                    target_values = [
                        float(stream_row[feat_slot])
                        if feat_slot in stream_row.index and not pd.isna(stream_row[feat_slot])
                        else 0.0
                        for feat_slot in edge_target_columns
                    ]
                    y_edge_mask.append(1.0)
                else:
                    target_values = [0.0] * len(edge_target_columns)
                    y_edge_mask.append(0.0)
                y_edge_true.append(target_values)
            yield y_edge_true, y_edge_mask, edge_target_columns

    def fit_y_edge_scaler_fast(
        self,
        *,
        limit: int | None = None,
        stream_target_dim: int | None = None,
        std_eps: float = 1e-6,
        log_column_name: str | None = None,
        mass_flow_transform: str = "log1p",
        mass_flow_log_tau: float = 1.0,
        mass_flow_log_scale: float = 1.0,
        mass_flow_log_eps: float = 1.0e-8,
        physical_quantile_column_name: str | None = None,
        physical_quantile_probs: Sequence[float] = (0.05, 0.90, 0.95, 0.99),
    ) -> tuple[YEdgeScalerFitResult, YEdgeLog1pColumnScalerFitResult | None]:
        """Vectorized v3 stream-edge scaler fit without per-sample GraphSample construction."""
        if not (self._use_v3_canonical and self.data_cfg.topology_mode == "stream_edge"):
            raise NotImplementedError("fast y_edge scaler path currently supports v3 stream_edge datasets only.")
        dim = int(stream_target_dim) if stream_target_dim is not None else 0
        if dim <= 0:
            dim = len(STREAM_EDGE_FEATURE_SLOTS)
        columns = self._edge_target_columns_for_stats()
        if len(columns) < dim:
            columns = columns + [f"y_edge_{i}" for i in range(len(columns), dim)]
        if len(columns) != dim:
            raise ValueError(f"fit_y_edge_scaler_fast: expected {dim} columns, got {len(columns)}.")
        log_col_idx: int | None = None
        if log_column_name is not None:
            if log_column_name not in columns:
                raise ValueError(f"fit_y_edge_scaler_fast: missing column {log_column_name!r}; columns={columns}.")
            log_col_idx = int(columns.index(log_column_name))
        quantile_col_idx: int | None = None
        if physical_quantile_column_name is not None:
            if physical_quantile_column_name not in columns:
                raise ValueError(
                    "fit_y_edge_scaler_fast: missing physical quantile column "
                    f"{physical_quantile_column_name!r}; columns={columns}."
                )
            quantile_col_idx = int(columns.index(physical_quantile_column_name))

        max_len = len(self) if limit is None else min(int(limit), len(self))
        stats_frame = self.frame.iloc[self._rows[:max_len]].copy()
        if stats_frame.empty:
            raise ValueError("fit_y_edge_scaler_fast: empty dataset.")
        if "ID" not in stats_frame.columns:
            raise KeyError("fit_y_edge_scaler_fast requires an 'ID' column.")
        process_col = self.data_cfg.process_id_column
        stats_frame["_ID_STR"] = stats_frame["ID"].map(_id_key)

        sums = torch.zeros(dim, dtype=torch.float64)
        sumsq = torch.zeros(dim, dtype=torch.float64)
        counts_t = torch.zeros(dim, dtype=torch.long)
        log_total = 0.0
        log_total_sq = 0.0
        log_count = 0
        physical_quantile_values: list[float] = []

        for process_id, idx in stats_frame.groupby(process_col, sort=False).groups.items():
            pid = _canonical_process_number(str(process_id))
            id_keys = stats_frame.loc[idx, "_ID_STR"].astype(str).tolist()
            stream_frame = self._stream_frame_for_process_num(str(pid))
            available = [key for key in id_keys if key in stream_frame.index]
            missing_sample_ids = sorted(set(id_keys) - set(available))
            required_edges = [
                er
                for _, er in self._v3_edges_for_pid(pid).iterrows()
                if canonicalize_stream_key(
                    er.get("main_data_stream_key"),
                    process_id=process_id,
                    source="canonical_edges.csv",
                )
            ]
            if missing_sample_ids and required_edges:
                er = required_edges[0]
                raw_key = _clean_csv_text(er.get("main_data_stream_key"))
                raise missing_required_stream_key_error(
                    process_id=process_id,
                    sample_id=missing_sample_ids[0],
                    canonical_edge_id=er.get("canonical_edge_id"),
                    raw_graph_key=raw_key,
                    canonical_graph_key=canonicalize_stream_key(raw_key),
                    available_csv_keys=[],
                    csv_path=self._stream_csv_path(str(pid)),
                )
            if not available:
                continue
            proc_stream = stream_frame.loc[available]
            for _, er in self._v3_edges_for_pid(pid).iterrows():
                raw_stream_key = _clean_csv_text(er.get("main_data_stream_key"))
                stream_key = canonicalize_stream_key(
                    raw_stream_key,
                    process_id=process_id,
                    source="canonical_edges.csv",
                )
                if not stream_key:
                    continue
                keys = [stream_key]
                sn_norm = canonicalize_stream_key(
                    _clean_csv_text(er.get("stream_name_norm")),
                    process_id=process_id,
                    source="canonical_edges.csv:stream_name_norm",
                )
                if sn_norm and sn_norm != stream_key:
                    keys.append(sn_norm)
                rows = proc_stream[
                    proc_stream[STREAM_KEY_CANONICAL_COLUMN].astype(str).isin(keys)
                ]
                matched_ids = set(rows["_ID_STR"].astype(str).tolist())
                missing_ids = sorted(set(available) - matched_ids)
                if missing_ids:
                    sample_rows = proc_stream.loc[[missing_ids[0]]]
                    available_keys = sorted(
                        set(
                            sample_rows[STREAM_KEY_CANONICAL_COLUMN]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )
                    )
                    raise missing_required_stream_key_error(
                        process_id=process_id,
                        sample_id=missing_ids[0],
                        canonical_edge_id=er.get("canonical_edge_id"),
                        raw_graph_key=raw_stream_key,
                        canonical_graph_key=stream_key,
                        available_csv_keys=available_keys,
                        csv_path=self._stream_csv_path(str(pid)),
                    )
                for j, col in enumerate(columns):
                    if col in rows.columns:
                        values = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
                    else:
                        values = pd.Series([0.0] * len(rows), index=rows.index)
                    vals = torch.as_tensor(values.astype(float).to_numpy(), dtype=torch.float64)
                    sums[j] += vals.sum()
                    sumsq[j] += (vals * vals).sum()
                    counts_t[j] += int(vals.numel())
                if log_col_idx is not None:
                    col = columns[log_col_idx]
                    if col in rows.columns:
                        values = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
                        for raw in values.astype(float).to_numpy():
                            fv = transform_mass_flow_scalar(
                                float(raw),
                                transform=mass_flow_transform,
                                tau=mass_flow_log_tau,
                                scale=mass_flow_log_scale,
                                eps=mass_flow_log_eps,
                            )
                            log_total += fv
                            log_total_sq += fv * fv
                            log_count += 1
                if quantile_col_idx is not None:
                    col = columns[quantile_col_idx]
                    if col in rows.columns:
                        values = pd.to_numeric(rows[col], errors="coerce").dropna().astype(float)
                        physical_quantile_values.extend(float(value) for value in values if math.isfinite(float(value)))

        mean = torch.zeros(dim, dtype=torch.float32)
        std = torch.ones(dim, dtype=torch.float32)
        counts: list[int] = []
        warn_msgs: list[str] = []
        for j, col in enumerate(columns):
            count = int(counts_t[j].item())
            counts.append(count)
            if count <= 0:
                mean[j] = 0.0
                std[j] = 1.0
                msg = (
                    f"compute_y_edge_scaler: column {col!r} has zero mask==1 values on the fit split; "
                    "using mean=0 std=1."
                )
                warn_msgs.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                continue
            mean64 = sums[j] / float(count)
            var64 = (sumsq[j] / float(count)) - (mean64 * mean64)
            mean[j] = mean64.float()
            raw_std = float(torch.sqrt(torch.clamp(var64, min=0.0)).item())
            if not math.isfinite(raw_std) or raw_std < float(std_eps):
                msg = (
                    f"compute_y_edge_scaler: column {col!r} has std={raw_std!r} (< eps={std_eps}); "
                    "clamping std to eps for numerical stability."
                )
                warn_msgs.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                std[j] = torch.tensor(float(std_eps), dtype=torch.float32)
            else:
                std[j] = torch.tensor(raw_std, dtype=torch.float32)

        log_fit: YEdgeLog1pColumnScalerFitResult | None = None
        if log_column_name is not None:
            log_warn_msgs: list[str] = []
            if log_count <= 0:
                msg = (
                    f"compute_y_edge_log1p_column_scaler: column {log_column_name!r} has zero mask==1 values on the "
                    "fit split; using mean=0 std=1."
                )
                log_warn_msgs.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                log_fit = YEdgeLog1pColumnScalerFitResult(
                    mean=torch.tensor(0.0, dtype=torch.float32),
                    std=torch.tensor(1.0, dtype=torch.float32),
                    column=log_column_name,
                    count=0,
                    warnings=log_warn_msgs,
                )
            else:
                log_mean = log_total / float(log_count)
                log_var = max(log_total_sq / float(log_count) - log_mean * log_mean, 0.0)
                log_std = math.sqrt(log_var)
                if not math.isfinite(log_std) or log_std < float(std_eps):
                    msg = (
                        f"compute_y_edge_log1p_column_scaler: column {log_column_name!r} has std={log_std!r} "
                        f"(< eps={std_eps}); clamping std to eps for numerical stability."
                    )
                    log_warn_msgs.append(msg)
                    warnings.warn(msg, UserWarning, stacklevel=2)
                    log_std = float(std_eps)
                log_fit = YEdgeLog1pColumnScalerFitResult(
                    mean=torch.tensor(float(log_mean), dtype=torch.float32),
                    std=torch.tensor(float(log_std), dtype=torch.float32),
                    column=log_column_name,
                    count=int(log_count),
                    warnings=log_warn_msgs,
                )

        physical_quantiles: dict[str, dict[str, float]] = {}
        if physical_quantile_column_name is not None and physical_quantile_values:
            quantile_tensor = torch.as_tensor(physical_quantile_values, dtype=torch.float64)
            physical_quantiles[physical_quantile_column_name] = {
                f"p{int(round(float(prob) * 100)):02d}": float(
                    torch.quantile(quantile_tensor, float(prob)).item()
                )
                for prob in physical_quantile_probs
            }

        return (
            YEdgeScalerFitResult(
                mean=mean,
                std=std,
                columns=columns,
                masked_value_counts_per_column=counts,
                warnings=warn_msgs,
                physical_quantiles=physical_quantiles,
            ),
            log_fit,
        )

    def _load_spec(self, process_id: str) -> ProcessSpec:
        process_label = f"Process{_canonical_process_number(process_id)}"
        if self.data_cfg.cache_process_specs and process_label in self._spec_cache:
            return self._spec_cache[process_label]

        spec_dir = (self.project_root / self.data_cfg.process_spec_dir).resolve()
        matches = sorted(spec_dir.glob(self.data_cfg.process_spec_pattern))
        path_map = {fp.stem.replace("_Adjacency_Matrix", ""): fp for fp in matches}
        if process_label not in path_map:
            raise FileNotFoundError(
                f"No process spec file found for process_id={process_id!r} under {spec_dir} "
                f"with pattern {self.data_cfg.process_spec_pattern!r}. "
                f"Available ids: {sorted(path_map.keys())[:10]}{'...' if len(path_map) > 10 else ''}"
            )
        spec = parse_process_file(
            str(path_map[process_label]),
            process_id=process_label,
        )
        if self.data_cfg.cache_process_specs:
            self._spec_cache[process_label] = spec
        return spec

    def _build_v3_canonical_sample(
        self,
        *,
        spec: ProcessSpec,
        process_id: str,
        row: pd.Series,
        data_row: Mapping[str, float],
    ) -> tuple[GraphSample, Dict[str, int], pd.DataFrame]:
        assert self._v3_nodes is not None and self._v3_edges is not None and self._v3_answers is not None
        pid = _canonical_process_number(process_id)
        nodes_df = self._v3_nodes_for_pid(pid)
        base_node_names = [str(x) for x in nodes_df["node_name"].tolist()]
        topology = self._independent_feed_topology_bindings(pid, spec)
        node_names = base_node_names + [
            item.feed_node_name for item in topology
        ]
        node_to_idx = {name: i for i, name in enumerate(node_names)}

        edges_df = self._v3_edges_for_pid(pid)
        base_edge_ids = [str(x) for x in edges_df["canonical_edge_id"].tolist()]
        edge_ids = base_edge_ids + [item.context_edge_id for item in topology]
        edge_id_to_pos = {eid: i for i, eid in enumerate(edge_ids)}

        answers_df = self._v3_answers_for_pid(pid)
        slot_edge_index: Dict[str, int] = {}
        for slot, task_name in V3_SLOT_TO_TASK_NAME.items():
            sub = answers_df[answers_df["task_name"] == task_name]
            if len(sub) != 1:
                raise ValueError(
                    f"v3 target_answer_edges: expected exactly one row for process {pid} task_name={task_name!r}, "
                    f"got {len(sub)}"
                )
            arow = sub.iloc[0]
            eid = str(arow["canonical_answer_edge_id"])
            if eid not in edge_id_to_pos:
                raise ValueError(
                    f"v3 canonical_answer_edge_id {eid!r} for {task_name} not found in canonical_edges for process {pid}."
                )
            if str(arow["answer_dst_node"]) != "V_OUTPUT":
                raise ValueError(
                    f"v3 answer edge {eid} for {task_name} must have answer_dst_node==V_OUTPUT, "
                    f"got {arow['answer_dst_node']!r}."
                )
            erow = edges_df.loc[edges_df["canonical_edge_id"].astype(str) == eid].iloc[0]
            if str(erow["dst_node"]) != "V_OUTPUT":
                raise ValueError(
                    f"v3 answer edge {eid} for {task_name} must have canonical_edges.dst_node==V_OUTPUT, "
                    f"got {erow['dst_node']!r}."
                )
            slot_edge_index[slot] = int(edge_id_to_pos[eid])

        row_id = row.get("ID")
        if row_id is None or pd.isna(row_id):
            raise KeyError("v3 stream join requires an 'ID' column in the dataset row.")
        stream_rows = self._stream_rows_for_process_id(process_id, row_id)
        stream_by_name = self._canonical_stream_index_for_sample(
            stream_rows,
            process_id=process_id,
            sample_id=row_id,
        )
        self._enrich_data_row_with_v4_stream_targets(
            data_row=data_row,
            process_id=process_id,
            stream_by_name=stream_by_name,
        )

        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_roles: list[int] = []
        edge_property_roles: list[int] = []
        edge_stream_ids: list[int] = []
        edge_struct_attr: list[list[float]] = []
        edge_struct_mask: list[list[int]] = []
        y_edge_true: list[list[float]] = []
        y_edge_mask: list[float] = []
        edge_names: list[str] = []
        edge_feature_mask: list[float] = []
        edge_is_context: list[float] = []
        edge_is_predictable: list[float] = []
        edge_is_supervised: list[float] = []
        edge_is_target: list[float] = []
        edge_pinn_mask: list[float] = []
        target_edge_ids = self._target_edge_ids_for_pid(pid)
        if str(getattr(self.data_cfg, "edge_target_columns_mode", "auto")).lower() == "explicit":
            explicit = getattr(self.data_cfg, "edge_target_columns", None)
            if not explicit:
                raise ValueError("edge_target_columns_mode=explicit requires data.edge_target_columns (non-empty list).")
            edge_target_columns = [str(c) for c in explicit]
        else:
            edge_target_columns = list(STREAM_EDGE_FEATURE_SLOTS)

        for _, er in edges_df.iterrows():
            s_name = str(er["src_node"])
            d_name = str(er["dst_node"])
            if s_name not in node_to_idx or d_name not in node_to_idx:
                raise KeyError(
                    f"v3 edge references unknown node: {s_name}->{d_name} (canonical_edge_id={er.get('canonical_edge_id')})"
                )
            edge_src.append(node_to_idx[s_name])
            edge_dst.append(node_to_idx[d_name])
            role = str(er.get("stream_role", "unknown"))
            edge_roles.append(STREAM_ROLE_TO_IDX.get(role, STREAM_ROLE_TO_IDX["unknown"]))
            edge_property_roles.append(
                property_stream_role_id(role, er.get("dst_node_raw", ""))
            )
            raw_stream_key, stream_key, stream_row = self._resolve_v3_stream_row(
                edge_row=er,
                stream_by_name=stream_by_name,
                process_id=process_id,
                sample_id=row_id,
            )
            edge_names.append(raw_stream_key)
            ok = stream_row is not None and bool(stream_key)
            edge_stream_ids.append(self._stable_stream_id(stream_key) if stream_key else 0)
            target_values: list[float] = []
            if ok:
                for feat_slot in edge_target_columns:
                    if feat_slot in stream_row.index and not pd.isna(stream_row[feat_slot]):
                        target_values.append(float(stream_row[feat_slot]))
                    else:
                        target_values.append(0.0)
                edge_feature_mask.append(1.0)
                y_edge_mask.append(1.0)
            else:
                target_values = [0.0] * len(edge_target_columns)
                edge_feature_mask.append(0.0)
                y_edge_mask.append(0.0)
            join_ok = 1.0 if ok else 0.0
            has_key = 1.0 if stream_key else 0.0
            edge_struct_attr.append(
                _edge_struct_values(
                    is_input_edge=er.get("is_input_edge", False),
                    is_output_edge=er.get("is_output_edge", False),
                    is_internal_edge=er.get("is_internal_edge", False),
                    has_stream_key=has_key,
                    join_ok=join_ok,
                    data_cfg=self.data_cfg,
                )
            )
            edge_struct_mask.append([1] * len(_edge_struct_feature_columns(self.data_cfg)))
            y_edge_true.append(target_values)
            edge_is_context.append(0.0)
            edge_is_predictable.append(1.0)
            edge_is_supervised.append(float(join_ok))
            edge_is_target.append(
                1.0 if str(er.get("canonical_edge_id")) in target_edge_ids else 0.0
            )
            edge_pinn_mask.append(1.0)

        for item in topology:
            if item.feed_node_name not in node_to_idx:
                raise RuntimeError(
                    f"Process{pid}: F2 feed node {item.feed_node_name!r} was not added."
                )
            if item.destination_node_name not in node_to_idx:
                raise RuntimeError(
                    f"Process{pid}: F2 destination "
                    f"{item.destination_node_name!r} is absent."
                )
            edge_src.append(node_to_idx[item.feed_node_name])
            edge_dst.append(node_to_idx[item.destination_node_name])
            edge_roles.append(STREAM_ROLE_TO_IDX["feed"])
            edge_property_roles.append(property_stream_role_id("feed", ""))
            edge_stream_ids.append(
                self._stable_stream_id(
                    f"F2_CONTEXT::Process{pid}::{item.feed_name}"
                )
            )
            edge_struct_attr.append(
                _edge_struct_values(
                    is_input_edge=True,
                    is_output_edge=False,
                    is_internal_edge=False,
                    has_stream_key=0.0,
                    join_ok=0.0,
                    data_cfg=self.data_cfg,
                )
            )
            edge_struct_mask.append(
                [1] * len(_edge_struct_feature_columns(self.data_cfg))
            )
            edge_names.append("")
            edge_feature_mask.append(0.0)
            y_edge_true.append([0.0] * len(edge_target_columns))
            y_edge_mask.append(0.0)
            edge_is_context.append(1.0)
            edge_is_predictable.append(0.0)
            edge_is_supervised.append(0.0)
            edge_is_target.append(0.0)
            edge_pinn_mask.append(0.0)

        incoming_map = build_incoming_node_map(spec)
        v_input_index = (
            self._canonical_v_input_node_index(pid)
            if self._known_feed_cfg.enabled
            and self._known_feed_cfg.mode == "v_input_node"
            else None
        )
        feed_values, feed_masks = (
            self._resolve_known_feed_values(
                pid=pid,
                spec=spec,
                data_row=data_row,
                incoming_map=incoming_map,
            )
            if self._known_feed_cfg.enabled
            else ([0.0] * len(KNOWN_FEED_FEATURE_NAMES), [0] * len(KNOWN_FEED_FEATURE_NAMES))
        )
        feed_name_to_index = {
            name: index for index, name in enumerate(KNOWN_FEED_NAMES)
        }
        topology_by_node = {
            item.feed_node_name: item for item in topology
        }
        x_role: list[int] = []
        x_unit: list[int] = []
        x_hx_role: list[int] = []
        x_oper: list[list[float]] = []
        x_oper_mask: list[list[int]] = []
        node_is_context: list[float] = []
        node_pinn_mask: list[float] = []
        for node_index, node_name in enumerate(node_names):
            if node_index == v_input_index:
                tail_values, tail_masks = self._known_feed_node_tail(
                    node_name=node_name,
                    feed_values=feed_values,
                    feed_masks=feed_masks,
                    v_input_with_raw_feeds=True,
                )
                x_role.append(ROLE_TO_IDX["input_virtual"])
                x_unit.append(UNIT_TO_IDX["input_virtual"])
                x_hx_role.append(HX_ROLE_TO_IDX[None])
                x_oper.append([0.0] * len(OPER_FEATURE_SLOTS) + tail_values)
                x_oper_mask.append([0] * len(OPER_FEATURE_SLOTS) + tail_masks)
                node_is_context.append(0.0)
                node_pinn_mask.append(1.0)
                continue
            if node_name in topology_by_node:
                feed_index = feed_name_to_index[
                    topology_by_node[node_name].feed_name
                ]
                node_feed_values, node_feed_masks = self._known_feed_node_tail(
                    node_name=node_name,
                    feed_values=feed_values,
                    feed_masks=feed_masks,
                    independent_feed_index=feed_index,
                )
                x_role.append(ROLE_TO_IDX["source"])
                x_unit.append(UNIT_TO_IDX["stream"])
                x_hx_role.append(HX_ROLE_TO_IDX[None])
                x_oper.append(
                    [0.0] * len(OPER_FEATURE_SLOTS) + node_feed_values
                )
                x_oper_mask.append(
                    [0] * len(OPER_FEATURE_SLOTS) + node_feed_masks
                )
                node_is_context.append(1.0)
                node_pinn_mask.append(0.0)
                continue
            if node_name == "V_INPUT":
                x_role.append(ROLE_TO_IDX["input_virtual"])
                x_unit.append(UNIT_TO_IDX["input_virtual"])
                x_hx_role.append(HX_ROLE_TO_IDX[None])
                values = [0.0] * len(OPER_FEATURE_SLOTS)
                masks = [0] * len(OPER_FEATURE_SLOTS)
                if self._known_feed_cfg.enabled:
                    tail_values, tail_masks = self._known_feed_node_tail(
                        node_name=node_name,
                        feed_values=feed_values,
                        feed_masks=feed_masks,
                    )
                    values += tail_values
                    masks += tail_masks
                x_oper.append(values)
                x_oper_mask.append(masks)
                node_is_context.append(0.0)
                node_pinn_mask.append(1.0)
                continue
            if node_name == "V_OUTPUT":
                x_role.append(ROLE_TO_IDX["output_virtual"])
                x_unit.append(UNIT_TO_IDX["output_virtual"])
                x_hx_role.append(HX_ROLE_TO_IDX[None])
                values = [0.0] * len(OPER_FEATURE_SLOTS)
                masks = [0] * len(OPER_FEATURE_SLOTS)
                if self._known_feed_cfg.enabled:
                    tail_values, tail_masks = self._known_feed_node_tail(
                        node_name=node_name,
                        feed_values=feed_values,
                        feed_masks=feed_masks,
                    )
                    values += tail_values
                    masks += tail_masks
                x_oper.append(values)
                x_oper_mask.append(masks)
                node_is_context.append(0.0)
                node_pinn_mask.append(1.0)
                continue
            node = spec.nodes.get(node_name)
            x_role.append(ROLE_TO_IDX["unit"])
            x_unit.append(UNIT_TO_IDX[node.type_canonical] if node is not None else UNIT_TO_IDX["stream"])
            x_hx_role.append(HX_ROLE_TO_IDX[node.hx_role] if node is not None else HX_ROLE_TO_IDX[None])
            values_n: list[float] = []
            masks_n: list[int] = []
            for slot in OPER_FEATURE_SLOTS:
                cell_spec = node.features[slot] if node is not None else None
                if cell_spec is None:
                    values_n.append(0.0)
                    masks_n.append(0)
                    continue
                value, mask = resolve_feature_value(
                    cell_spec,
                    ResolveContext(
                        process_spec=spec,
                        data_row=dict(data_row),
                        incoming_map=incoming_map,
                        node_name=node_name,
                        slot=slot,
                        passthrough_policy=self.data_cfg.passthrough_policy,
                    ),
                )
                values_n.append(value)
                masks_n.append(mask)
            if self._known_feed_cfg.enabled:
                tail_values, tail_masks = self._known_feed_node_tail(
                    node_name=node_name,
                    feed_values=feed_values,
                    feed_masks=feed_masks,
                )
                values_n += tail_values
                masks_n += tail_masks
            x_oper.append(values_n)
            x_oper_mask.append(masks_n)
            node_is_context.append(0.0)
            node_pinn_mask.append(1.0)

        base_targets = build_graph_sample(spec, data_row, passthrough_policy=self.data_cfg.passthrough_policy).targets
        node_q, node_w, node_qw_valid, excl_mass, excl_component, excl_atom, excl_energy = self._node_qw_terms_for_sample(
            pid=pid,
            row_id=row_id,
            node_names=node_names,
        )
        for node_index, is_context in enumerate(node_is_context):
            if is_context <= 0.5:
                continue
            node_qw_valid[node_index] = 0.0
            excl_mass[node_index] = 1.0
            excl_component[node_index] = 1.0
            excl_atom[node_index] = 1.0
            excl_energy[node_index] = 1.0
        hx_pair_current_edge_index, hx_paired_edge_index, hx_pair_mask, hx_pair_side = (
            self._build_hx_pair_metadata(
                pid=pid,
                edges_df=edges_df,
                all_edge_ids=edge_ids,
            )
        )
        sample = GraphSample(
            process_id=process_id,
            node_names=node_names,
            edge_index=[edge_src, edge_dst],
            x_role=x_role,
            x_unit=x_unit,
            x_hx_role=x_hx_role,
            x_oper=x_oper,
            x_oper_mask=x_oper_mask,
            targets=base_targets,
            edge_stream_role=edge_roles,
            edge_property_stream_role=edge_property_roles,
            edge_stream_id=edge_stream_ids,
            edge_oper=edge_struct_attr,
            edge_oper_mask=edge_struct_mask,
            edge_stream_names=edge_names,
            edge_feature_mask=edge_feature_mask,
            node_is_context=node_is_context,
            node_pinn_mask=node_pinn_mask,
            edge_is_context=edge_is_context,
            edge_is_predictable=edge_is_predictable,
            edge_is_supervised=edge_is_supervised,
            edge_is_target=edge_is_target,
            edge_pinn_mask=edge_pinn_mask,
            hx_pair_current_edge_index=hx_pair_current_edge_index,
            hx_paired_edge_index=hx_paired_edge_index,
            hx_pair_mask=hx_pair_mask,
            hx_pair_side=hx_pair_side,
            graph_feed_values=(
                list(feed_values) if self._feed_head_cfg.enabled else []
            ),
            graph_feed_mask=(
                list(feed_masks) if self._feed_head_cfg.enabled else []
            ),
            edge_struct_attr=edge_struct_attr,
            y_edge_true=y_edge_true,
            y_edge_mask=y_edge_mask,
            canonical_edge_ids=edge_ids,
            answer_edge_pos=dict(slot_edge_index),
            answer_edge_ids={
                slot: str(
                    answers_df.loc[answers_df["task_name"] == task_name, "canonical_answer_edge_id"].iloc[0]
                )
                for slot, task_name in V3_SLOT_TO_TASK_NAME.items()
            },
            edge_target_columns=edge_target_columns,
            node_q=node_q,
            node_w=node_w,
            node_qw_valid_mask=node_qw_valid,
            node_balance_exclude_mass=excl_mass,
            node_balance_exclude_component=excl_component,
            node_balance_exclude_atom=excl_atom,
            node_balance_exclude_energy=excl_energy,
        )
        if getattr(self.data_cfg, "log_v3_graph_stats_once", True):
            self._maybe_log_v3_graph_stats(
                process_id=process_id,
                pid=pid,
                sample=sample,
                slot_edge_index=slot_edge_index,
                answers_df=answers_df,
                edges_df=edges_df,
            )
        return sample, slot_edge_index, edges_df

    def _maybe_log_v3_graph_stats(
        self,
        *,
        process_id: str,
        pid: int,
        sample: GraphSample,
        slot_edge_index: Dict[str, int],
        answers_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> None:
        if process_id in self._v3_logged_stats:
            return
        self._v3_logged_stats.add(process_id)
        m = sample.edge_feature_mask
        n_feat_m1 = sum(1 for x in m if float(x) >= 1.0)
        n_feat_m0 = sum(1 for x in m if float(x) < 1.0)
        n_edges = len(m)
        n_context_nodes = sum(
            1 for value in sample.node_is_context if float(value) > 0.5
        )
        n_context_edges = sum(
            1 for value in sample.edge_is_context if float(value) > 0.5
        )
        n_pinn_edges = sum(
            1 for value in sample.edge_pinn_mask if float(value) > 0.5
        )
        n_target_edges = sum(
            1 for value in sample.edge_is_target if float(value) > 0.5
        )
        n_with_key = 0
        for _, er in edges_df.iterrows():
            sk = _clean_csv_text(er.get("main_data_stream_key"))
            if sk:
                n_with_key += 1
        n_without_key = n_edges - n_with_key
        th = answers_df.loc[answers_df["task_name"] == "target_h2", "canonical_answer_edge_id"].iloc[0]
        tg = answers_df.loc[answers_df["task_name"] == "tailgas_co2", "canonical_answer_edge_id"].iloc[0]
        msg = (
            f"[v3 dataloader] process_id={process_id!r} (pid={pid}) num_nodes={len(sample.node_names)} "
            f"num_edges={n_edges} "
            f"num_mapped_edges={n_with_key} num_unmapped_edges={n_without_key} "
            f"y_edge_true_shape=({len(sample.y_edge_true)}, "
            f"{len(sample.y_edge_true[0]) if sample.y_edge_true else 0}) "
            f"y_edge_mask_shape=({len(sample.y_edge_mask)},) y_edge_mask_sum={sum(sample.y_edge_mask):.0f} "
            f"edge_struct_attr_shape=({len(sample.edge_struct_attr)}, "
            f"{len(sample.edge_struct_attr[0]) if sample.edge_struct_attr else 0}) "
            f"num_edge_feature_mask_1={n_feat_m1} num_edge_feature_mask_0={n_feat_m0} "
            f"context_nodes={n_context_nodes} context_edges={n_context_edges} "
            f"pinn_edges={n_pinn_edges} target_edges={n_target_edges} "
            f"target_h2_answer_edge_id={th} tailgas_co2_answer_edge_id={tg} "
            f"target_h2_answer_edge_pos_local={slot_edge_index.get('target')} "
            f"tailgas_co2_answer_edge_pos_local={slot_edge_index.get('tailgas')}"
        )
        print(msg, file=sys.stderr)

    def _resolve_fixed_task_variable(
        self,
        *,
        process_id: str,
        slot_name: str,
        fallback_variable: str | None,
    ) -> str | None:
        """
        Return process-aware fixed-task target column when available.

        Priority:
        1) PROCESS_FIXED_TARGET_COLUMNS[process_id][slot_name]
        2) config fallback (fixed_tasks.<slot>.target_variable)
        """
        if slot_name in FIXED_SLOTS:
            mapped = PROCESS_FIXED_TARGET_COLUMNS.get(process_id, {}).get(slot_name)
            if mapped:
                return mapped
        return fallback_variable

    @staticmethod
    def _process_number(process_id: str) -> str:
        normalized = canonicalize_process_id(process_id)
        if not normalized:
            raise ValueError(f"Cannot parse process number from process_id={process_id!r}")
        return str(int(normalized[1:]))

    @staticmethod
    def _stable_stream_id(stream_name: str) -> int:
        return int(zlib.crc32(str(stream_name).encode("utf-8")) & 0xFFFFFFFF)

    def _load_stream_mapping(self) -> pd.DataFrame:
        if self._stream_mapping_frame is not None:
            return self._stream_mapping_frame
        path = (self.project_root / self.data_cfg.stream_edge_mapping_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"stream edge mapping CSV not found: {path}")
        frame = pd.read_csv(path, dtype=str).fillna("")
        self._stream_mapping_frame = frame
        return frame

    def _mapping_rows_for_process(self, process_id: str) -> pd.DataFrame:
        process_num = self._process_number(process_id)
        frame = self._load_stream_mapping()
        rows = frame[frame["process_id"].astype(str) == process_num].copy()
        if rows.empty:
            raise ValueError(f"No stream edge mapping rows found for process_id={process_id!r}.")
        return rows.reset_index(drop=True)

    def _stream_rows_for_process_id(self, process_id: str, row_id: Any) -> pd.DataFrame:
        process_num = self._process_number(process_id)
        row_key = _id_key(row_id)
        cache_key = f"{process_num}:{row_key}"
        if cache_key in self._stream_rows_cache:
            rows = self._stream_rows_cache.pop(cache_key)
            self._stream_rows_cache[cache_key] = rows
            return rows
        frame = self._stream_frame_for_process_num(process_num)
        try:
            rows = frame.loc[[row_key]].copy()
        except KeyError:
            raise ValueError(f"No stream rows found for process_id={process_id!r}, ID={row_id!r}.")
        cache_size = int(getattr(self.data_cfg, "stream_row_cache_size", 128) or 0)
        if cache_size > 0:
            self._stream_rows_cache[cache_key] = rows
            while len(self._stream_rows_cache) > cache_size:
                self._stream_rows_cache.popitem(last=False)
        return rows

    def _stream_csv_path(self, process_num: str) -> Path:
        return (
            self.project_root
            / self.data_cfg.stream_data_dir
            / f"{process_num}.Process_Streams.csv"
        ).resolve()

    def _canonical_stream_index_for_sample(
        self,
        stream_rows: pd.DataFrame,
        *,
        process_id: str,
        sample_id: Any,
    ) -> Mapping[str, pd.Series]:
        process_num = self._process_number(process_id)
        return build_canonical_stream_row_index(
            stream_rows,
            process_id=process_id,
            sample_id=sample_id,
            csv_path=self._stream_csv_path(process_num),
        )

    def _stream_frame_for_process_num(self, process_num: str) -> pd.DataFrame:
        if process_num not in self._stream_process_frame_cache:
            path = self._stream_csv_path(process_num)
            if not path.is_file():
                raise FileNotFoundError(f"Process stream CSV not found: {path}")
            required_cols = {"ID", "Stream_Name"}
            required_cols.update(str(c) for c in getattr(self.data_cfg, "edge_target_columns", []) or [])
            required_cols.update(STREAM_EDGE_FEATURE_SLOTS)
            frame = pd.read_csv(
                path,
                usecols=lambda c: str(c) in required_cols,
                dtype={"Stream_Name": str},
            )
            frame["_ID_STR"] = frame["ID"].map(_id_key)
            frame[STREAM_KEY_CANONICAL_COLUMN] = frame["Stream_Name"].map(
                lambda value: canonicalize_stream_key(
                    value,
                    process_id=process_num,
                    source=path,
                )
            )
            allowed_ids = self._stream_allowed_ids_by_process.get(process_num)
            if allowed_ids:
                frame = frame[frame["_ID_STR"].isin(allowed_ids)].copy()
            keyed = frame[
                frame["_ID_STR"].astype(bool)
                & frame[STREAM_KEY_CANONICAL_COLUMN].astype(bool)
            ]
            duplicates = keyed.duplicated(
                subset=["_ID_STR", STREAM_KEY_CANONICAL_COLUMN],
                keep=False,
            )
            if bool(duplicates.any()):
                bad = keyed.loc[
                    duplicates,
                    ["_ID_STR", "Stream_Name", STREAM_KEY_CANONICAL_COLUMN],
                ].head(20)
                raise ValueError(
                    "Canonical stream-key collision; refusing ambiguous join. "
                    f"process_id={canonicalize_process_id(process_num)!r} "
                    f"csv_path={str(path)!r} examples={bad.to_dict(orient='records')!r}"
                )
            if self._use_v3_canonical:
                process_id = f"Process{int(process_num)}"
                available_keys = sorted(
                    set(
                        keyed[STREAM_KEY_CANONICAL_COLUMN]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )
                )
                available_key_set = set(available_keys)
                for _, edge_row in self._v3_edges_for_pid(
                    int(process_num)
                ).iterrows():
                    raw_key = _clean_csv_text(
                        edge_row.get("main_data_stream_key")
                    )
                    canonical_key = canonicalize_stream_key(
                        raw_key,
                        process_id=process_id,
                        source="canonical_edges.csv",
                    )
                    if canonical_key and canonical_key not in available_key_set:
                        raise missing_required_stream_key_error(
                            process_id=process_id,
                            sample_id="<startup-domain-audit>",
                            canonical_edge_id=edge_row.get("canonical_edge_id"),
                            raw_graph_key=raw_key,
                            canonical_graph_key=canonical_key,
                            available_csv_keys=available_keys,
                            csv_path=path,
                        )
            for col in frame.columns:
                if col in {
                    "ID",
                    "_ID_STR",
                    "Stream_Name",
                    STREAM_KEY_CANONICAL_COLUMN,
                }:
                    continue
                frame[col] = pd.to_numeric(frame[col], errors="coerce", downcast="float")
            frame = frame.set_index("_ID_STR", drop=False)
            self._stream_process_frame_cache[process_num] = frame
        return self._stream_process_frame_cache[process_num]

    def _main_frame_for_process_num(self, process_num: str) -> pd.DataFrame:
        if process_num not in self._main_process_frame_cache:
            main_dir = getattr(self.data_cfg, "main_data_dir", "data/main_data")
            path = (self.project_root / str(main_dir) / f"{process_num}.Process_Main.csv").resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Process main CSV not found: {path}")
            frame = pd.read_csv(path)
            if "ID" not in frame.columns:
                raise KeyError(f"Process main CSV requires an ID column: {path}")
            frame["_ID_STR"] = frame["ID"].map(_id_key)
            for col in frame.columns:
                if col in {"ID", "_ID_STR"}:
                    continue
                frame[col] = pd.to_numeric(frame[col], errors="coerce", downcast="float")
            frame = frame.set_index("_ID_STR", drop=False)
            self._main_process_frame_cache[process_num] = frame
        return self._main_process_frame_cache[process_num]

    def _node_qw_terms_for_sample(
        self,
        *,
        pid: int,
        row_id: Any,
        node_names: Sequence[str],
    ) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
        q = [0.0] * len(node_names)
        w = [0.0] * len(node_names)
        valid = [1.0] * len(node_names)
        excl_mass = [0.0] * len(node_names)
        excl_component = [0.0] * len(node_names)
        excl_atom = [0.0] * len(node_names)
        excl_energy = [0.0] * len(node_names)

        if int(pid) == 3:
            for i, name in enumerate(node_names):
                if str(name) in P03_INCOMPLETE_BALANCE_NODES:
                    excl_mass[i] = 1.0
                    excl_component[i] = 1.0
                    excl_energy[i] = 1.0

        bad_energy_nodes = self._node_energy_exclude_cache.get(_sample_cache_key(f"Process{int(pid)}", row_id), set())
        if bad_energy_nodes:
            for i, name in enumerate(node_names):
                if str(name) in bad_energy_nodes:
                    excl_energy[i] = 1.0

        mapping = PROCESS_NODE_ENERGY_MAP.get(int(pid), {})
        if not mapping:
            return q, w, valid, excl_mass, excl_component, excl_atom, excl_energy

        frame = self._main_frame_for_process_num(str(pid))
        row_key = _id_key(row_id)
        try:
            main_row = frame.loc[row_key]
            if isinstance(main_row, pd.DataFrame):
                main_row = main_row.iloc[0]
        except KeyError:
            return q, w, [0.0] * len(node_names), excl_mass, excl_component, excl_atom, excl_energy

        columns_lower = {str(col).lower(): str(col) for col in frame.columns}
        for i, node_name in enumerate(node_names):
            spec = mapping.get(str(node_name))
            if not spec:
                continue
            for field_name, target in (("q", q), ("w", w)):
                col = spec.get(field_name)
                if not col:
                    continue
                real_col = columns_lower.get(str(col).lower())
                if real_col is None:
                    valid[i] = 0.0
                    continue
                value = main_row.get(real_col)
                if pd.isna(value):
                    valid[i] = 0.0
                    continue
                target[i] = float(value)
        return q, w, valid, excl_mass, excl_component, excl_atom, excl_energy

    @staticmethod
    def _normalized_identifier(value: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(value).upper())

    def _resolve_fixed_slot_edge_index(
        self,
        mapping_rows: pd.DataFrame,
        *,
        process_id: str,
        preferred_variable: str | None,
        slot_name: str,
    ) -> int:
        process_fixed = PROCESS_FIXED_EDGE_STREAM_NAMES.get(process_id, {}).get(slot_name)
        if process_fixed:
            fixed_rows = mapping_rows[
                (mapping_rows["stream_name"].astype(str) == process_fixed)
                | (mapping_rows["stream_name_norm"].astype(str) == process_fixed)
            ]
            fixed_output = fixed_rows[fixed_rows["dst_node"].astype(str) == "V_OUTPUT"]
            if not fixed_output.empty:
                return int(fixed_output.index[0])
            if not fixed_rows.empty:
                return int(fixed_rows.index[0])

        if preferred_variable:
            target_key = self._normalized_identifier(preferred_variable)
            target_key = re.sub(r"(MOLEFLOW|MOLE|FLOW|FRAC|FRACTION)$", "", target_key)
            target_key = re.sub(r"(H2O|CH4|CO2|H2|CO|O2|N2)$", "", target_key)
            candidates: list[tuple[int, int]] = []
            for idx, row in mapping_rows.iterrows():
                for col in ("stream_name_norm", "stream_name"):
                    stream_key = self._normalized_identifier(str(row.get(col, "")))
                    if not stream_key:
                        continue
                    score_bonus = 0
                    if str(row.get("dst_node", "")) == "V_OUTPUT":
                        score_bonus += 500
                    role = str(row.get("stream_role", ""))
                    if slot_name == "target" and "product" in role:
                        score_bonus += 200
                    if slot_name == "tailgas" and ("exhaust" in role or role in {"output", "zero_or_side_output"}):
                        score_bonus += 200
                    if target_key == stream_key:
                        candidates.append((idx, 1000 + len(stream_key) + score_bonus))
                    elif target_key.startswith(stream_key) or stream_key in target_key:
                        candidates.append((idx, len(stream_key) + score_bonus))
            if candidates:
                candidates.sort(key=lambda item: item[1], reverse=True)
                return int(candidates[0][0])

        role_hint = "product" if slot_name == "target" else "exhaust|output|zero_or_side_output"
        role_matches = mapping_rows[
            mapping_rows["stream_role"].astype(str).str.contains(role_hint, case=False, regex=True)
        ]
        role_output = role_matches[role_matches["dst_node"].astype(str) == "V_OUTPUT"]
        if not role_output.empty:
            return int(role_output.index[0])
        if not role_matches.empty:
            return int(role_matches.index[0])
        target_edges = mapping_rows[
            (mapping_rows["is_target_edge"].astype(str) == "1")
            & (mapping_rows["dst_node"].astype(str) == "V_OUTPUT")
        ]
        if not target_edges.empty:
            return int(target_edges.index[0])
        return 0

    def _build_stream_edge_sample(
        self,
        *,
        spec: ProcessSpec,
        process_id: str,
        row: pd.Series,
        data_row: Mapping[str, float],
    ) -> tuple[GraphSample, pd.DataFrame]:
        mapping_rows = self._mapping_rows_for_process(process_id)
        row_id = row.get("ID")
        if row_id is None or pd.isna(row_id):
            raise KeyError("stream_edge topology requires an 'ID' column to join Process_Streams rows.")
        stream_rows = self._stream_rows_for_process_id(process_id, row_id)
        stream_by_name = self._canonical_stream_index_for_sample(
            stream_rows,
            process_id=process_id,
            sample_id=row_id,
        )

        node_names: list[str] = []
        for _, item in mapping_rows.iterrows():
            for key in ("src_node", "dst_node"):
                name = str(item[key])
                if name not in node_names:
                    node_names.append(name)
        node_to_idx = {name: idx for idx, name in enumerate(node_names)}

        x_role: list[int] = []
        x_unit: list[int] = []
        x_hx_role: list[int] = []
        x_oper: list[list[float]] = []
        x_oper_mask: list[list[int]] = []
        incoming_map = build_incoming_node_map(spec)
        for node_name in node_names:
            if node_name == "V_INPUT":
                x_role.append(ROLE_TO_IDX["input_virtual"])
                x_unit.append(UNIT_TO_IDX["input_virtual"])
                x_hx_role.append(HX_ROLE_TO_IDX[None])
                x_oper.append([0.0] * len(OPER_FEATURE_SLOTS))
                x_oper_mask.append([0] * len(OPER_FEATURE_SLOTS))
                continue
            if node_name == "V_OUTPUT":
                x_role.append(ROLE_TO_IDX["output_virtual"])
                x_unit.append(UNIT_TO_IDX["output_virtual"])
                x_hx_role.append(HX_ROLE_TO_IDX[None])
                x_oper.append([0.0] * len(OPER_FEATURE_SLOTS))
                x_oper_mask.append([0] * len(OPER_FEATURE_SLOTS))
                continue

            node = spec.nodes.get(node_name)
            x_role.append(ROLE_TO_IDX["unit"])
            x_unit.append(UNIT_TO_IDX[node.type_canonical] if node is not None else UNIT_TO_IDX["stream"])
            x_hx_role.append(HX_ROLE_TO_IDX[node.hx_role] if node is not None else HX_ROLE_TO_IDX[None])
            values: list[float] = []
            masks: list[int] = []
            for slot in OPER_FEATURE_SLOTS:
                cell_spec = node.features[slot] if node is not None else None
                if cell_spec is None:
                    values.append(0.0)
                    masks.append(0)
                    continue
                value, mask = resolve_feature_value(
                    cell_spec,
                    ResolveContext(
                        process_spec=spec,
                        data_row=dict(data_row),
                        incoming_map=incoming_map,
                        node_name=node_name,
                        slot=slot,
                        passthrough_policy=self.data_cfg.passthrough_policy,
                    ),
                )
                values.append(value)
                masks.append(mask)
            x_oper.append(values)
            x_oper_mask.append(masks)

        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_roles: list[int] = []
        edge_property_roles: list[int] = []
        edge_stream_ids: list[int] = []
        edge_struct_attr: list[list[float]] = []
        edge_struct_mask: list[list[int]] = []
        y_edge_true: list[list[float]] = []
        y_edge_mask: list[float] = []
        edge_names: list[str] = []
        for _, item in mapping_rows.iterrows():
            stream_name = str(item["stream_name"])
            stream_name_norm = canonicalize_stream_key(
                item.get("stream_name_norm", stream_name),
                process_id=process_id,
                source=self.data_cfg.stream_edge_mapping_path,
            )
            canonical_stream_name = canonicalize_stream_key(
                stream_name,
                process_id=process_id,
                source=self.data_cfg.stream_edge_mapping_path,
            )
            edge_src.append(node_to_idx[str(item["src_node"])])
            edge_dst.append(node_to_idx[str(item["dst_node"])])
            role = str(item.get("stream_role", "unknown"))
            edge_roles.append(STREAM_ROLE_TO_IDX.get(role, STREAM_ROLE_TO_IDX["unknown"]))
            edge_property_roles.append(
                property_stream_role_id(
                    role,
                    item.get("dst_node_raw", item.get("dst_node", "")),
                )
            )
            edge_stream_ids.append(self._stable_stream_id(stream_name_norm))
            edge_names.append(stream_name)
            stream_row = stream_by_name.get(canonical_stream_name)
            if stream_row is None:
                stream_row = stream_by_name.get(stream_name_norm)
            values = []
            for feat_slot in STREAM_EDGE_FEATURE_SLOTS:
                if stream_row is not None and feat_slot in stream_row and not pd.isna(stream_row[feat_slot]):
                    values.append(float(stream_row[feat_slot]))
                else:
                    values.append(0.0)
            join_ok = 1.0 if stream_row is not None else 0.0
            y_edge_true.append(values)
            y_edge_mask.append(join_ok)
            edge_struct_attr.append(
                _edge_struct_values(
                    is_input_edge=item.get("is_input_edge", False),
                    is_output_edge=item.get("is_output_edge", False),
                    is_internal_edge=item.get("is_internal_edge", False),
                    has_stream_key=1.0 if stream_name_norm else 0.0,
                    join_ok=join_ok,
                    data_cfg=self.data_cfg,
                )
            )
            edge_struct_mask.append([1] * len(_edge_struct_feature_columns(self.data_cfg)))

        return (
            GraphSample(
                process_id=process_id,
                node_names=node_names,
                edge_index=[edge_src, edge_dst],
                x_role=x_role,
                x_unit=x_unit,
                x_hx_role=x_hx_role,
                x_oper=x_oper,
                x_oper_mask=x_oper_mask,
                targets=build_graph_sample(spec, dict(data_row), passthrough_policy=self.data_cfg.passthrough_policy).targets,
                edge_stream_role=edge_roles,
                edge_property_stream_role=edge_property_roles,
                edge_stream_id=edge_stream_ids,
                edge_oper=edge_struct_attr,
                edge_oper_mask=edge_struct_mask,
                edge_stream_names=edge_names,
                edge_feature_mask=[],
                edge_struct_attr=edge_struct_attr,
                y_edge_true=y_edge_true,
                y_edge_mask=y_edge_mask,
                canonical_edge_ids=[],
                answer_edge_pos={},
                answer_edge_ids={},
                edge_target_columns=list(STREAM_EDGE_FEATURE_SLOTS),
            ),
            mapping_rows,
        )

    def _apply_normalization(self, sample: GraphSample) -> GraphSample:
        if not self.data_cfg.normalize_x_oper and not self.data_cfg.normalize_targets:
            return sample

        x_oper = torch.tensor(sample.x_oper, dtype=torch.float32)
        if self.data_cfg.normalize_x_oper:
            if self.oper_mean is None or self.oper_std is None:
                raise ValueError("normalize_x_oper=True but oper_mean/oper_std were not provided.")
            if int(self.oper_mean.numel()) != int(x_oper.shape[1]):
                raise ValueError(
                    "operating normalizer dimension mismatch: "
                    f"mean={int(self.oper_mean.numel())} x_oper={int(x_oper.shape[1])}. "
                    "Do not reuse an F0 normalizer/checkpoint for F1/F2."
                )
            std = self.oper_std.clone()
            std[std == 0] = 1.0
            x_oper = (x_oper - self.oper_mean) / std
            if self._known_feed_cfg.enabled:
                start = len(OPER_FEATURE_SLOTS)
                feed_mask = torch.tensor(
                    sample.x_oper_mask, dtype=x_oper.dtype
                )[:, start:]
                x_oper[:, start:] = torch.where(
                    feed_mask > 0,
                    x_oper[:, start:],
                    torch.zeros_like(x_oper[:, start:]),
                )

        graph_feed_values = list(sample.graph_feed_values)
        graph_feed_mask = list(sample.graph_feed_mask)
        if self._feed_head_cfg.enabled:
            if len(graph_feed_values) != len(KNOWN_FEED_NAMES) or len(
                graph_feed_mask
            ) != len(KNOWN_FEED_NAMES):
                raise ValueError(
                    "Feed-head graph tensors must contain CH4, AIR, WATER values and masks."
                )
            feed_tensor = torch.tensor(graph_feed_values, dtype=torch.float32)
            feed_mask_tensor = torch.tensor(graph_feed_mask, dtype=torch.float32)
            if self.data_cfg.normalize_x_oper:
                start = len(OPER_FEATURE_SLOTS)
                stop = start + len(KNOWN_FEED_NAMES)
                feed_std = self.oper_std[start:stop].clone()
                feed_std[feed_std == 0] = 1.0
                feed_tensor = (feed_tensor - self.oper_mean[start:stop]) / feed_std
            feed_tensor = torch.where(
                feed_mask_tensor > 0,
                feed_tensor,
                torch.zeros_like(feed_tensor),
            )
            graph_feed_values = feed_tensor.tolist()

        targets = sample.targets
        if self.data_cfg.normalize_targets:
            decoder_on = any(self.data_cfg.decoder_tasks[c].enabled for c in DECODER_CATEGORIES)
            aux_on = self.data_cfg.aux_task.enabled
            new_targets: Dict[str, Dict[str, float]] = {
                k: dict(v) if isinstance(v, dict) else {} for k, v in targets.items()
            }
            if decoder_on:
                for category in DECODER_CATEGORIES:
                    node_map = new_targets.get(category, {})
                    if not node_map:
                        continue
                    mean = self.target_mean.get(category)
                    std = self.target_std.get(category)
                    if mean is None or std is None:
                        raise ValueError(
                            f"normalize_targets=True but missing stats for category={category!r}."
                        )
                    denom = std if std != 0 else 1.0
                    new_targets[category] = {
                        k: (float(v) - mean) / denom for k, v in node_map.items()
                    }
            elif aux_on:
                slot_name = "aux"
                node_map = new_targets.get(slot_name, {})
                mean = self.target_mean.get(slot_name)
                std = self.target_std.get(slot_name)
                if mean is None or std is None:
                    raise ValueError(
                        f"normalize_targets=True but missing stats for slot={slot_name!r}."
                    )
                denom = std if std != 0 else 1.0
                new_targets[slot_name] = {
                    k: (float(v) - mean) / denom for k, v in node_map.items()
                }
            targets = new_targets

        return GraphSample(
            process_id=sample.process_id,
            node_names=sample.node_names,
            edge_index=sample.edge_index,
            x_role=sample.x_role,
            x_unit=sample.x_unit,
            x_hx_role=sample.x_hx_role,
            x_oper=x_oper.tolist(),
            x_oper_mask=sample.x_oper_mask,
            targets=targets,
            edge_stream_role=sample.edge_stream_role,
            edge_property_stream_role=sample.edge_property_stream_role,
            edge_stream_id=sample.edge_stream_id,
            edge_oper=sample.edge_oper,
            edge_oper_mask=sample.edge_oper_mask,
            edge_stream_names=sample.edge_stream_names,
            edge_feature_mask=list(sample.edge_feature_mask),
            node_is_context=list(sample.node_is_context),
            node_pinn_mask=list(sample.node_pinn_mask),
            edge_is_context=list(sample.edge_is_context),
            edge_is_predictable=list(sample.edge_is_predictable),
            edge_is_supervised=list(sample.edge_is_supervised),
            edge_is_target=list(sample.edge_is_target),
            edge_pinn_mask=list(sample.edge_pinn_mask),
            hx_pair_current_edge_index=list(sample.hx_pair_current_edge_index),
            hx_paired_edge_index=list(sample.hx_paired_edge_index),
            hx_pair_mask=list(sample.hx_pair_mask),
            hx_pair_side=list(sample.hx_pair_side),
            graph_feed_values=graph_feed_values,
            graph_feed_mask=graph_feed_mask,
            edge_struct_attr=sample.edge_struct_attr,
            y_edge_true=sample.y_edge_true,
            y_edge_mask=sample.y_edge_mask,
            canonical_edge_ids=sample.canonical_edge_ids,
            answer_edge_pos=sample.answer_edge_pos,
            answer_edge_ids=sample.answer_edge_ids,
            edge_target_columns=sample.edge_target_columns,
            node_q=sample.node_q,
            node_w=sample.node_w,
            node_qw_valid_mask=sample.node_qw_valid_mask,
            node_balance_exclude_mass=sample.node_balance_exclude_mass,
            node_balance_exclude_component=sample.node_balance_exclude_component,
            node_balance_exclude_atom=sample.node_balance_exclude_atom,
            node_balance_exclude_energy=sample.node_balance_exclude_energy,
        )

    def _extract_slot(
        self,
        *,
        slot_name: str,
        preferred_variable: str | None,
        row: pd.Series,
        sample: GraphSample,
        spec: ProcessSpec,
        row_idx: int,
        process_id: str,
    ) -> Tuple[float, float, int, Dict[str, Any]]:
        routing_meta = self._build_fixed_slot_routing_meta(
            slot_name=slot_name,
            preferred_variable=preferred_variable,
            sample=sample,
            spec=spec,
        )
        # Priority 1: direct numeric column in row (used for H2 / CO2 style supervision).
        if preferred_variable:
            raw = row.get(preferred_variable)
            if raw is not None and not pd.isna(raw):
                resolved_idx, resolved_name, resolved_role, resolution_mode = self._resolve_fixed_slot_node(
                    sample=sample,
                    spec=spec,
                    routing_meta=routing_meta,
                )
                routing_meta.update(
                    {
                        "source": "row_column",
                        "current_node_index": resolved_idx,
                        "current_node_name": resolved_name,
                        "current_node_role": resolved_role,
                        "resolution_mode": resolution_mode,
                    }
                )
                self._maybe_log_fixed_slot_routing(
                    slot_name=slot_name,
                    process_id=process_id,
                    row_idx=row_idx,
                    routing_meta=routing_meta,
                )
                return float(raw), 1.0, resolved_idx, routing_meta

        # Priority 2: graph-derived category from sample.targets (used for aux categories).
        if preferred_variable and preferred_variable in sample.targets and sample.targets[preferred_variable]:
            node_map = sample.targets[preferred_variable]
            first_node = next(iter(node_map))
            node_idx = sample.node_names.index(first_node) if first_node in sample.node_names else 0
            routing_meta.update(
                {
                    "source": "graph_preferred_variable",
                    "current_node_index": node_idx,
                    "current_node_name": first_node,
                    "current_node_role": spec.nodes[first_node].role if first_node in spec.nodes else "",
                }
            )
            self._maybe_log_fixed_slot_routing(
                slot_name=slot_name,
                process_id=process_id,
                row_idx=row_idx,
                routing_meta=routing_meta,
            )
            return float(sum(node_map.values())), 1.0, node_idx, routing_meta

        # Priority 3: fallback to slot-named entry in sample.targets, if present.
        if slot_name in sample.targets and sample.targets[slot_name]:
            node_map = sample.targets[slot_name]
            first_node = next(iter(node_map))
            node_idx = sample.node_names.index(first_node) if first_node in sample.node_names else 0
            routing_meta.update(
                {
                    "source": "graph_slot_name",
                    "current_node_index": node_idx,
                    "current_node_name": first_node,
                    "current_node_role": spec.nodes[first_node].role if first_node in spec.nodes else "",
                }
            )
            self._maybe_log_fixed_slot_routing(
                slot_name=slot_name,
                process_id=process_id,
                row_idx=row_idx,
                routing_meta=routing_meta,
            )
            return float(sum(node_map.values())), 1.0, node_idx, routing_meta

        routing_meta.update(
            {
                "source": "missing",
                "current_node_index": 0,
                "current_node_name": sample.node_names[0] if sample.node_names else "",
                "current_node_role": (
                    spec.nodes[sample.node_names[0]].role
                    if sample.node_names and sample.node_names[0] in spec.nodes
                    else ""
                ),
            }
        )
        self._maybe_log_fixed_slot_routing(
            slot_name=slot_name,
            process_id=process_id,
            row_idx=row_idx,
            routing_meta=routing_meta,
        )
        return 0.0, 0.0, 0, routing_meta

    def _build_fixed_slot_routing_meta(
        self,
        *,
        slot_name: str,
        preferred_variable: str | None,
        sample: GraphSample,
        spec: ProcessSpec,
    ) -> Dict[str, Any]:
        fixed = getattr(spec, "fixed_readout", None) or {}
        if slot_name in fixed:
            node_name = fixed[slot_name]
            if node_name not in sample.node_names:
                raise ValueError(
                    f"process_id={spec.process_id!r} fixed_readout[{slot_name!r}]={node_name!r} "
                    f"not found in graph node_names for this sample."
                )
            idx = int(sample.node_names.index(node_name))
            node = spec.nodes.get(node_name)
            node_role = node.role if node is not None else ""
            cand = [
                {
                    "node_index": idx,
                    "node_name": node_name,
                    "node_role": node_role,
                    "score": 10**9,
                    "semantic_signal": True,
                    "reasons": ["excel:BFP_READOUT"],
                }
            ]
            sink_count = sum(
                1 for nm in sample.node_names if spec.nodes.get(nm) and spec.nodes[nm].role == "sink"
            )
            out_prefix_count = sum(1 for nm in sample.node_names if str(nm).upper().startswith("OUT_"))
            return {
                "preferred_variable": preferred_variable or "",
                "top_candidates": cand,
                "best_node_index": idx,
                "best_node_name": node_name,
                "best_node_role": node_role,
                "best_score": cand[0]["score"],
                "best_semantic_signal": True,
                "best_reasons": ["excel:BFP_READOUT"],
                "sink_count": sink_count,
                "semantic_candidate_count": 1,
                "out_prefix_count": out_prefix_count,
                "forced_resolution": "excel_spec",
            }

        primary_tokens = set(_tokenize_identifier(preferred_variable or slot_name))
        slot_aliases = {
            "target": {"PROD", "PRODUCT", "H2", "HYDROGEN", "OUT", "EXIT", "R2OUT", "PRODUCTGAS"},
            "tailgas": {
                "PROD",
                "PRODUCT",
                "CO2",
                "TAIL",
                "TAILGAS",
                "EXHAUST",
                "VENT",
                "OUT",
                "EX",
                "OFFGAS",
            },
        }.get(slot_name, set())
        negative_tokens = {
            "target": {"CO2", "TAIL", "TAILGAS", "EXHAUST", "VENT", "EX", "OFFGAS"},
            "tailgas": {"H2", "HYDROGEN"},
        }.get(slot_name, set())
        candidates: List[Dict[str, Any]] = []
        for idx, node_name in enumerate(sample.node_names):
            node = spec.nodes.get(node_name)
            node_role = node.role if node is not None else ""
            name_tokens = set(_tokenize_identifier(node_name))
            primary_match = sorted(primary_tokens & name_tokens)
            alias_match = sorted(slot_aliases & name_tokens)
            negative_match = sorted(negative_tokens & name_tokens)
            has_semantic_signal = bool(primary_match or alias_match)
            score = 0
            reasons: List[str] = []
            if node_role == "sink":
                score += 100
                reasons.append("sink")
            if node_name.upper().startswith("OUT_"):
                score += 25
                reasons.append("out_prefix")
            if primary_match:
                score += 30 * len(primary_match)
                reasons.append(f"primary:{'|'.join(primary_match)}")
            if alias_match:
                score += 12 * len(alias_match)
                reasons.append(f"alias:{'|'.join(alias_match)}")
            if negative_match:
                score -= 20 * len(negative_match)
                reasons.append(f"negative:{'|'.join(negative_match)}")
            candidates.append(
                {
                    "node_index": idx,
                    "node_name": node_name,
                    "node_role": node_role,
                    "score": score,
                    "semantic_signal": has_semantic_signal,
                    "reasons": reasons,
                }
            )
        candidates.sort(
            key=lambda item: (
                item["score"],
                1 if item["node_role"] == "sink" else 0,
                item["node_name"],
            ),
            reverse=True,
        )
        best = candidates[0] if candidates else None
        sink_candidates = [item for item in candidates if item["node_role"] == "sink"]
        semantic_candidates = [item for item in candidates if item["semantic_signal"]]
        out_prefix_candidates = [item for item in candidates if item["node_name"].upper().startswith("OUT_")]
        return {
            "preferred_variable": preferred_variable or "",
            "top_candidates": candidates[:5],
            "best_node_index": best["node_index"] if best else 0,
            "best_node_name": best["node_name"] if best else "",
            "best_node_role": best["node_role"] if best else "",
            "best_score": best["score"] if best else 0,
            "best_semantic_signal": bool(best["semantic_signal"]) if best else False,
            "best_reasons": list(best["reasons"]) if best else [],
            "sink_count": len(sink_candidates),
            "semantic_candidate_count": len(semantic_candidates),
            "out_prefix_count": len(out_prefix_candidates),
        }

    def _resolve_fixed_slot_node(
        self,
        *,
        sample: GraphSample,
        spec: ProcessSpec,
        routing_meta: Mapping[str, Any],
    ) -> Tuple[int, str, str, str]:
        if routing_meta.get("forced_resolution") == "excel_spec":
            return (
                int(routing_meta["best_node_index"]),
                str(routing_meta["best_node_name"]),
                str(routing_meta["best_node_role"]),
                "excel_spec",
            )

        fallback_idx = 0
        fallback_name = sample.node_names[0] if sample.node_names else ""
        fallback_role = spec.nodes[fallback_name].role if fallback_name in spec.nodes else ""

        best_idx = int(routing_meta.get("best_node_index", fallback_idx))
        best_name = str(routing_meta.get("best_node_name", fallback_name))
        best_role = str(routing_meta.get("best_node_role", fallback_role))
        best_semantic = bool(routing_meta.get("best_semantic_signal", False))
        sink_count = int(routing_meta.get("sink_count", 0))
        out_prefix_count = int(routing_meta.get("out_prefix_count", 0))

        if best_semantic:
            return best_idx, best_name, best_role, "semantic_best"
        if sink_count == 1 and best_role == "sink":
            return best_idx, best_name, best_role, "single_sink_fallback"
        if out_prefix_count == 1 and best_name.upper().startswith("OUT_"):
            return best_idx, best_name, best_role, "single_out_prefix_fallback"
        return fallback_idx, fallback_name, fallback_role, "default_zero_fallback"

    def _maybe_log_fixed_slot_routing(
        self,
        *,
        slot_name: str,
        process_id: str,
        row_idx: int,
        routing_meta: Mapping[str, Any],
    ) -> None:
        if slot_name not in self._debug_slot_log_counts:
            return
        if self._debug_slot_log_counts[slot_name] >= 6:
            return
        self._debug_slot_log_counts[slot_name] += 1
        current_idx = int(routing_meta.get("current_node_index", 0))
        best_idx = int(routing_meta.get("best_node_index", 0))
        # region agent log
        write_training_debug_event(
            session_id="8909e4",
            run_id=f"{process_id}:{row_idx}:{slot_name}",
            hypothesis_id="H1_H2_H3",
            location="src/process_graph/data/tabular_dataset.py:_extract_slot",
            message="fixed_slot_routing_probe",
            data={
                "process_id": process_id,
                "row_index": int(row_idx),
                "slot_name": slot_name,
                "preferred_variable": routing_meta.get("preferred_variable", ""),
                "source": routing_meta.get("source", ""),
                "current_node_index": current_idx,
                "current_node_name": routing_meta.get("current_node_name", ""),
                "current_node_role": routing_meta.get("current_node_role", ""),
                "best_node_index": best_idx,
                "best_node_name": routing_meta.get("best_node_name", ""),
                "best_node_role": routing_meta.get("best_node_role", ""),
                "best_score": int(routing_meta.get("best_score", 0)),
                "best_reasons": list(routing_meta.get("best_reasons", [])),
                "best_differs_from_current": current_idx != best_idx,
                "top_candidates": list(routing_meta.get("top_candidates", [])),
            },
        )
        # endregion

    def _transform_fixed_slot_value(self, slot_name: str, value: float, mask: float) -> float:
        if mask <= 0:
            return float(value)

        cfg = self.data_cfg.fixed_tasks[slot_name]
        clip_max = self.fixed_slot_clip_max.get(slot_name)
        out = float(value)
        if clip_max is not None:
            out = min(out, float(clip_max))

        if cfg.transform == "log1p":
            if out <= -1.0:
                raise ValueError(
                    f"{slot_name} value must be > -1.0 for log1p transform, got {out}."
                )
            out = math.log1p(out)

        if self.data_cfg.normalize_targets:
            mean = self.target_mean.get(slot_name)
            std = self.target_std.get(slot_name)
            if mean is None or std is None:
                raise ValueError(
                    f"normalize_targets=True but missing stats for fixed slot={slot_name!r}."
                )
            denom = std if std != 0 else 1.0
            out = (out - mean) / denom

        return float(out)

    def _extract_category_labels(
        self, sample: GraphSample
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
        enabled = {n for n, cfg in self.data_cfg.decoder_tasks.items() if cfg.enabled}
        node_name_to_idx = {name: i for i, name in enumerate(sample.node_names)}
        cat_indices: Dict[str, List[int]] = {c: [] for c in DECODER_CATEGORIES}
        cat_values: Dict[str, List[float]] = {c: [] for c in DECODER_CATEGORIES}
        for category in DECODER_CATEGORIES:
            if category not in enabled:
                continue
            node_map = sample.targets.get(category) or {}
            for node_name, value in node_map.items():
                idx = node_name_to_idx.get(node_name)
                if idx is None:
                    continue
                try:
                    cat_values[category].append(float(value))
                    cat_indices[category].append(int(idx))
                except (TypeError, ValueError):
                    continue
        return cat_indices, cat_values

    def __getitem__(self, index: int) -> GraphSampleRecord:
        row_idx = self._rows[index]
        source_row_idx = self._source_row_indices[index] if index < len(self._source_row_indices) else row_idx
        row = self.frame.iloc[row_idx]
        process_col = self.data_cfg.process_id_column
        if process_col not in row:
            raise KeyError(f"Missing required column {process_col!r} in dataset CSV.")
        process_id = str(row[process_col])
        reserved = {
            process_col,
            self.data_cfg.split_column,
        }
        data_row = _to_float_mapping(row, reserved)
        split_val = str(row[self.data_cfg.split_column]) if self.data_cfg.split_column in row.index else ""
        if "ID" in row.index and row.get("ID") is not None and not pd.isna(row.get("ID")):
            sample_id_val = str(row["ID"])
        else:
            sample_id_val = str(int(source_row_idx))

        spec = self._load_spec(process_id)
        mapping_rows: pd.DataFrame | None = None
        v3_slot_edge_index: Dict[str, int] | None = None
        if self.data_cfg.topology_mode == "stream_edge":
            if self._use_v3_canonical:
                sample, v3_slot_edge_index, mapping_rows = self._build_v3_canonical_sample(
                    spec=spec,
                    process_id=process_id,
                    row=row,
                    data_row=data_row,
                )
            else:
                sample, mapping_rows = self._build_stream_edge_sample(
                    spec=spec,
                    process_id=process_id,
                    row=row,
                    data_row=data_row,
                )
        else:
            sample = build_graph_sample(spec, data_row, passthrough_policy=self.data_cfg.passthrough_policy)

        if not self.data_cfg.use_oper_mask:
            sample = GraphSample(
                process_id=sample.process_id,
                node_names=sample.node_names,
                edge_index=sample.edge_index,
                x_role=sample.x_role,
                x_unit=sample.x_unit,
                x_hx_role=sample.x_hx_role,
                x_oper=sample.x_oper,
                x_oper_mask=[[1] * len(sample.x_oper_mask[0]) for _ in sample.x_oper_mask],
                targets=sample.targets,
                edge_stream_role=sample.edge_stream_role,
                edge_property_stream_role=sample.edge_property_stream_role,
                edge_stream_id=sample.edge_stream_id,
                edge_oper=sample.edge_oper,
                edge_oper_mask=sample.edge_oper_mask,
                edge_stream_names=sample.edge_stream_names,
                edge_feature_mask=list(sample.edge_feature_mask),
                node_is_context=list(sample.node_is_context),
                node_pinn_mask=list(sample.node_pinn_mask),
                edge_is_context=list(sample.edge_is_context),
                edge_is_predictable=list(sample.edge_is_predictable),
                edge_is_supervised=list(sample.edge_is_supervised),
                edge_is_target=list(sample.edge_is_target),
                edge_pinn_mask=list(sample.edge_pinn_mask),
                hx_pair_current_edge_index=list(sample.hx_pair_current_edge_index),
                hx_paired_edge_index=list(sample.hx_paired_edge_index),
                hx_pair_mask=list(sample.hx_pair_mask),
                hx_pair_side=list(sample.hx_pair_side),
                edge_struct_attr=sample.edge_struct_attr,
                y_edge_true=sample.y_edge_true,
                y_edge_mask=sample.y_edge_mask,
                canonical_edge_ids=sample.canonical_edge_ids,
                answer_edge_pos=sample.answer_edge_pos,
                answer_edge_ids=sample.answer_edge_ids,
                edge_target_columns=sample.edge_target_columns,
                node_q=sample.node_q,
                node_w=sample.node_w,
                node_qw_valid_mask=sample.node_qw_valid_mask,
                node_balance_exclude_mass=sample.node_balance_exclude_mass,
                node_balance_exclude_component=sample.node_balance_exclude_component,
                node_balance_exclude_atom=sample.node_balance_exclude_atom,
                node_balance_exclude_energy=sample.node_balance_exclude_energy,
            )

        if self.data_cfg.missing_value_strategy == "nan":
            raise NotImplementedError("missing_value_strategy='nan' is not implemented yet.")

        use_v3_targets = self._use_v3_canonical and self.data_cfg.topology_mode == "stream_edge"
        pid_num = _canonical_process_number(process_id)
        v3_answers_df: pd.DataFrame | None = None
        if use_v3_targets:
            if self._v3_answers is None or v3_slot_edge_index is None:
                raise RuntimeError("v3 targets requested but v3 tables or slot_edge_index are missing.")
            v3_answers_df = self._v3_answers_for_pid(pid_num)

        target_cfg = self.data_cfg.fixed_tasks["target"]
        aux_cfg = self.data_cfg.aux_task
        target_only = getattr(self.data_cfg, "task_mode", "multitask") == "target_only"
        target_variable = self._resolve_fixed_task_variable(
            process_id=process_id,
            slot_name="target",
            fallback_variable=target_cfg.target_variable,
        )

        if use_v3_targets:
            h2_ans = v3_answers_df[v3_answers_df["task_name"] == "target_h2"]  # type: ignore[union-attr]
            if len(h2_ans) != 1:
                raise ValueError(
                    f"v3 target_answer_edges: need exactly one target_h2 row for process {pid_num}, got {len(h2_ans)}"
                )
            tc_h2 = str(h2_ans.iloc[0]["target_column"])
            raw_target_value, target_mask = _v3_supervision_scalar(row, tc_h2)
            target_idx = 0
            target_route_meta = {
                "source": "v3_target_answer_edges",
                "target_column": tc_h2,
                "best_node_index": 0,
                "best_node_name": "",
            }
            target_edge_idx = int(v3_slot_edge_index["target"])
        elif mapping_rows is not None and target_variable:
            raw = row.get(target_variable)
            raw_target_value = 0.0 if raw is None or pd.isna(raw) else float(raw)
            target_mask = 0.0 if raw is None or pd.isna(raw) else 1.0
            target_idx = 0
            target_route_meta = {"source": "row_column_edge", "best_node_index": 0, "best_node_name": ""}
            target_edge_idx = self._resolve_fixed_slot_edge_index(
                mapping_rows,
                process_id=process_id,
                preferred_variable=target_variable,
                slot_name="target",
            )
        else:
            raw_target_value, target_mask, target_idx, target_route_meta = self._extract_slot(
                slot_name="target",
                preferred_variable=target_variable,
                row=row,
                sample=sample,
                spec=spec,
                row_idx=row_idx,
                process_id=process_id,
            )
            target_edge_idx = None
            if mapping_rows is not None:
                target_edge_idx = self._resolve_fixed_slot_edge_index(
                    mapping_rows,
                    process_id=process_id,
                    preferred_variable=target_variable,
                    slot_name="target",
                )
        if not target_only:
            tailgas_cfg = self.data_cfg.fixed_tasks["tailgas"]
            tailgas_variable = self._resolve_fixed_task_variable(
                process_id=process_id,
                slot_name="tailgas",
                fallback_variable=tailgas_cfg.target_variable,
            )
            if use_v3_targets:
                co2_ans = v3_answers_df[v3_answers_df["task_name"] == "tailgas_co2"]  # type: ignore[union-attr]
                if len(co2_ans) != 1:
                    raise ValueError(
                        f"v3 target_answer_edges: need exactly one tailgas_co2 row for process {pid_num}, "
                        f"got {len(co2_ans)}"
                    )
                tc_co2 = str(co2_ans.iloc[0]["target_column"])
                raw_tailgas_value, tailgas_mask = _v3_supervision_scalar(row, tc_co2)
                tailgas_idx = 0
                tailgas_route_meta = {
                    "source": "v3_target_answer_edges",
                    "target_column": tc_co2,
                    "best_node_index": 0,
                    "best_node_name": "",
                }
                tailgas_edge_idx = int(v3_slot_edge_index["tailgas"])
            elif mapping_rows is not None and tailgas_variable:
                raw = row.get(tailgas_variable)
                raw_tailgas_value = 0.0 if raw is None or pd.isna(raw) else float(raw)
                tailgas_mask = 0.0 if raw is None or pd.isna(raw) else 1.0
                tailgas_idx = 0
                tailgas_route_meta = {"source": "row_column_edge", "best_node_index": 0, "best_node_name": ""}
                tailgas_edge_idx = self._resolve_fixed_slot_edge_index(
                    mapping_rows,
                    process_id=process_id,
                    preferred_variable=tailgas_variable,
                    slot_name="tailgas",
                )
            else:
                raw_tailgas_value, tailgas_mask, tailgas_idx, tailgas_route_meta = self._extract_slot(
                    slot_name="tailgas",
                    preferred_variable=tailgas_variable,
                    row=row,
                    sample=sample,
                    spec=spec,
                    row_idx=row_idx,
                    process_id=process_id,
                )
                tailgas_edge_idx = None
                if mapping_rows is not None:
                    tailgas_edge_idx = self._resolve_fixed_slot_edge_index(
                        mapping_rows,
                        process_id=process_id,
                        preferred_variable=tailgas_variable,
                        slot_name="tailgas",
                    )
        raw_sample = sample
        sample = self._apply_normalization(sample)
        self._log_known_feed_sample_once(
            process_id=process_id,
            sample_id=sample_id_val,
            raw_sample=raw_sample,
            scaled_sample=sample,
        )
        target_value = self._transform_fixed_slot_value("target", raw_target_value, target_mask)
        if target_only:
            sample_meta = {
                "row_index": int(row_idx),
                "dataset_split": split_val,
                "sample_id": sample_id_val,
                "process_id": process_id,
                "raw_target": float(raw_target_value),
                "target": float(target_value),
                "target_node_index": int(target_idx),
                "target_node_name": sample.node_names[target_idx] if sample.node_names else "",
                "target_best_node_index": int(target_route_meta.get("best_node_index", 0)),
                "target_best_node_name": str(target_route_meta.get("best_node_name", "")),
                "target_route_source": str(target_route_meta.get("source", "")),
                "target_route_mode": str(target_route_meta.get("resolution_mode", "")),
                "target_route_reasons": list(target_route_meta.get("best_reasons", [])),
                "target_best_differs": bool(
                    int(target_route_meta.get("best_node_index", 0)) != int(target_idx)
                ),
                "target_from_excel_spec": bool(target_route_meta.get("forced_resolution") == "excel_spec"),
                "target_variable_resolved": str(target_variable or ""),
            }
            slot_edge_index = {"target": int(target_edge_idx)} if target_edge_idx is not None else {}
            if target_edge_idx is not None and mapping_rows is not None:
                edge_row = mapping_rows.iloc[int(target_edge_idx)]
                sample_meta.update(
                    {
                        "target_edge_index": int(target_edge_idx),
                        "target_edge_stream_name": _edge_row_primary_stream_label(edge_row),
                        "target_edge_src_node": str(edge_row.get("src_node", "")),
                        "target_edge_dst_node": str(edge_row.get("dst_node", "")),
                        "target_edge_role": str(edge_row.get("stream_role", "")),
                    }
                )
        else:
            tailgas_value = self._transform_fixed_slot_value(
                "tailgas", raw_tailgas_value, tailgas_mask
            )
            sample_meta = {
                "row_index": int(row_idx),
                "dataset_split": split_val,
                "sample_id": sample_id_val,
                "process_id": process_id,
                "raw_target": float(raw_target_value),
                "raw_tailgas": float(raw_tailgas_value),
                "target": float(target_value),
                "tailgas": float(tailgas_value),
                "target_node_index": int(target_idx),
                "target_node_name": sample.node_names[target_idx] if sample.node_names else "",
                "target_best_node_index": int(target_route_meta.get("best_node_index", 0)),
                "target_best_node_name": str(target_route_meta.get("best_node_name", "")),
                "target_route_source": str(target_route_meta.get("source", "")),
                "target_route_mode": str(target_route_meta.get("resolution_mode", "")),
                "target_route_reasons": list(target_route_meta.get("best_reasons", [])),
                "target_best_differs": bool(
                    int(target_route_meta.get("best_node_index", 0)) != int(target_idx)
                ),
                "tailgas_node_index": int(tailgas_idx),
                "tailgas_node_name": sample.node_names[tailgas_idx] if sample.node_names else "",
                "tailgas_best_node_index": int(tailgas_route_meta.get("best_node_index", 0)),
                "tailgas_best_node_name": str(tailgas_route_meta.get("best_node_name", "")),
                "tailgas_route_source": str(tailgas_route_meta.get("source", "")),
                "tailgas_route_mode": str(tailgas_route_meta.get("resolution_mode", "")),
                "tailgas_route_reasons": list(tailgas_route_meta.get("best_reasons", [])),
                "tailgas_best_differs": bool(
                    int(tailgas_route_meta.get("best_node_index", 0)) != int(tailgas_idx)
                ),
                "target_from_excel_spec": bool(target_route_meta.get("forced_resolution") == "excel_spec"),
                "tailgas_from_excel_spec": bool(tailgas_route_meta.get("forced_resolution") == "excel_spec"),
                "target_variable_resolved": str(target_variable or ""),
                "tailgas_variable_resolved": str(tailgas_variable or ""),
            }
            slot_edge_index = {}
            if target_edge_idx is not None:
                slot_edge_index["target"] = int(target_edge_idx)
            if tailgas_edge_idx is not None:
                slot_edge_index["tailgas"] = int(tailgas_edge_idx)
            if mapping_rows is not None:
                if target_edge_idx is not None:
                    edge_row = mapping_rows.iloc[int(target_edge_idx)]
                    sample_meta.update(
                        {
                            "target_edge_index": int(target_edge_idx),
                            "target_edge_stream_name": _edge_row_primary_stream_label(edge_row),
                            "target_edge_src_node": str(edge_row.get("src_node", "")),
                            "target_edge_dst_node": str(edge_row.get("dst_node", "")),
                            "target_edge_role": str(edge_row.get("stream_role", "")),
                        }
                    )
                if tailgas_edge_idx is not None:
                    edge_row = mapping_rows.iloc[int(tailgas_edge_idx)]
                    sample_meta.update(
                        {
                            "tailgas_edge_index": int(tailgas_edge_idx),
                            "tailgas_edge_stream_name": _edge_row_primary_stream_label(edge_row),
                            "tailgas_edge_src_node": str(edge_row.get("src_node", "")),
                            "tailgas_edge_dst_node": str(edge_row.get("dst_node", "")),
                            "tailgas_edge_role": str(edge_row.get("stream_role", "")),
                        }
                    )

        decoder_on = any(self.data_cfg.decoder_tasks[c].enabled for c in DECODER_CATEGORIES)
        if decoder_on:
            cat_idx, cat_val = self._extract_category_labels(sample)
            if target_only:
                return GraphSampleRecord(
                    graph=sample,
                    slot_targets={"target": target_value},
                    slot_masks={"target": target_mask},
                    slot_node_index={"target": target_idx},
                    category_node_indices=cat_idx,
                    category_values=cat_val,
                    sample_meta=sample_meta,
                    slot_edge_index=slot_edge_index,
                )
            return GraphSampleRecord(
                graph=sample,
                slot_targets={"target": target_value, "tailgas": tailgas_value},
                slot_masks={"target": target_mask, "tailgas": tailgas_mask},
                slot_node_index={"target": target_idx, "tailgas": tailgas_idx},
                category_node_indices=cat_idx,
                category_values=cat_val,
                sample_meta=sample_meta,
                slot_edge_index=slot_edge_index,
            )

        if not aux_cfg.enabled:
            empty_cat_idx = {c: [] for c in DECODER_CATEGORIES}
            empty_cat_val = {c: [] for c in DECODER_CATEGORIES}
            if target_only:
                return GraphSampleRecord(
                    graph=sample,
                    slot_targets={"target": target_value},
                    slot_masks={"target": target_mask},
                    slot_node_index={"target": target_idx},
                    category_node_indices=empty_cat_idx,
                    category_values=empty_cat_val,
                    sample_meta=sample_meta,
                    slot_edge_index=slot_edge_index,
                )
            return GraphSampleRecord(
                graph=sample,
                slot_targets={"target": target_value, "tailgas": tailgas_value},
                slot_masks={"target": target_mask, "tailgas": tailgas_mask},
                slot_node_index={"target": target_idx, "tailgas": tailgas_idx},
                category_node_indices=empty_cat_idx,
                category_values=empty_cat_val,
                sample_meta=sample_meta,
                slot_edge_index=slot_edge_index,
            )

        if target_only:
            raise ValueError(
                "aux_task.enabled=true is incompatible with data.task_mode='target_only' "
                "(loaders should reject this combination)."
            )

        aux_preferred = aux_cfg.target_variable or aux_cfg.task_name
        aux_value, aux_mask, aux_idx, _ = self._extract_slot(
            slot_name="aux",
            preferred_variable=aux_preferred,
            row=row,
            sample=sample,
            spec=spec,
            row_idx=row_idx,
            process_id=process_id,
        )
        empty_cat_idx = {c: [] for c in DECODER_CATEGORIES}
        empty_cat_val = {c: [] for c in DECODER_CATEGORIES}
        return GraphSampleRecord(
            graph=sample,
            slot_targets={
                "target": target_value,
                "tailgas": tailgas_value,
                "aux": aux_value,
            },
            slot_masks={
                "target": target_mask,
                "tailgas": tailgas_mask,
                "aux": aux_mask,
            },
            slot_node_index={
                "target": target_idx,
                "tailgas": tailgas_idx,
                "aux": aux_idx,
            },
            category_node_indices=empty_cat_idx,
            category_values=empty_cat_val,
            sample_meta=sample_meta,
            slot_edge_index=slot_edge_index,
        )


def _stack_oper(records: Sequence[GraphSampleRecord]) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [torch.tensor(record.graph.x_oper, dtype=torch.float32) for record in records]
    masks = [torch.tensor(record.graph.x_oper_mask, dtype=torch.float32) for record in records]
    return torch.cat(tensors, dim=0), torch.cat(masks, dim=0)


def _stack_long(records: Sequence[GraphSampleRecord], field: str) -> torch.Tensor:
    tensors = [torch.tensor(getattr(record.graph, field), dtype=torch.long) for record in records]
    return torch.cat(tensors, dim=0)


def _has_edge_features(records: Sequence[GraphSampleRecord]) -> bool:
    return bool(records and records[0].graph.edge_oper)


def _stack_edge_oper(records: Sequence[GraphSampleRecord]) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [torch.tensor(record.graph.edge_oper, dtype=torch.float32) for record in records]
    masks = [torch.tensor(record.graph.edge_oper_mask, dtype=torch.float32) for record in records]
    return torch.cat(tensors, dim=0), torch.cat(masks, dim=0)


def _cat_edge_feature_mask(records: Sequence[GraphSampleRecord]) -> torch.Tensor | None:
    if not records or not records[0].graph.edge_feature_mask:
        return None
    parts = [torch.tensor(r.graph.edge_feature_mask, dtype=torch.float32) for r in records]
    return torch.cat(parts, dim=0)


def _cat_y_edge(records: Sequence[GraphSampleRecord]) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if not records or not records[0].graph.y_edge_true:
        return None, None
    y_true = torch.cat([torch.tensor(r.graph.y_edge_true, dtype=torch.float32) for r in records], dim=0)
    y_mask = torch.cat([torch.tensor(r.graph.y_edge_mask, dtype=torch.float32) for r in records], dim=0)
    return y_true, y_mask


def collate_graph_batch(
    records: List[GraphSampleRecord],
    *,
    fixed_slots: Sequence[str] | None = None,
) -> GraphBatch:
    if not records:
        raise ValueError("collate_graph_batch received an empty batch.")

    slots: Tuple[str, ...] = tuple(fixed_slots) if fixed_slots is not None else FIXED_SLOTS
    if not slots:
        raise ValueError("collate_graph_batch: fixed_slots must be non-empty.")
    for slot in slots:
        if slot not in FIXED_SLOTS:
            raise ValueError(f"collate_graph_batch: unknown fixed slot {slot!r}. Expected one of {FIXED_SLOTS}.")

    use_aux = "aux" in records[0].slot_targets

    edge_chunks: List[torch.Tensor] = []
    batch_vec_chunks: List[torch.Tensor] = []
    edge_batch_chunks: List[torch.Tensor] = []
    hx_pair_current_chunks: List[torch.Tensor] = []
    hx_paired_chunks: List[torch.Tensor] = []
    hx_pair_mask_chunks: List[torch.Tensor] = []
    hx_pair_side_chunks: List[torch.Tensor] = []
    node_offset = 0
    edge_offset = 0

    fixed_index: Dict[str, List[int]] = {slot: [] for slot in slots}
    fixed_edge_index: Dict[str, List[int]] = {slot: [] for slot in slots}
    aux_index: List[int] = []
    category_flat_index: Dict[str, List[int]] = {c: [] for c in DECODER_CATEGORIES}
    category_flat_values: Dict[str, List[float]] = {c: [] for c in DECODER_CATEGORIES}
    answer_edge_pos_batch: List[Dict[str, int]] = []

    export_proc: List[str] = []
    export_sample_id: List[str] = []
    export_split: List[str] = []
    export_canonical_edge_id: List[str] = []
    export_stream_key: List[str] = []
    export_stream_role: List[str] = []
    export_is_input: List[float] = []
    export_is_output: List[float] = []
    export_is_internal: List[float] = []
    export_answer_tasks: List[str] = []
    export_src_node: List[str] = []
    export_dst_node: List[str] = []
    export_is_context: List[float] = []
    export_is_predictable: List[float] = []
    export_is_supervised: List[float] = []
    export_is_target: List[float] = []
    export_include_in_pinn: List[float] = []

    for graph_idx, record in enumerate(records):
        sample = record.graph
        edge_index = torch.tensor(sample.edge_index, dtype=torch.long)
        edge_chunks.append(edge_index + node_offset)
        num_nodes = len(sample.node_names)
        num_edges = edge_index.size(1)
        batch_vec_chunks.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        edge_batch_chunks.append(torch.full((num_edges,), graph_idx, dtype=torch.long))
        if sample.hx_paired_edge_index:
            if not (
                len(sample.hx_pair_current_edge_index)
                == len(sample.hx_paired_edge_index)
                == len(sample.hx_pair_mask)
                == len(sample.hx_pair_side)
            ):
                raise ValueError(
                    "HX relation metadata arrays must have equal lengths."
                )
            local_current = torch.tensor(
                sample.hx_pair_current_edge_index, dtype=torch.long
            )
            local_pair = torch.tensor(
                sample.hx_paired_edge_index, dtype=torch.long
            )
            invalid = (
                (local_current < 0)
                | (local_current >= num_edges)
                | (local_pair < 0)
                | (local_pair >= num_edges)
            )
            if bool(invalid.any()):
                raise ValueError(
                    f"Graph {graph_idx} contains an invalid local HX pair index."
                )
            hx_pair_current_chunks.append(local_current + int(edge_offset))
            hx_paired_chunks.append(local_pair + int(edge_offset))
            hx_pair_mask_chunks.append(
                torch.tensor(sample.hx_pair_mask, dtype=torch.float32)
            )
            hx_pair_side_chunks.append(
                torch.tensor(sample.hx_pair_side, dtype=torch.long)
            )
        else:
            hx_pair_current_chunks.append(torch.empty(0, dtype=torch.long))
            hx_paired_chunks.append(torch.empty(0, dtype=torch.long))
            hx_pair_mask_chunks.append(torch.empty(0, dtype=torch.float32))
            hx_pair_side_chunks.append(torch.empty(0, dtype=torch.long))

        ge_pos: Dict[str, int] = {}
        if record.slot_edge_index:
            for slot_name, local_edge_idx in record.slot_edge_index.items():
                li = max(0, min(int(local_edge_idx), num_edges - 1))
                ge_pos[str(slot_name)] = int(edge_offset + li)
        answer_edge_pos_batch.append(ge_pos)

        for slot in slots:
            if record.slot_edge_index and slot in record.slot_edge_index:
                local_edge_idx = int(record.slot_edge_index.get(slot, 0))
                local_edge_idx = max(0, min(local_edge_idx, num_edges - 1))
                fixed_edge_index[slot].append(edge_offset + local_edge_idx)
            else:
                local_idx = int(record.slot_node_index.get(slot, 0))
                local_idx = max(0, min(local_idx, num_nodes - 1))
                fixed_index[slot].append(node_offset + local_idx)

        if use_aux:
            local_idx = int(record.slot_node_index.get("aux", 0))
            local_idx = max(0, min(local_idx, num_nodes - 1))
            aux_index.append(node_offset + local_idx)

        for category in DECODER_CATEGORIES:
            local_indices = record.category_node_indices.get(category, [])
            values = record.category_values.get(category, [])
            for local_idx, value in zip(local_indices, values):
                local_idx = max(0, min(int(local_idx), num_nodes - 1))
                category_flat_index[category].append(node_offset + local_idx)
                category_flat_values[category].append(float(value))

        if _has_edge_features([record]):
            g = record.graph
            meta = record.sample_meta
            ne = int(len(g.edge_stream_role))
            for ei in range(ne):
                export_proc.append(str(g.process_id))
                export_sample_id.append(str(meta.get("sample_id", "")))
                export_split.append(str(meta.get("dataset_split", "")))
                ceid = str(g.canonical_edge_ids[ei]) if ei < len(g.canonical_edge_ids) else ""
                export_canonical_edge_id.append(ceid)
                src_i = int(g.edge_index[0][ei]) if ei < len(g.edge_index[0]) else -1
                dst_i = int(g.edge_index[1][ei]) if ei < len(g.edge_index[1]) else -1
                export_src_node.append(str(g.node_names[src_i]) if 0 <= src_i < len(g.node_names) else "")
                export_dst_node.append(str(g.node_names[dst_i]) if 0 <= dst_i < len(g.node_names) else "")
                sk = str(g.edge_stream_names[ei]) if ei < len(g.edge_stream_names) else ""
                export_stream_key.append(sk)
                ri = int(g.edge_stream_role[ei]) if ei < len(g.edge_stream_role) else 0
                export_stream_role.append(
                    STREAM_ROLE_VOCAB[ri] if 0 <= ri < len(STREAM_ROLE_VOCAB) else "unknown"
                )
                st = g.edge_struct_attr[ei] if ei < len(g.edge_struct_attr) else [0.0, 0.0, 0.0, 0.0, 0.0]
                export_is_input.append(float(st[0]))
                export_is_output.append(float(st[1]))
                export_is_internal.append(float(st[2]))
                tns: list[str] = []
                for slot, eid in (g.answer_edge_ids or {}).items():
                    if str(eid) == ceid:
                        tns.append(str(V3_SLOT_TO_TASK_NAME.get(slot, slot)))
                export_answer_tasks.append(";".join(sorted(tns)))
                export_is_context.append(
                    float(g.edge_is_context[ei])
                    if ei < len(g.edge_is_context)
                    else 0.0
                )
                export_is_predictable.append(
                    float(g.edge_is_predictable[ei])
                    if ei < len(g.edge_is_predictable)
                    else 1.0
                )
                export_is_supervised.append(
                    float(g.edge_is_supervised[ei])
                    if ei < len(g.edge_is_supervised)
                    else (
                        float(g.y_edge_mask[ei])
                        if ei < len(g.y_edge_mask)
                        else 0.0
                    )
                )
                export_is_target.append(
                    float(g.edge_is_target[ei])
                    if ei < len(g.edge_is_target)
                    else 0.0
                )
                export_include_in_pinn.append(
                    float(g.edge_pinn_mask[ei])
                    if ei < len(g.edge_pinn_mask)
                    else 1.0
                )

        node_offset += num_nodes
        edge_offset += num_edges

    edge_index = torch.cat(edge_chunks, dim=1)
    batch = torch.cat(batch_vec_chunks, dim=0)
    edge_batch = torch.cat(edge_batch_chunks, dim=0)
    x_oper, x_oper_mask = _stack_oper(records)
    x_role = _stack_long(records, "x_role")
    x_unit = _stack_long(records, "x_unit")
    x_hx_role = _stack_long(records, "x_hx_role")
    def _cat_graph_node_float(name: str) -> torch.Tensor | None:
        if not records or not getattr(records[0].graph, name, None):
            return None
        parts = [torch.tensor(getattr(record.graph, name), dtype=torch.float32) for record in records]
        return torch.cat(parts, dim=0)

    def _cat_graph_edge_float(name: str) -> torch.Tensor | None:
        if not records or not getattr(records[0].graph, name, None):
            return None
        parts = [
            torch.tensor(getattr(record.graph, name), dtype=torch.float32)
            for record in records
        ]
        return torch.cat(parts, dim=0)

    node_q = _cat_graph_node_float("node_q")
    node_w = _cat_graph_node_float("node_w")
    node_qw_valid_mask = _cat_graph_node_float("node_qw_valid_mask")
    node_exclude_mass = _cat_graph_node_float("node_balance_exclude_mass")
    node_exclude_component = _cat_graph_node_float("node_balance_exclude_component")
    node_exclude_atom = _cat_graph_node_float("node_balance_exclude_atom")
    node_exclude_energy = _cat_graph_node_float("node_balance_exclude_energy")
    node_is_context = _cat_graph_node_float("node_is_context")
    node_pinn_mask = _cat_graph_node_float("node_pinn_mask")
    edge_is_context = _cat_graph_edge_float("edge_is_context")
    edge_is_predictable = _cat_graph_edge_float("edge_is_predictable")
    edge_is_supervised = _cat_graph_edge_float("edge_is_supervised")
    edge_is_target = _cat_graph_edge_float("edge_is_target")
    edge_pinn_mask = _cat_graph_edge_float("edge_pinn_mask")
    graph_feed_values: torch.Tensor | None = None
    graph_feed_mask: torch.Tensor | None = None
    if records[0].graph.graph_feed_values:
        for graph_idx, record in enumerate(records):
            if len(record.graph.graph_feed_values) != len(KNOWN_FEED_NAMES):
                raise ValueError(
                    f"Graph {graph_idx} feed values must use CH4, AIR, WATER order."
                )
            if len(record.graph.graph_feed_mask) != len(KNOWN_FEED_NAMES):
                raise ValueError(
                    f"Graph {graph_idx} feed mask must use CH4, AIR, WATER order."
                )
        graph_feed_values = torch.tensor(
            [record.graph.graph_feed_values for record in records],
            dtype=torch.float32,
        )
        graph_feed_mask = torch.tensor(
            [record.graph.graph_feed_mask for record in records],
            dtype=torch.float32,
        )

    model_kwargs = {
        "edge_index": edge_index,
        "x_role": x_role,
        "x_unit": x_unit,
        "x_hx_role": x_hx_role,
        "x_oper": x_oper,
        "x_oper_mask": x_oper_mask,
        "batch": batch,
        "edge_batch": edge_batch,
    }
    if node_q is not None:
        model_kwargs["node_q"] = node_q
    if node_w is not None:
        model_kwargs["node_w"] = node_w
    if node_qw_valid_mask is not None:
        model_kwargs["node_qw_valid_mask"] = node_qw_valid_mask
    if node_exclude_mass is not None:
        model_kwargs["node_balance_exclude_mass"] = node_exclude_mass
    if node_exclude_component is not None:
        model_kwargs["node_balance_exclude_component"] = node_exclude_component
    if node_exclude_atom is not None:
        model_kwargs["node_balance_exclude_atom"] = node_exclude_atom
    if node_exclude_energy is not None:
        model_kwargs["node_balance_exclude_energy"] = node_exclude_energy
    if node_is_context is not None:
        model_kwargs["node_is_context"] = node_is_context
    if node_pinn_mask is not None:
        model_kwargs["node_pinn_mask"] = node_pinn_mask
    if edge_is_context is not None:
        model_kwargs["edge_is_context"] = edge_is_context
    if edge_is_predictable is not None:
        model_kwargs["edge_is_predictable"] = edge_is_predictable
    if edge_is_supervised is not None:
        model_kwargs["edge_is_supervised"] = edge_is_supervised
    if edge_is_target is not None:
        model_kwargs["edge_is_target"] = edge_is_target
    if edge_pinn_mask is not None:
        model_kwargs["edge_pinn_mask"] = edge_pinn_mask
    if graph_feed_values is not None and graph_feed_mask is not None:
        model_kwargs["graph_feed_values"] = graph_feed_values
        model_kwargs["graph_feed_mask"] = graph_feed_mask
        model_kwargs["graph_feed_input"] = torch.cat(
            [graph_feed_values, graph_feed_mask], dim=-1
        )
    if hx_paired_chunks:
        if len(hx_paired_chunks) != len(records):
            raise ValueError(
                "HX pair metadata must be present for every graph in a batch."
            )
        hx_pair_current_edge_index = torch.cat(hx_pair_current_chunks, dim=0)
        hx_paired_edge_index = torch.cat(hx_paired_chunks, dim=0)
        hx_pair_mask = torch.cat(hx_pair_mask_chunks, dim=0)
        hx_pair_side = torch.cat(hx_pair_side_chunks, dim=0)
        valid_hx = hx_pair_mask > 0.5
        if bool(valid_hx.any()):
            pair_rows = hx_paired_edge_index[valid_hx]
            current_rows = hx_pair_current_edge_index[valid_hx]
            if bool((pair_rows < 0).any()):
                raise ValueError("Valid HX pairs cannot contain -1 indices.")
            if not torch.equal(
                edge_batch[pair_rows], edge_batch[current_rows]
            ):
                raise ValueError("HX pair index crosses graph boundaries after collation.")
            directed = set(zip(current_rows.tolist(), pair_rows.tolist()))
            if any((partner, edge) not in directed for edge, partner in directed):
                raise ValueError("HX pair index is not symmetric after collation.")
        model_kwargs["hx_pair_current_edge_index"] = hx_pair_current_edge_index
        model_kwargs["hx_paired_edge_index"] = hx_paired_edge_index
        model_kwargs["hx_pair_mask"] = hx_pair_mask
        model_kwargs["hx_pair_side"] = hx_pair_side
    targets: Dict[str, torch.Tensor] = {}
    target_masks: Dict[str, torch.Tensor] = {}
    task_inputs: Dict[str, Dict[str, torch.Tensor]] = {}

    if _has_edge_features(records):
        edge_oper, edge_oper_mask = _stack_edge_oper(records)
        model_kwargs.update(
            {
                "edge_stream_role": _stack_long(records, "edge_stream_role"),
                "edge_property_stream_role": _stack_long(
                    records, "edge_property_stream_role"
                ),
                "edge_stream_id": _stack_long(records, "edge_stream_id"),
                "edge_oper": edge_oper,
                "edge_oper_mask": edge_oper_mask,
                "edge_struct_attr": edge_oper,
            }
        )
        efm = _cat_edge_feature_mask(records)
        if efm is not None:
            model_kwargs["edge_feature_mask"] = efm
        y_edge_true, y_edge_mask = _cat_y_edge(records)
        if y_edge_true is not None and y_edge_mask is not None:
            model_kwargs["y_edge_true"] = y_edge_true
            model_kwargs["y_edge_mask"] = y_edge_mask
            targets["edge_stream"] = y_edge_true
            target_masks["edge_stream"] = y_edge_mask
            task_inputs["edge_stream"] = {"edge_batch": edge_batch}

    edge_export_meta: EdgeBatchExportMeta | None = None
    if _has_edge_features(records) and export_proc:
        edge_export_meta = EdgeBatchExportMeta(
            process_id=export_proc,
            sample_id=export_sample_id,
            dataset_split=export_split,
            canonical_edge_id=export_canonical_edge_id,
            main_data_stream_key=export_stream_key,
            stream_role=export_stream_role,
            is_input_edge=export_is_input,
            is_output_edge=export_is_output,
            is_internal_edge=export_is_internal,
            answer_task_names=export_answer_tasks,
            src_node=export_src_node,
            dst_node=export_dst_node,
            is_context=export_is_context,
            is_predictable=export_is_predictable,
            is_supervised=export_is_supervised,
            is_target=export_is_target,
            include_in_pinn=export_include_in_pinn,
        )

    for slot in slots:
        values = [float(record.slot_targets.get(slot, 0.0)) for record in records]
        masks = [float(record.slot_masks.get(slot, 0.0)) for record in records]
        targets[slot] = torch.tensor(values, dtype=torch.float32).view(-1, 1)
        target_masks[slot] = torch.tensor(masks, dtype=torch.float32).view(-1, 1)
        if fixed_edge_index[slot]:
            task_inputs[slot] = {
                "edge_readout_index": torch.tensor(fixed_edge_index[slot], dtype=torch.long)
            }
        else:
            task_inputs[slot] = {"node_index": torch.tensor(fixed_index[slot], dtype=torch.long)}

    if use_aux:
        values = [float(record.slot_targets.get("aux", 0.0)) for record in records]
        masks = [float(record.slot_masks.get("aux", 0.0)) for record in records]
        targets["aux"] = torch.tensor(values, dtype=torch.float32).view(-1, 1)
        target_masks["aux"] = torch.tensor(masks, dtype=torch.float32).view(-1, 1)
        task_inputs["aux"] = {"node_index": torch.tensor(aux_index, dtype=torch.long)}
    else:
        for category in DECODER_CATEGORIES:
            idx_list = category_flat_index[category]
            val_list = category_flat_values[category]
            targets[category] = torch.tensor(val_list, dtype=torch.float32).view(-1, 1)
            target_masks[category] = torch.ones(len(val_list), dtype=torch.float32).view(-1, 1)
            task_inputs[category] = {"node_index": torch.tensor(idx_list, dtype=torch.long)}

    return GraphBatch(
        model_kwargs=model_kwargs,
        targets=targets,
        target_masks=target_masks,
        task_inputs=task_inputs,
        sample_meta=[dict(record.sample_meta) for record in records],
        canonical_edge_ids=[list(record.graph.canonical_edge_ids) for record in records],
        answer_edge_pos=answer_edge_pos_batch,
        answer_edge_ids=[dict(record.graph.answer_edge_ids) for record in records],
        edge_target_columns=list(records[0].graph.edge_target_columns) if records[0].graph.edge_target_columns else None,
        edge_export_meta=edge_export_meta,
    )


def compute_oper_normalizer(dataset: Iterable[GraphSampleRecord]) -> Tuple[torch.Tensor, torch.Tensor]:
    sums: torch.Tensor | None = None
    sumsq: torch.Tensor | None = None
    count = 0
    for record in dataset:
        x = torch.tensor(record.graph.x_oper, dtype=torch.float64)
        if x.numel() == 0:
            continue
        if sums is None:
            sums = torch.zeros(x.shape[1], dtype=torch.float64)
            sumsq = torch.zeros(x.shape[1], dtype=torch.float64)
        sums += x.sum(dim=0)
        assert sumsq is not None
        sumsq += (x * x).sum(dim=0)
        count += int(x.shape[0])
    if sums is None or sumsq is None or count <= 0:
        raise ValueError("Cannot compute oper normalizer on an empty dataset.")
    mean64 = sums / float(count)
    var64 = (sumsq / float(count)) - (mean64 * mean64)
    std64 = torch.sqrt(torch.clamp(var64, min=0.0))
    return mean64.float(), std64.float()


def compute_oper_normalizer_from_x_oper(
    x_oper_iter: Iterable[Sequence[Sequence[float]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sums: torch.Tensor | None = None
    sumsq: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    for x_oper in x_oper_iter:
        x = torch.tensor(x_oper, dtype=torch.float64)
        if x.numel() == 0:
            continue
        if sums is None:
            sums = torch.zeros(x.shape[1], dtype=torch.float64)
            sumsq = torch.zeros(x.shape[1], dtype=torch.float64)
            counts = torch.zeros(x.shape[1], dtype=torch.float64)
        finite = torch.isfinite(x)
        clean = torch.where(finite, x, torch.zeros_like(x))
        sums += clean.sum(dim=0)
        assert sumsq is not None and counts is not None
        sumsq += (clean * clean).sum(dim=0)
        counts += finite.sum(dim=0)
    if sums is None or sumsq is None or counts is None:
        raise ValueError("Cannot compute oper normalizer on an empty dataset.")
    safe_counts = torch.clamp(counts, min=1.0)
    mean64 = sums / safe_counts
    var64 = (sumsq / safe_counts) - (mean64 * mean64)
    std64 = torch.sqrt(torch.clamp(var64, min=0.0))
    mean64 = torch.where(counts > 0, mean64, torch.zeros_like(mean64))
    std64 = torch.where(counts > 0, std64, torch.ones_like(std64))
    return mean64.float(), std64.float()


def compute_target_normalizer(
    dataset: Iterable[GraphSampleRecord], slots: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    sums: Dict[str, float] = {slot: 0.0 for slot in slots}
    counts: Dict[str, int] = {slot: 0 for slot in slots}
    for record in dataset:
        for slot in slots:
            mask = float(record.slot_masks.get(slot, 0.0))
            if mask <= 0:
                continue
            sums[slot] += float(record.slot_targets.get(slot, 0.0))
            counts[slot] += 1

    mean = {slot: (sums[slot] / counts[slot]) if counts[slot] else 0.0 for slot in slots}

    var_accum: Dict[str, float] = {slot: 0.0 for slot in slots}
    for record in dataset:
        for slot in slots:
            mask = float(record.slot_masks.get(slot, 0.0))
            if mask <= 0:
                continue
            value = float(record.slot_targets.get(slot, 0.0))
            var_accum[slot] += (value - mean[slot]) ** 2

    std = {slot: (var_accum[slot] / counts[slot]) ** 0.5 if counts[slot] else 1.0 for slot in slots}
    return mean, std


def compute_category_target_normalizer(
    dataset: Iterable[GraphSampleRecord], categories: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Mean/std over all supervised (node, value) pairs per decoder category."""

    sums: Dict[str, float] = {c: 0.0 for c in categories}
    sumsq: Dict[str, float] = {c: 0.0 for c in categories}
    counts: Dict[str, int] = {c: 0 for c in categories}
    for record in dataset:
        for category in categories:
            for value in record.category_values.get(category, []) or []:
                fv = float(value)
                sums[category] += fv
                sumsq[category] += fv * fv
                counts[category] += 1

    mean = {c: (sums[c] / counts[c]) if counts[c] else 0.0 for c in categories}
    std = {
        c: max((sumsq[c] / counts[c]) - (mean[c] * mean[c]), 0.0) ** 0.5 if counts[c] else 1.0
        for c in categories
    }
    return mean, std


def compute_y_edge_scaler(
    dataset: Iterable[GraphSampleRecord],
    *,
    stream_target_dim: int | None = None,
    std_eps: float = 1e-6,
) -> YEdgeScalerFitResult:
    """Per-column mean/std over train rows where y_edge_mask==1 (StandardScaler)."""
    first_columns: Sequence[str] | None = None
    dim = int(stream_target_dim) if stream_target_dim is not None else 0
    if dim <= 0:
        dim = len(STREAM_EDGE_FEATURE_SLOTS)
    sums = torch.zeros(dim, dtype=torch.float64)
    sumsq = torch.zeros(dim, dtype=torch.float64)
    counts_t = torch.zeros(dim, dtype=torch.long)
    warn_msgs: list[str] = []
    seen_records = 0

    for record in dataset:
        seen_records += 1
        if first_columns is None:
            first_columns = record.graph.edge_target_columns or list(STREAM_EDGE_FEATURE_SLOTS)
        yt = record.graph.y_edge_true
        ym = record.graph.y_edge_mask
        if not yt:
            continue
        n_edge = len(yt)
        if len(ym) != n_edge:
            raise ValueError(
                f"compute_y_edge_scaler: y_edge_mask length {len(ym)} != y_edge_true rows {n_edge}."
            )
        for ei, row in enumerate(yt):
            if float(ym[ei]) <= 0.0:
                continue
            if len(row) != dim:
                raise ValueError(
                    f"compute_y_edge_scaler: expected row width {dim}, got {len(row)} "
                    f"(process_id={record.graph.process_id!r})."
                )
            for j in range(dim):
                fv = float(row[j])
                sums[j] += fv
                sumsq[j] += fv * fv
                counts_t[j] += 1

    if seen_records <= 0:
        raise ValueError("compute_y_edge_scaler: empty dataset.")
    cols = list(first_columns) if first_columns else list(STREAM_EDGE_FEATURE_SLOTS)
    if len(cols) < dim:
        cols = cols + [f"y_edge_{i}" for i in range(len(cols), dim)]

    mean = torch.zeros(dim, dtype=torch.float32)
    std = torch.ones(dim, dtype=torch.float32)
    counts: list[int] = []
    for j in range(dim):
        count = int(counts_t[j].item())
        counts.append(count)
        if count <= 0:
            mean[j] = 0.0
            std[j] = 1.0
            msg = (
                f"compute_y_edge_scaler: column {cols[j]!r} has zero mask==1 values on the fit split; "
                "using mean=0 std=1."
            )
            warn_msgs.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        mean64 = sums[j] / float(count)
        var64 = (sumsq[j] / float(count)) - (mean64 * mean64)
        mean[j] = mean64.float()
        raw_std = float(torch.sqrt(torch.clamp(var64, min=0.0)).item())
        if not math.isfinite(raw_std) or raw_std < float(std_eps):
            msg = (
                f"compute_y_edge_scaler: column {cols[j]!r} has std={raw_std!r} (< eps={std_eps}); "
                "clamping std to eps for numerical stability."
            )
            warn_msgs.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            std[j] = torch.tensor(float(std_eps), dtype=torch.float32)
        else:
            std[j] = torch.tensor(raw_std, dtype=torch.float32)

    return YEdgeScalerFitResult(
        mean=mean,
        std=std,
        columns=cols,
        masked_value_counts_per_column=counts,
        warnings=warn_msgs,
    )


def compute_y_edge_scaler_from_arrays(
    edge_iter: Iterable[tuple[Sequence[Sequence[float]], Sequence[float], Sequence[str]]],
    *,
    stream_target_dim: int | None = None,
    std_eps: float = 1e-6,
    log_column_name: str | None = None,
    mass_flow_transform: str = "log1p",
    mass_flow_log_tau: float = 1.0,
    mass_flow_log_scale: float = 1.0,
    mass_flow_log_eps: float = 1.0e-8,
) -> tuple[YEdgeScalerFitResult, YEdgeLog1pColumnScalerFitResult | None]:
    """Compute raw y-edge scaler and optional transformed-column scaler in one lightweight pass."""
    first_columns: Sequence[str] | None = None
    dim = int(stream_target_dim) if stream_target_dim is not None else 0
    if dim <= 0:
        dim = len(STREAM_EDGE_FEATURE_SLOTS)
    sums = torch.zeros(dim, dtype=torch.float64)
    sumsq = torch.zeros(dim, dtype=torch.float64)
    counts_t = torch.zeros(dim, dtype=torch.long)
    warn_msgs: list[str] = []
    seen_records = 0

    log_col_idx: int | None = None
    log_total = 0.0
    log_total_sq = 0.0
    log_count = 0
    log_warn_msgs: list[str] = []

    for yt, ym, columns in edge_iter:
        seen_records += 1
        if first_columns is None:
            first_columns = list(columns) if columns else list(STREAM_EDGE_FEATURE_SLOTS)
            cols = list(first_columns)
            if len(cols) < dim:
                cols = cols + [f"y_edge_{i}" for i in range(len(cols), dim)]
            if log_column_name is not None:
                if log_column_name not in cols:
                    raise ValueError(
                        f"compute_y_edge_scaler_from_arrays: missing column {log_column_name!r}; columns={cols}."
                    )
                log_col_idx = int(cols.index(log_column_name))
        if not yt:
            continue
        n_edge = len(yt)
        if len(ym) != n_edge:
            raise ValueError(
                f"compute_y_edge_scaler_from_arrays: y_edge_mask length {len(ym)} != y_edge_true rows {n_edge}."
            )
        for ei, row in enumerate(yt):
            if float(ym[ei]) <= 0.0:
                continue
            if len(row) != dim:
                raise ValueError(f"compute_y_edge_scaler_from_arrays: expected row width {dim}, got {len(row)}.")
            for j in range(dim):
                fv = float(row[j])
                sums[j] += fv
                sumsq[j] += fv * fv
                counts_t[j] += 1
            if log_column_name is not None:
                if log_col_idx is None:
                    raise AssertionError("log column index was not initialized.")
                fv_log = transform_mass_flow_scalar(
                    float(row[log_col_idx]),
                    transform=mass_flow_transform,
                    tau=mass_flow_log_tau,
                    scale=mass_flow_log_scale,
                    eps=mass_flow_log_eps,
                )
                log_total += fv_log
                log_total_sq += fv_log * fv_log
                log_count += 1

    if seen_records <= 0:
        raise ValueError("compute_y_edge_scaler_from_arrays: empty dataset.")
    cols = list(first_columns) if first_columns else list(STREAM_EDGE_FEATURE_SLOTS)
    if len(cols) < dim:
        cols = cols + [f"y_edge_{i}" for i in range(len(cols), dim)]

    mean = torch.zeros(dim, dtype=torch.float32)
    std = torch.ones(dim, dtype=torch.float32)
    counts: list[int] = []
    for j in range(dim):
        count = int(counts_t[j].item())
        counts.append(count)
        if count <= 0:
            mean[j] = 0.0
            std[j] = 1.0
            msg = (
                f"compute_y_edge_scaler: column {cols[j]!r} has zero mask==1 values on the fit split; "
                "using mean=0 std=1."
            )
            warn_msgs.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        mean64 = sums[j] / float(count)
        var64 = (sumsq[j] / float(count)) - (mean64 * mean64)
        mean[j] = mean64.float()
        raw_std = float(torch.sqrt(torch.clamp(var64, min=0.0)).item())
        if not math.isfinite(raw_std) or raw_std < float(std_eps):
            msg = (
                f"compute_y_edge_scaler: column {cols[j]!r} has std={raw_std!r} (< eps={std_eps}); "
                "clamping std to eps for numerical stability."
            )
            warn_msgs.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            std[j] = torch.tensor(float(std_eps), dtype=torch.float32)
        else:
            std[j] = torch.tensor(raw_std, dtype=torch.float32)

    log_fit: YEdgeLog1pColumnScalerFitResult | None = None
    if log_column_name is not None:
        if log_count <= 0:
            msg = (
                f"compute_y_edge_log1p_column_scaler: column {log_column_name!r} has zero mask==1 values on the "
                "fit split; using mean=0 std=1."
            )
            log_warn_msgs.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            log_fit = YEdgeLog1pColumnScalerFitResult(
                mean=torch.tensor(0.0, dtype=torch.float32),
                std=torch.tensor(1.0, dtype=torch.float32),
                column=log_column_name,
                count=0,
                warnings=log_warn_msgs,
            )
        else:
            log_mean = log_total / float(log_count)
            log_var = max(log_total_sq / float(log_count) - log_mean * log_mean, 0.0)
            log_std = math.sqrt(log_var)
            if not math.isfinite(log_std) or log_std < float(std_eps):
                msg = (
                    f"compute_y_edge_log1p_column_scaler: column {log_column_name!r} has std={log_std!r} "
                    f"(< eps={std_eps}); clamping std to eps for numerical stability."
                )
                log_warn_msgs.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                log_std = float(std_eps)
            log_fit = YEdgeLog1pColumnScalerFitResult(
                mean=torch.tensor(float(log_mean), dtype=torch.float32),
                std=torch.tensor(float(log_std), dtype=torch.float32),
                column=log_column_name,
                count=int(log_count),
                warnings=log_warn_msgs,
            )

    return (
        YEdgeScalerFitResult(
            mean=mean,
            std=std,
            columns=cols,
            masked_value_counts_per_column=counts,
            warnings=warn_msgs,
        ),
        log_fit,
    )


def compute_y_edge_log1p_column_scaler(
    dataset: Iterable[GraphSampleRecord],
    *,
    column_name: str,
    stream_target_dim: int | None = None,
    std_eps: float = 1e-6,
    mass_flow_transform: str = "log1p",
    mass_flow_log_tau: float = 1.0,
    mass_flow_log_scale: float = 1.0,
    mass_flow_log_eps: float = 1.0e-8,
) -> YEdgeLog1pColumnScalerFitResult:
    """Train-only scaler after the configured non-negative Mass_Flow transform."""
    target_col = str(column_name)
    first_columns: Sequence[str] | None = None
    col_idx: int | None = None
    dim = int(stream_target_dim) if stream_target_dim is not None else 0
    if dim <= 0:
        dim = len(STREAM_EDGE_FEATURE_SLOTS)
    total = 0.0
    total_sq = 0.0
    count = 0
    seen_records = 0
    warn_msgs: list[str] = []

    for record in dataset:
        seen_records += 1
        if first_columns is None:
            first_columns = record.graph.edge_target_columns or list(STREAM_EDGE_FEATURE_SLOTS)
            cols = list(first_columns) if first_columns else list(STREAM_EDGE_FEATURE_SLOTS)
            if len(cols) < dim:
                cols = cols + [f"y_edge_{i}" for i in range(len(cols), dim)]
            if target_col not in cols:
                raise ValueError(f"compute_y_edge_log1p_column_scaler: missing column {target_col!r}; columns={cols}.")
            col_idx = int(cols.index(target_col))
        yt = record.graph.y_edge_true
        ym = record.graph.y_edge_mask
        if not yt:
            continue
        n_edge = len(yt)
        if len(ym) != n_edge:
            raise ValueError(
                f"compute_y_edge_log1p_column_scaler: y_edge_mask length {len(ym)} != y_edge_true rows {n_edge}."
            )
        if col_idx is None:
            raise AssertionError("column index was not initialized.")
        for ei, row in enumerate(yt):
            if float(ym[ei]) <= 0.0:
                continue
            if len(row) != dim:
                raise ValueError(
                    f"compute_y_edge_log1p_column_scaler: expected row width {dim}, got {len(row)} "
                    f"(process_id={record.graph.process_id!r})."
                )
            fv = transform_mass_flow_scalar(
                float(row[col_idx]),
                transform=mass_flow_transform,
                tau=mass_flow_log_tau,
                scale=mass_flow_log_scale,
                eps=mass_flow_log_eps,
            )
            total += fv
            total_sq += fv * fv
            count += 1

    if seen_records <= 0:
        raise ValueError("compute_y_edge_log1p_column_scaler: empty dataset.")
    if count <= 0:
        msg = (
            f"compute_y_edge_log1p_column_scaler: column {target_col!r} has zero mask==1 values on the fit split; "
            "using mean=0 std=1."
        )
        warn_msgs.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return YEdgeLog1pColumnScalerFitResult(
            mean=torch.tensor(0.0, dtype=torch.float32),
            std=torch.tensor(1.0, dtype=torch.float32),
            column=target_col,
            count=0,
            warnings=warn_msgs,
        )

    mean = total / float(count)
    var = max(total_sq / float(count) - mean * mean, 0.0)
    raw_std = math.sqrt(var)
    if not math.isfinite(raw_std) or raw_std < float(std_eps):
        msg = (
            f"compute_y_edge_log1p_column_scaler: column {target_col!r} has std={raw_std!r} (< eps={std_eps}); "
            "clamping std to eps for numerical stability."
        )
        warn_msgs.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        raw_std = float(std_eps)

    return YEdgeLog1pColumnScalerFitResult(
        mean=torch.tensor(float(mean), dtype=torch.float32),
        std=torch.tensor(float(raw_std), dtype=torch.float32),
        column=target_col,
        count=int(count),
        warnings=warn_msgs,
    )
