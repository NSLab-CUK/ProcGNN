from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


STREAM_KEY_CANONICALIZATION_VERSION = 1
STREAM_KEY_CANONICAL_COLUMN = "_STREAM_KEY_CANONICAL"
_MISSING_TEXT = {"", "nan", "none", "null", "<na>"}


def _normalize_unicode_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value), flags=re.UNICODE).strip()


def canonicalize_stream_key(
    value: Any,
    process_id: Any = None,
    source: Any = None,
) -> str:
    """Return the lossless canonical key used for graph-to-stream joins."""
    del process_id, source  # Kept in the API for diagnostic call sites.
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = _normalize_unicode_whitespace(value)
    if text.lower() in _MISSING_TEXT:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite() or number != number.to_integral_value():
        return text
    return str(int(number))


def canonicalize_process_id(value: Any) -> str:
    """Normalize supported process identifiers to ``Pxx``."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = _normalize_unicode_whitespace(value)
    if text.lower() in _MISSING_TEXT:
        return ""
    match = re.fullmatch(r"(?i)(?:process|p)?0*(\d+)", text)
    if match is None:
        raise ValueError(
            f"Unsupported process_id={value!r}; expected forms such as 3, 03, P3, P03, or Process3."
        )
    process_number = int(match.group(1))
    if process_number <= 0:
        raise ValueError(f"process_id must be positive, got {value!r}.")
    return f"P{process_number:02d}"


class CanonicalStreamKeyCollisionError(ValueError):
    pass


class MissingRequiredStreamKeyError(ValueError):
    pass


def build_canonical_stream_row_index(
    stream_rows: pd.DataFrame,
    *,
    process_id: Any,
    sample_id: Any,
    csv_path: str | Path,
    raw_key_column: str = "Stream_Name",
) -> Mapping[str, pd.Series]:
    """Build one unambiguous row per canonical stream key for one sample."""
    if raw_key_column not in stream_rows.columns:
        raise KeyError(f"Missing required stream key column {raw_key_column!r}: {csv_path}")
    canonical_to_rows: dict[str, list[int]] = {}
    for row_position, raw_key in enumerate(stream_rows[raw_key_column].tolist()):
        canonical_key = canonicalize_stream_key(
            raw_key,
            process_id=process_id,
            source=csv_path,
        )
        if not canonical_key:
            continue
        canonical_to_rows.setdefault(canonical_key, []).append(row_position)

    collisions = {
        key: positions for key, positions in canonical_to_rows.items() if len(positions) > 1
    }
    if collisions:
        details = []
        for canonical_key, positions in sorted(collisions.items()):
            raw_keys = [stream_rows.iloc[position][raw_key_column] for position in positions]
            details.append(
                f"canonical={canonical_key!r} raw_keys={raw_keys!r} row_positions={positions!r}"
            )
        raise CanonicalStreamKeyCollisionError(
            "Canonical stream-key collision; refusing ambiguous join. "
            f"process_id={canonicalize_process_id(process_id)!r} sample_id={sample_id!r} "
            f"csv_path={str(csv_path)!r} collisions=[{'; '.join(details)}]"
        )

    return {
        canonical_key: stream_rows.iloc[positions[0]]
        for canonical_key, positions in canonical_to_rows.items()
    }


def missing_required_stream_key_error(
    *,
    process_id: Any,
    sample_id: Any,
    canonical_edge_id: Any,
    raw_graph_key: Any,
    canonical_graph_key: str,
    available_csv_keys: list[str],
    csv_path: str | Path,
) -> MissingRequiredStreamKeyError:
    return MissingRequiredStreamKeyError(
        "Missing required stream row; refusing silent zero supervision. "
        f"process_id={canonicalize_process_id(process_id)!r} sample_id={sample_id!r} "
        f"canonical_edge_id={str(canonical_edge_id)!r} raw_graph_key={raw_graph_key!r} "
        f"canonical_graph_key={canonical_graph_key!r} "
        f"available_csv_keys={sorted(available_csv_keys)!r} csv_path={str(csv_path)!r}"
    )
