"""Answer-edge mapping utilities for edge_all evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Tuple


_COEF_PREFIX = re.compile(r"^\s*[\d.]+\s*\*\s*(.+)$")
_LINEAR_EXPR = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\*\s*([A-Za-z0-9_]+)\s*$")


@dataclass(frozen=True)
class AnswerMetricSpec:
    task_name: str
    target_column: str
    primary_metric: str
    formula: str
    auxiliary_metric: str
    scale_factor: float


def resolve_answer_target_column_index(
    *,
    task_name: str,
    target_column: str,
    edge_target_columns: Sequence[str],
) -> Tuple[int, str]:
    """
    Return (column_index_in_edge_target_columns, rule_description_for_logging).

    Raises RuntimeError if no unique mapping exists.
    """
    cols = [str(c) for c in edge_target_columns]
    tc_raw = str(target_column).strip()
    if tc_raw in cols:
        return cols.index(tc_raw), f"exact_match:{tc_raw!r}"

    m = _COEF_PREFIX.match(tc_raw)
    tc = m.group(1).strip() if m else tc_raw
    if tc in cols:
        return cols.index(tc), f"strip_linear_coefficient:{tc_raw!r}->{tc!r}"

    upper = tc.upper()
    candidates: list[tuple[int, str]] = []

    if task_name == "target_h2":
        # Prefer mole-flow family when the column name is *MoleFlow / Mole_Flow* style so that
        # names like PROD_H2_MoleFlow (or 0.8*PROD_H2_MoleFlow) do not also match the broad Frac_H2 rule.
        moleflow_style = "MOLEFLOW" in upper or "_MOLEFLOW" in upper or "MOLE_FLOW" in upper
        if moleflow_style:
            for i, c in enumerate(cols):
                if str(c) == "Mole_Flow":
                    candidates.append((i, "task_h2_moleflow_family->Mole_Flow"))
        elif "H2" in upper and "H2O" not in upper:
            for i, c in enumerate(cols):
                if str(c) == "Frac_H2":
                    candidates.append((i, "task_h2_frac_family->Frac_H2"))
    if task_name == "tailgas_co2":
        if "MOLEFLOW" in upper.replace("_", "") or "MOLE_FLOW" in upper:
            for i, c in enumerate(cols):
                if str(c) == "Mole_Flow":
                    candidates.append((i, "task_co2_moleflow_family->Mole_Flow"))
        elif "CO2" in upper:
            for i, c in enumerate(cols):
                if str(c) == "Frac_CO2":
                    candidates.append((i, "task_co2_frac_family->Frac_CO2"))

    if len(candidates) == 1:
        idx, rule = candidates[0]
        return idx, f"{rule}(from target_column={target_column!r})"
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous mapping for task_name={task_name!r} target_column={target_column!r} "
            f"into edge_target_columns={cols!r}; candidates={candidates!r}."
        )
    raise RuntimeError(
        f"Cannot map task_name={task_name!r} target_column={target_column!r} "
        f"into edge_target_columns={cols!r} (no exact match and no rule applied)."
    )


def parse_target_column_expression(target_column: str) -> tuple[float, str]:
    raw = str(target_column).strip()
    m = _LINEAR_EXPR.match(raw)
    if m:
        return float(m.group(1)), str(m.group(2))
    return 1.0, raw


def resolve_answer_metric_spec(*, task_name: str, target_column: str) -> AnswerMetricSpec:
    scale_factor, base_col = parse_target_column_expression(target_column)
    upper = base_col.upper()
    if "H2" in upper and "H2O" not in upper and ("MOLE" in upper):
        return AnswerMetricSpec(
            task_name=task_name,
            target_column=target_column,
            primary_metric="derived_H2_amount",
            formula=f"{scale_factor:g} * Mole_Flow * Frac_H2",
            auxiliary_metric="Frac_H2",
            scale_factor=scale_factor,
        )
    if "CO2" in upper and ("MOLE" in upper):
        return AnswerMetricSpec(
            task_name=task_name,
            target_column=target_column,
            primary_metric="derived_CO2_amount",
            formula=f"{scale_factor:g} * Mole_Flow * Frac_CO2",
            auxiliary_metric="Frac_CO2",
            scale_factor=scale_factor,
        )
    # Fallback keeps current behavior for non-mole targets.
    aux = "Frac_H2" if task_name == "target_h2" else "Frac_CO2"
    return AnswerMetricSpec(
        task_name=task_name,
        target_column=target_column,
        primary_metric=f"mapped_{aux}",
        formula=aux,
        auxiliary_metric=aux,
        scale_factor=1.0,
    )
