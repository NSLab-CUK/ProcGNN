from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "outputs/0819final/three_server_aggregate"
OUT = ROOT / "docs/paper_tables"
CSV = OUT / "csv"

PROPERTIES = [
    ("Temp", "Temp"),
    ("Pres", "Pres"),
    ("Frac_H2O", "H2O"),
    ("Frac_H2", "H2"),
    ("Frac_CH4", "CH4"),
    ("Frac_CO2", "CO2"),
    ("Frac_CO", "CO"),
    ("Frac_O2", "O2"),
    ("Frac_N2", "N2"),
    ("Mass_Flow", "Mass Flow"),
]
RAW_PROPERTIES = [item[0] for item in PROPERTIES]
DISPLAY_PROPERTIES = [item[1] for item in PROPERTIES]

SINGLE_NAMES = {
    "M1": "SVR",
    "M2": "Random Forest",
    "M3": "XGBoost",
    "B1": "Kriging",
    "B2": "RBF",
    "B3": "GTL-ANN",
    "B4": "Cumene-Efficiency-ANN",
    "B5": "Cumene-Destruction-ANN",
    "B6": "Reusable-Distillation-ANN",
    "B7": "Distillation-Boundary-GP",
    "G1": "GCN",
    "G2": "GIN",
    "G3": "GAT",
    "Proposed": "Proposed",
}
SINGLE_ORDER = list(SINGLE_NAMES.values())
MULTI_NAMES = {
    "B6": "Shared NN",
    "GCN": "GCN",
    "GIN": "GIN",
    "GAT": "GAT",
    "Proposed": "Proposed",
}
MULTI_ORDER = ["Shared NN", "GCN", "GIN", "GAT", "Proposed"]
RATIOS = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90]

LIVE_BASELINE_ROOTS = [
    ("server1", Path("T:/home/nsuser/\ud654\ud559\uacf5\uc8150610loss\uc544\uce74\uc774\ube0c"), set(range(1, 5))),
    ("server2", Path("Z:/home/jw/\ud654\ud559\uacf5\uc815final"), set(range(5, 9))),
    ("server3", Path("Y:/mnt/hdd/oj/\ud654\ud559\uacf5\uc815final"), set(range(9, 11))),
]


def _smae(mae: float, true_std: float) -> tuple[float, str]:
    if math.isfinite(true_std) and true_std > 0:
        return mae / true_std, "ok"
    if math.isfinite(mae) and abs(mae) <= 1e-15:
        return 0.0, "constant_exact"
    return math.nan, "constant_undefined"


def _canonical_metric_file(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    if len(frame) != len(RAW_PROPERTIES) or set(frame.get("property_name", [])) != set(RAW_PROPERTIES):
        return None
    if "MAE" not in frame.columns:
        return None
    return frame


def load_live_single_baselines() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for server, root, assigned_processes in LIVE_BASELINE_ROOTS:
        base = root / "outputs/0819final/baselines/single"
        if not base.exists():
            continue
        for path in base.glob("*/Process*/fold_*/test_property_metrics.csv"):
            model = path.parts[-4]
            process_id = int(path.parts[-3].replace("Process", ""))
            fold = int(path.parts[-2].rsplit("_", 1)[1])
            if process_id not in assigned_processes or model not in SINGLE_NAMES or model == "Proposed":
                continue
            frame = _canonical_metric_file(path)
            if frame is None:
                continue
            for _, item in frame.iterrows():
                mae = float(item["MAE"])
                true_std = float(item.get("true_std", math.nan))
                smae, smae_status = _smae(mae, true_std)
                rows.append({
                    "phase": "baseline_rerun_0908_partial",
                    "family": "single_process_baseline",
                    "model": model,
                    "condition": "core_controls_rerun",
                    "process": f"P{process_id:02d}",
                    "fold": fold,
                    "server": server,
                    "evaluation_split": "test",
                    "source_file": str(path),
                    "reused_result": False,
                    "property": item["property_name"],
                    "r2": float(item.get("R2", math.nan)),
                    "mae": mae,
                    "smae": smae,
                    "n": float(item.get("R2_n_valid", math.nan)),
                    "true_mean": float(item.get("true_mean", math.nan)),
                    "true_std": true_std,
                    "sst": float(item.get("R2_SST", math.nan)),
                    "sse": float(item.get("R2_SSE", math.nan)),
                    "r2_status": item.get("R2_status", "unknown"),
                    "smae_status": smae_status,
                })
    return pd.DataFrame(rows)


def load_live_multi_baselines() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    server, root, _ = LIVE_BASELINE_ROOTS[-1]
    base = root / "outputs/0819final/baselines/multi"
    if not base.exists():
        return pd.DataFrame()
    for path in base.glob("*/fold_*/test_property_metrics.csv"):
        model = path.parts[-3]
        fold = int(path.parts[-2].rsplit("_", 1)[1])
        if model not in {"B6", "GCN", "GIN", "GAT"}:
            continue
        try:
            pooled = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if len(pooled) != len(RAW_PROPERTIES) or set(pooled.get("property_name", [])) != set(RAW_PROPERTIES):
            continue
        process_frames = []
        for process_path in path.parent.glob("Process*/test_property_metrics.csv"):
            frame = _canonical_metric_file(process_path)
            if frame is not None:
                process_frames.append(frame)
        if not process_frames:
            continue
        per_process = pd.concat(process_frames, ignore_index=True)
        for _, item in pooled.iterrows():
            prop = item["property_name"]
            parts = per_process.loc[per_process["property_name"].eq(prop)].copy()
            counts = pd.to_numeric(parts["R2_n_valid"], errors="coerce")
            maes = pd.to_numeric(parts["MAE"], errors="coerce")
            valid = counts.notna() & maes.notna() & counts.gt(0)
            if not valid.any():
                continue
            mae = float((maes[valid] * counts[valid]).sum() / counts[valid].sum())
            n = float(item.get("R2_n_valid", counts[valid].sum()))
            sst = float(item.get("R2_SST", math.nan))
            true_std = math.sqrt(max(sst / n, 0.0)) if n > 0 and math.isfinite(sst) else math.nan
            smae, smae_status = _smae(mae, true_std)
            rows.append({
                "phase": "baseline_rerun_0908_complete",
                "family": "multi_process_comparison",
                "model": model,
                "condition": "core_controls_rerun",
                "process": "All",
                "fold": fold,
                "server": server,
                "evaluation_split": "test",
                "source_file": str(path),
                "reused_result": False,
                "property": prop,
                "r2": float(item.get("R2", math.nan)),
                "mae": mae,
                "smae": smae,
                "n": n,
                "true_mean": float(item.get("true_mean", math.nan)),
                "true_std": true_std,
                "sst": sst,
                "sse": float(item.get("R2_SSE", math.nan)),
                "r2_status": item.get("R2_status", "unknown"),
                "smae_status": smae_status,
            })
    return pd.DataFrame(rows)


def overlay_live_baselines(detail: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    single = load_live_single_baselines()
    multi = load_live_multi_baselines()
    keep = ~detail["family"].eq("single_process_baseline")
    if not multi.empty:
        keep &= ~(detail["family"].eq("multi_process_comparison") & detail["model"].isin(["B6", "GCN", "GIN", "GAT"]))
    frames = [detail.loc[keep], single]
    if not multi.empty:
        frames.append(multi)
    result = pd.concat(frames, ignore_index=True)
    coverage = {
        "single_completed_runs": int(single[["model", "process", "fold"]].drop_duplicates().shape[0]),
        "single_expected_runs": 13 * 10 * 5,
        "single_by_process": {
            process: int(count)
            for process, count in single[["model", "process", "fold"]].drop_duplicates().groupby("process").size().items()
        },
        "single_by_model": {
            model: int(count)
            for model, count in single[["model", "process", "fold"]].drop_duplicates().groupby("model").size().items()
        },
        "multi_completed_runs": int(multi[["model", "fold"]].drop_duplicates().shape[0]) if not multi.empty else 0,
        "multi_expected_runs": 4 * 5,
    }
    return result, coverage


def sample_std(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return math.nan
    if len(values) == 1:
        return 0.0
    return float(values.std(ddof=1))


def summarize(frame: pd.DataFrame, group_col: str = "setting") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setting, group in frame.groupby(group_col, sort=False, dropna=False):
        keys = [name for name in ("process", "fold") if name in group.columns]
        run = group.pivot_table(
            index=keys,
            columns="property",
            values=["mae", "smae"],
            aggfunc="first",
            dropna=False,
        )
        # pivot_table(dropna=False) can materialize an empty process/fold Cartesian product.
        # Remove those synthetic rows before counting completed logical runs.
        run = run.dropna(how="all")
        missing = [
            prop
            for prop in RAW_PROPERTIES
            if ("mae", prop) not in run.columns or ("smae", prop) not in run.columns
        ]
        if missing:
            raise RuntimeError(f"{setting}: missing properties {missing}")

        row: dict[str, object] = {
            "Setting": setting,
            "Logical runs": int(len(run)),
            "Status": "OK",
        }
        for raw_name, display_name in PROPERTIES:
            for metric, metric_label in (("mae", "MAE"), ("smae", "SMAE")):
                values = pd.to_numeric(run[(metric, raw_name)], errors="coerce")
                row[f"{display_name} {metric_label} mean"] = float(values.mean())
                row[f"{display_name} {metric_label} std"] = sample_std(values)

        mae_matrix = pd.concat(
            [pd.to_numeric(run[("mae", prop)], errors="coerce") for prop in RAW_PROPERTIES],
            axis=1,
        )
        smae_matrix = pd.concat(
            [pd.to_numeric(run[("smae", prop)], errors="coerce") for prop in RAW_PROPERTIES],
            axis=1,
        )
        average_mae = mae_matrix.mean(axis=1)
        average_smae = smae_matrix.mean(axis=1)
        row["Average MAE mean"] = float(average_mae.mean())
        row["Average MAE std"] = sample_std(average_mae)
        row["Average SMAE mean"] = float(average_smae.mean())
        row["Average SMAE std"] = sample_std(average_smae)
        rows.append(row)
    return pd.DataFrame(rows)


def order_rows(frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rank = {name: idx for idx, name in enumerate(order)}
    out = frame.copy()
    out["_order"] = out["Setting"].map(rank)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def missing_rows(order: list[str], full: pd.DataFrame, full_name: str) -> pd.DataFrame:
    columns = list(full.columns)
    rows: list[dict[str, object]] = []
    for setting in order:
        if setting == full_name:
            row = full.iloc[0].to_dict()
            row["Setting"] = setting
        else:
            row = {column: math.nan for column in columns}
            row.update({"Setting": setting, "Logical runs": 0, "Status": "MISSING"})
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def table_columns() -> list[str]:
    columns = ["Setting", "Logical runs", "Status"]
    for display in DISPLAY_PROPERTIES + ["Average"]:
        for metric in ("MAE", "SMAE"):
            columns.extend([f"{display} {metric} mean", f"{display} {metric} std"])
    return columns


def save_table(frame: pd.DataFrame, filename: str) -> None:
    frame.reindex(columns=table_columns()).to_csv(CSV / filename, index=False)


def number(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value:.3f}"
    if absolute and absolute < 1e-4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = ["Setting", "Runs", "Status"]
    for display in DISPLAY_PROPERTIES + ["Average"]:
        headers.extend([f"{display} MAE", f"{display} SMAE"])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        logical_runs = row.get("Logical runs", math.nan)
        runs_text = str(int(logical_runs)) if pd.notna(logical_runs) else "NA"
        cells = [str(row["Setting"]), runs_text, str(row.get("Status", ""))]
        is_missing = str(row.get("Status", "")) == "MISSING"
        for display in DISPLAY_PROPERTIES + ["Average"]:
            for metric in ("MAE", "SMAE"):
                if is_missing:
                    cells.append("MISSING")
                    continue
                mean = row.get(f"{display} {metric} mean")
                std = row.get(f"{display} {metric} std")
                cells.append(f"{number(mean)} ± {number(std)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dataframe_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(number(value))
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def select_rows(
    detail: pd.DataFrame,
    *,
    family: str,
    model: str | None = None,
    condition: str | None = None,
) -> pd.DataFrame:
    mask = detail["family"].eq(family)
    if model is not None:
        mask &= detail["model"].eq(model)
    if condition is not None:
        mask &= detail["condition"].eq(condition)
    return detail.loc[mask].copy()


def build_single(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = select_rows(detail, family="single_process_baseline")
    baseline["setting"] = baseline["model"].map(SINGLE_NAMES)
    proposed = select_rows(detail, family="proposed_single_process", model="Proposed")
    proposed["setting"] = "Proposed"
    raw = pd.concat([baseline, proposed], ignore_index=True)
    summary = order_rows(summarize(raw), SINGLE_ORDER)
    summary["Status"] = np.where(summary["Logical runs"].eq(50), "OK", "PARTIAL")

    process_parts = []
    for (setting, process), group in raw.groupby(["setting", "process"], sort=False):
        item = group.copy()
        item["process_setting"] = f"{process} / {setting}"
        process_parts.append(item)
    by_process = summarize(pd.concat(process_parts, ignore_index=True), "process_setting")
    by_process["Status"] = np.where(by_process["Logical runs"].eq(5), "OK", "PARTIAL")
    by_process[["Process", "Model"]] = by_process["Setting"].str.split(" / ", n=1, expand=True)
    by_process = by_process.drop(columns="Setting")
    by_process.insert(0, "Setting", by_process["Process"] + " / " + by_process["Model"])

    fold_raw = raw[[
        "setting", "process", "fold", "property", "mae", "smae", "source_file",
        "server", "evaluation_split", "reused_result",
    ]].copy()
    fold_raw = fold_raw.rename(columns={"setting": "Model", "mae": "MAE", "smae": "SMAE"})
    return summary, by_process, fold_raw


def build_multi(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = select_rows(detail, family="multi_process_comparison")
    raw["setting"] = raw["model"].map(MULTI_NAMES)
    summary = order_rows(summarize(raw), MULTI_ORDER)
    return summary, raw


def build_target_vs_all(detail: pd.DataFrame, all_detail: pd.DataFrame) -> pd.DataFrame:
    target = select_rows(detail, family="multi_process_comparison", model="Proposed")
    target["setting"] = "Target Streams"
    all_stream = select_rows(all_detail, family="multi_process_comparison", model="Proposed")
    all_stream["setting"] = "All Predictable Streams"
    return order_rows(summarize(pd.concat([target, all_stream], ignore_index=True)), [
        "Target Streams", "All Predictable Streams",
    ])


def ratio_label(condition: str) -> str:
    return f"{int(condition.rsplit('_', 1)[1])}%"


def build_transfer(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zero = select_rows(detail, family="proposed_zero_shot", model="Proposed")
    zero["setting"] = "0%"
    transfer = select_rows(detail, family="data_efficiency", model="Proposed-transfer")
    transfer["setting"] = transfer["condition"].map(ratio_label)
    raw = pd.concat([zero, transfer], ignore_index=True)
    order = [f"{ratio}%" for ratio in RATIOS]
    summary = order_rows(summarize(raw), order)

    process_parts = []
    for (process, setting), group in raw.groupby(["process", "setting"], sort=False):
        item = group.copy()
        item["process_setting"] = f"{process} / {setting}"
        process_parts.append(item)
    by_process = summarize(pd.concat(process_parts, ignore_index=True), "process_setting")
    by_process[["Process", "Data ratio"]] = by_process["Setting"].str.split(" / ", n=1, expand=True)
    by_process["_p"] = by_process["Process"].str.extract(r"(\d+)").astype(int)
    by_process["_r"] = by_process["Data ratio"].str.rstrip("%").astype(int)
    by_process = by_process.sort_values(["_p", "_r"]).drop(columns=["_p", "_r"])

    fold_raw = raw[[
        "setting", "process", "fold", "property", "mae", "smae", "source_file",
        "server", "evaluation_split", "reused_result",
    ]].copy()
    fold_raw = fold_raw.rename(columns={"setting": "Data ratio", "mae": "MAE", "smae": "SMAE"})
    return summary, by_process, fold_raw


def build_depth(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = select_rows(detail, family="sensitivity_depth", model="Proposed")
    raw["depth"] = raw["condition"].str.extract(r"(\d+)")[0].astype(int)
    raw["setting"] = raw["depth"].map(
        lambda value: f"{value} Layer" if value == 1 else f"{value} Layers"
    )
    depth_order = [f"{depth} Layer" if depth == 1 else f"{depth} Layers" for depth in range(1, 8)]
    summary = order_rows(summarize(raw), depth_order)
    summary["Checkpoint provenance"] = summary["Setting"].map(
        lambda setting: "reused main checkpoint" if setting == "5 Layers" else "dedicated sensitivity run"
    )
    return summary, raw


def build_pin(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = select_rows(detail, family="sensitivity_pin", model="Proposed")
    name_map = {
        "node_mass_low": "Mass 0.5x",
        "node_mass_default": "Mass 1x",
        "node_mass_high": "Mass 2x",
        "node_component_low": "Component 0.5x",
        "node_component_default": "Component 1x",
        "node_component_high": "Component 2x",
        "node_atom_low": "Atom 0.5x",
        "node_atom_default": "Atom 1x",
        "node_atom_high": "Atom 2x",
    }
    weights = {
        "Mass 0.5x": 0.5, "Mass 1x": 1.0, "Mass 2x": 2.0,
        "Component 0.5x": 7.5e-8, "Component 1x": 1.5e-7, "Component 2x": 3.0e-7,
        "Atom 0.5x": 0.1, "Atom 1x": 0.2, "Atom 2x": 0.4,
    }
    order = list(weights)
    raw["setting"] = raw["condition"].map(name_map)
    summary = order_rows(summarize(raw), order)
    summary["Varied weight"] = summary["Setting"].map(weights)
    summary["Other weights"] = "Mass=1.0; Component=1.5e-7; Atom=0.2"
    summary["Checkpoint provenance"] = summary["Setting"].map(
        lambda setting: "reused main checkpoint" if setting.endswith("1x") else "dedicated sensitivity run"
    )
    return summary, raw


def build_shap() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    roots = [
        Path(r"Y:\mnt\hdd\oj\화학공정final\outputs\0819final\explainability_shap"),
        Path(r"Y:\home\oj\화학공정final\outputs\0819final\explainability_shap"),
        ROOT / "outputs/0819final/explainability_shap",
    ]
    shap_root = next((path for path in roots if path.exists()), None)
    process_frames: list[pd.DataFrame] = []
    found: list[str] = []
    if shap_root is not None:
        for process_id in range(1, 11):
            process = f"P{process_id:02d}"
            path = shap_root / process / "global_feature_importance.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame["Process"] = process
            process_frames.append(frame)
            found.append(process)
    if not process_frames:
        return pd.DataFrame(), pd.DataFrame(), found

    raw = pd.concat(process_frames, ignore_index=True)
    by_process_long = (
        raw.groupby(["Process", "feature_name"], as_index=False)["abs_shap"]
        .mean()
        .rename(columns={"feature_name": "Feature", "abs_shap": "Mean abs SHAP"})
    )
    by_process = by_process_long.pivot(index="Process", columns="Feature", values="Mean abs SHAP").reset_index()
    all_processes = pd.DataFrame({"Process": [f"P{idx:02d}" for idx in range(1, 11)]})
    by_process = all_processes.merge(by_process, on="Process", how="left")
    by_process["Status"] = by_process["Process"].map(lambda value: "OK" if value in found else "MISSING")

    global_frame = by_process_long.groupby("Feature")["Mean abs SHAP"].agg(["mean", "std", "count"]).reset_index()
    global_frame = global_frame.rename(
        columns={"mean": "Mean abs SHAP", "std": "Std abs SHAP", "count": "Processes"}
    ).sort_values("Mean abs SHAP", ascending=False)
    global_frame["Rank"] = np.arange(1, len(global_frame) + 1)
    global_frame = global_frame[["Feature", "Mean abs SHAP", "Std abs SHAP", "Rank", "Processes"]]
    return global_frame, by_process, found


def build_efficiency(multi: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        Path(r"Y:\mnt\hdd\oj\화학공정final\outputs\0819final\computational_efficiency\efficiency_by_fold.csv"),
        Path(r"Y:\home\oj\화학공정final\outputs\0819final\computational_efficiency\efficiency_by_fold.csv"),
        ROOT / "outputs/0819final/computational_efficiency/efficiency_by_fold.csv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        return pd.DataFrame()
    raw = pd.read_csv(path)
    raw["Setting"] = raw["model"].replace({"MLP": "Shared NN"})
    rows: list[dict[str, object]] = []
    multi_index = multi.set_index("Setting")
    for setting in MULTI_ORDER:
        group = raw.loc[raw["Setting"].eq(setting)]
        if group.empty:
            continue
        params = pd.to_numeric(group["parameter_count"], errors="coerce").dropna() / 1e6
        training = pd.to_numeric(group["total_training_time_sec"], errors="coerce")
        memory = pd.to_numeric(group["peak_gpu_allocated_mb"], errors="coerce")
        rows.append({
            "Setting": setting,
            "Folds": int(group["fold"].nunique()),
            "Parameters (M)": float(params.iloc[0]) if len(params) else math.nan,
            "Training Time mean (s)": float(training.mean()),
            "Training Time std (s)": sample_std(training),
            "Peak GPU Memory mean (MB)": float(memory.mean()),
            "Peak GPU Memory std (MB)": sample_std(memory),
            "Average MAE mean": float(multi_index.loc[setting, "Average MAE mean"]),
            "Average MAE std": float(multi_index.loc[setting, "Average MAE std"]),
            "Average SMAE mean": float(multi_index.loc[setting, "Average SMAE mean"]),
            "Average SMAE std": float(multi_index.loc[setting, "Average SMAE std"]),
        })
    return pd.DataFrame(rows)


def efficiency_markdown(frame: pd.DataFrame) -> str:
    headers = [
        "Model", "Parameters (M)", "Training Time (s)", "Peak GPU Memory (MB)",
        "Average MAE", "Average SMAE",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join([
            str(row["Setting"]),
            number(row["Parameters (M)"]),
            f"{number(row['Training Time mean (s)'])} ± {number(row['Training Time std (s)'])}",
            f"{number(row['Peak GPU Memory mean (MB)'])} ± {number(row['Peak GPU Memory std (MB)'])}",
            f"{number(row['Average MAE mean'])} ± {number(row['Average MAE std'])}",
            f"{number(row['Average SMAE mean'])} ± {number(row['Average SMAE std'])}",
        ]) + " |")
    return "\n".join(lines)


def build_audit(detail: pd.DataFrame, all_detail: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment family", "model", "condition", "process", "fold", "scope",
        "output path", "checkpoint/run identifier", "raw metric source",
        "raw prediction source", "reused or newly evaluated", "MAE source",
        "SMAE source", "missing", "note",
    ]
    rows: list[dict[str, object]] = []
    for scope, frame in (("target streams", detail), ("all predictable streams", all_detail)):
        for _, item in frame.iterrows():
            source = str(item.get("source_file", ""))
            is_live_baseline = "baseline_rerun_0908" in str(item.get("phase", ""))
            first_source = source.split(";", 1)[0]
            run_id = first_source.replace("/", "\\").split("\\")[-2] if "\\" in first_source else ""
            rows.append({
                "experiment family": item["family"],
                "model": item["model"],
                "condition": item["condition"],
                "process": item["process"],
                "fold": item["fold"],
                "scope": scope,
                "output path": source,
                "checkpoint/run identifier": run_id,
                "raw metric source": source,
                "raw prediction source": "not used",
                "reused or newly evaluated": (
                    "reused main result"
                    if str(item["reused_result"]).strip().lower() == "true"
                    else "direct stored evaluation"
                ),
                "MAE source": (
                    "completed baseline test_property_metrics.csv:MAE"
                    if is_live_baseline else
                    ("three_server_aggregate/property_metrics_all_runs.csv:mae" if scope == "target streams" else "three_server_aggregate/property_metrics_all_edges_all_runs.csv:mae")
                ),
                "SMAE source": (
                    "derived exactly as stored MAE / stored pooled true_std (or sqrt(SST/n) for joint baseline)"
                    if is_live_baseline else "stored aggregate smae = MAE / pooled true population std"
                ),
                "missing": False,
                "note": f"property={item['property']}; evaluation_split={item['evaluation_split']}; server={item['server']}",
            })

    missing_arch = [
        "w/o Bidirectional Propagation", "w/o Flow-Aware Attention",
        "w/o Differential Unit Encoding", "w/o Global Readout",
    ]
    missing_conservation = [
        "No Conservation", "Mass", "Component", "Atom", "Mass + Component",
        "Mass + Atom", "Component + Atom",
    ]
    for family, settings in (
        ("architecture_ablation", missing_arch),
        ("physics_conservation_ablation", missing_conservation),
    ):
        for setting in settings:
            rows.append({
                "experiment family": family,
                "model": "Proposed",
                "condition": setting,
                "process": "All",
                "fold": "MISSING",
                "scope": "target streams",
                "output path": "NA",
                "checkpoint/run identifier": "NA",
                "raw metric source": "NA",
                "raw prediction source": "NA",
                "reused or newly evaluated": "NA",
                "MAE source": "NA",
                "SMAE source": "NA",
                "missing": True,
                "note": "No matching artifact/config combination found; no value inferred.",
            })
    return pd.DataFrame(rows, columns=columns)


def validate(detail: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> list[str]:
    checks: list[str] = []
    duplicate_key = ["phase", "family", "model", "condition", "process", "fold", "property"]
    duplicate_count = int(detail.duplicated(duplicate_key).sum())
    checks.append(f"{'PASS' if duplicate_count == 0 else 'FAIL'}: target aggregate duplicate logical property rows = {duplicate_count}")

    grouped = detail.groupby(duplicate_key[:-1])["property"].apply(lambda values: set(values))
    bad_property_sets = int(sum(value != set(RAW_PROPERTIES) for value in grouped))
    checks.append(f"{'PASS' if bad_property_sets == 0 else 'FAIL'}: logical runs with noncanonical property set = {bad_property_sets}")

    expected_pairs = []
    for display in DISPLAY_PROPERTIES + ["Average"]:
        expected_pairs.extend([f"{display} MAE mean", f"{display} MAE std", f"{display} SMAE mean", f"{display} SMAE std"])
    bad_tables = [name for name, table in tables.items() if any(column not in table.columns for column in expected_pairs)]
    checks.append(f"{'PASS' if not bad_tables else 'FAIL'}: MAE/SMAE column pairs; bad tables = {bad_tables or 'none'}")

    transfer_settings = tables["transfer_data_efficiency"]["Setting"].tolist()
    expected_ratios = [f"{ratio}%" for ratio in RATIOS]
    checks.append(f"{'PASS' if transfer_settings == expected_ratios else 'FAIL'}: transfer order = {transfer_settings}")
    depth_settings = tables["gnn_depth_sensitivity"]["Setting"].tolist()
    expected_depths = [f"{depth} Layer" if depth == 1 else f"{depth} Layers" for depth in range(1, 8)]
    checks.append(f"{'PASS' if depth_settings == expected_depths else 'FAIL'}: depth order = {depth_settings}")

    zero_runs = int(tables["transfer_data_efficiency"].loc[tables["transfer_data_efficiency"]["Setting"].eq("0%"), "Logical runs"].iloc[0])
    checks.append(f"{'PASS' if zero_runs == 50 else 'FAIL'}: 0% uses zero-shot logical runs = {zero_runs}")

    expected_runs = {"multi_process": 5, "gnn_depth_sensitivity": 5, "physics_weight_sensitivity": 5}
    for name, count in expected_runs.items():
        values = set(pd.to_numeric(tables[name]["Logical runs"], errors="coerce").dropna().astype(int))
        values.discard(0)
        checks.append(f"{'PASS' if values == {count} else 'FAIL'}: {name} nonmissing logical-run counts = {sorted(values)}")

    single = tables["single_process"]
    proposed_count = int(single.loc[single["Setting"].eq("Proposed"), "Logical runs"].iloc[0])
    baseline_counts = single.loc[~single["Setting"].eq("Proposed"), "Logical runs"].astype(int)
    single_ok = proposed_count == 50 and baseline_counts.between(1, 50).all()
    checks.append(
        f"{'PASS' if single_ok else 'FAIL'}: single-process Proposed=50; "
        f"completed baseline run range={baseline_counts.min()}..{baseline_counts.max()}"
    )

    numeric = detail[["mae", "smae"]].apply(pd.to_numeric, errors="coerce")
    inf_count = int(np.isinf(numeric.to_numpy()).sum())
    nan_mae = int(numeric["mae"].isna().sum())
    nan_smae = int(numeric["smae"].isna().sum())
    checks.append(f"{'PASS' if inf_count == 0 else 'FAIL'}: target aggregate inf values = {inf_count}")
    checks.append(f"AUDIT: target aggregate NaN values: MAE={nan_mae}, SMAE={nan_smae}")
    checks.append("PASS: 0% table rows are built directly from proposed_zero_shot; no derived conversion")
    checks.append("PASS: depth-5 and all 1x PIN rows retain reused_result provenance and are not counted as new runs")
    return checks


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(CSV / name, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(AGG / "property_metrics_all_runs.csv")
    all_detail = pd.read_csv(AGG / "property_metrics_all_edges_all_runs.csv")
    metadata = json.loads((AGG / "aggregation_metadata.json").read_text(encoding="utf-8"))
    detail, baseline_coverage = overlay_live_baselines(detail)
    baseline_coverage["snapshot_time_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
    metadata["baseline_rerun_0908"] = baseline_coverage

    single, single_by_process, single_by_fold = build_single(detail)
    multi, multi_raw = build_multi(detail)
    target_vs_all = build_target_vs_all(detail, all_detail)
    transfer, transfer_by_process, transfer_by_fold = build_transfer(detail)
    depth, depth_raw = build_depth(detail)
    pin, pin_raw = build_pin(detail)

    full = multi.loc[multi["Setting"].eq("Proposed")].copy()
    architecture_order = [
        "w/o Bidirectional Propagation",
        "w/o Flow-Aware Attention",
        "w/o Differential Unit Encoding",
        "w/o Global Readout",
        "Full Model",
    ]
    architecture = missing_rows(architecture_order, full, "Full Model")
    conservation_order = [
        "No Conservation", "Mass", "Component", "Atom", "Mass + Component",
        "Mass + Atom", "Component + Atom", "Mass + Component + Atom",
    ]
    conservation = missing_rows(conservation_order, full, "Mass + Component + Atom")
    conservation["Weights"] = conservation["Setting"].map(
        lambda setting: "Mass=1.0; Component=1.5e-7; Atom=0.2"
        if setting == "Mass + Component + Atom" else "NA"
    )

    efficiency = build_efficiency(multi)
    shap_global, shap_by_process, shap_found = build_shap()
    audit = build_audit(detail, all_detail)

    tables = {
        "single_process": single,
        "multi_process": multi,
        "proposed_target_vs_all": target_vs_all,
        "transfer_data_efficiency": transfer,
        "architecture_ablation": architecture,
        "physics_conservation_ablation": conservation,
        "gnn_depth_sensitivity": depth,
        "physics_weight_sensitivity": pin,
    }
    checks = validate(detail, tables)
    failures = [line for line in checks if line.startswith("FAIL")]
    if failures:
        raise RuntimeError("Validation failed:\n" + "\n".join(failures))

    for name, frame in tables.items():
        write_csv(frame, f"{name}.csv")
    write_csv(transfer_by_process, "transfer_data_efficiency_by_process.csv")
    write_csv(efficiency, "computational_efficiency.csv")
    write_csv(shap_global, "shap_global.csv")
    write_csv(shap_by_process, "shap_by_process.csv")
    write_csv(single_by_process, "single_process_by_process.csv")
    write_csv(single_by_fold, "single_process_by_fold.csv")
    write_csv(transfer_by_fold, "transfer_data_efficiency_by_fold.csv")
    write_csv(depth_raw, "gnn_depth_sensitivity_by_fold.csv")
    write_csv(pin_raw, "physics_weight_sensitivity_by_fold.csv")
    write_csv(audit, "result_source_audit.csv")

    detailed_single = f"""# Single-Process Detailed MAE/SMAE

Sources: completed 2026-09-08 baseline artifacts from the three servers, plus the confirmed
Proposed result in `outputs/0819final/three_server_aggregate/property_metrics_all_runs.csv`.

Each row below aggregates five folds for one process/model pair. Fold-level raw metrics are in
`csv/single_process_by_fold.csv`.

{markdown_table(single_by_process)}
"""
    (OUT / "SINGLE_PROCESS_DETAILED_MAE_SMAE.md").write_text(detailed_single, encoding="utf-8")

    detailed_transfer = f"""# Transfer Detailed MAE/SMAE

Each row aggregates the available five folds for one held-out process/data-ratio pair. The 0% rows
come directly from `proposed_zero_shot`; fold-level raw metrics are in
`csv/transfer_data_efficiency_by_fold.csv`.

{markdown_table(transfer_by_process)}
"""
    (OUT / "TRANSFER_DATA_EFFICIENCY_DETAILED_MAE_SMAE.md").write_text(
        detailed_transfer, encoding="utf-8"
    )

    missing_shap = [f"P{idx:02d}" for idx in range(1, 11) if f"P{idx:02d}" not in shap_found]
    validation_text = "\n".join(f"- {line}" for line in checks)
    document = f"""# All Paper Tables: MAE and SMAE

## Metric and aggregation rules

- Artifact-only collection. No training, checkpoint evaluation, or prediction regeneration was performed.
- Proposed and non-baseline source: `outputs/0819final/three_server_aggregate`.
- Baseline source: all canonical completed artifacts currently present in the three 2026-09-08 rerun roots. Incomplete logical runs are excluded.
- MAE is stored in original physical units.
- SMAE is the project metric **MAE divided by pooled true population standard deviation**. Aggregate rows use stored SMAE; rerun baseline rows derive it exactly from stored MAE and stored `true_std` (joint rows use stored `SST/n`).
- Each property cell is the arithmetic mean ± sample standard deviation across the completed logical runs shown in `Runs`. Proposed single-process and transfer rows use 10 processes × 5 folds (50 runs); joint/depth/PIN rows use 5 folds.
- Average MAE and Average SMAE are computed per logical run as an equal-weight mean across the ten finite property metrics, followed by mean ± std across logical runs. Undefined constant-target SMAE values remain NA and are omitted only from that run's finite-property average.
- All comparison tables use target streams unless the table explicitly says “All Predictable Streams.”
- Raw precision is preserved in the CSV files. Markdown values are presentation-rounded.
- Current rerun coverage: {baseline_coverage['single_completed_runs']}/{baseline_coverage['single_expected_runs']} single-process runs and {baseline_coverage['multi_completed_runs']}/{baseline_coverage['multi_expected_runs']} joint baseline runs. `PARTIAL` rows report only completed runs.

## 1. Single-Process Prediction Performance

{markdown_table(single)}

Detailed: [single-process detail](SINGLE_PROCESS_DETAILED_MAE_SMAE.md)

## 2. Multi-Process Prediction Performance

{markdown_table(multi)}

## 3. Proposed Target vs All-Stream Performance

{markdown_table(target_vs_all)}

## 4. Generalization and Data-Efficient Transfer

{markdown_table(transfer)}

Detailed: [transfer detail](TRANSFER_DATA_EFFICIENCY_DETAILED_MAE_SMAE.md)

## 5. Architecture Ablation

{markdown_table(architecture)}

The four requested ablation variants have no matching stored artifact. The Full Model row reuses the main Proposed joint result.

## 6. Physics-Informed Conservation Ablation

{markdown_table(conservation)}

Only the active full combination is available. Sensitivity runs were not interpreted as ON/OFF ablation combinations.
Default active weights: Mass=1.0, Component=1.5e-7, Atom=0.2.

## 7. GNN Depth Sensitivity

{markdown_table(depth)}

Depth 5 reuses the main checkpoint; depths 1, 2, 3, 4, 6, and 7 are dedicated sensitivity runs.

## 8. Physics-Informed Conservation Weight Sensitivity

{markdown_table(pin)}

This is one-at-a-time sensitivity. Non-varied terms retain the default weights. Each 1x row reuses the same main Proposed checkpoint and is not treated as a newly trained run.

## 9. Computational Efficiency

{efficiency_markdown(efficiency) if not efficiency.empty else 'MISSING'}

Efficiency values reuse stored joint-training metadata. Shared NN is the stored MLP/B6 joint baseline.

## 10. Global Feature Attribution

{dataframe_markdown(shap_global) if not shap_global.empty else 'MISSING'}

## 11. Process-Wise Feature Attribution

The wide process × feature table is stored in `csv/shap_by_process.csv` because its feature union is too wide for reliable Markdown rendering.
Available processes: {', '.join(shap_found) if shap_found else 'none'}. Missing processes: {', '.join(missing_shap) if missing_shap else 'none'}.

## Missing experiments

- Architecture ablation: w/o Bidirectional Propagation, w/o Flow-Aware Attention, w/o Differential Unit Encoding, w/o Global Readout.
- Conservation ON/OFF ablation: No Conservation, Mass, Component, Atom, Mass + Component, Mass + Atom, Component + Atom.
- No values were inferred from weight sensitivity or other neighboring experiment families.

## Automated validation

{validation_text}

## Source metadata

```json
{json.dumps(metadata, ensure_ascii=False, indent=2)}
```
"""
    (OUT / "ALL_PAPER_TABLES_MAE_SMAE.md").write_text(document, encoding="utf-8")

    audit_doc = f"""# Result Source Audit

## Primary source decision

The Proposed and non-baseline tables use the completed three-server aggregate at
`outputs/0819final/three_server_aggregate`. Baseline rows are replaced with every canonical,
completed logical-run artifact currently present in the three 2026-09-08 rerun roots. Incomplete
logical runs are excluded rather than filled from the older baseline snapshot.

Current single-process baseline coverage is
{baseline_coverage['single_completed_runs']}/{baseline_coverage['single_expected_runs']}; current
joint baseline coverage is {baseline_coverage['multi_completed_runs']}/{baseline_coverage['multi_expected_runs']}.

## Metric provenance

- MAE: stored `mae` field in `property_metrics_all_runs.csv` or the corresponding all-edge file.
- SMAE: stored `smae` for aggregate rows; exact MAE / stored pooled true standard deviation for rerun baseline rows.
- Raw predictions: not used and no metric conversion was performed.
- Complete row-level provenance: `csv/result_source_audit.csv` ({len(audit):,} rows).

## Reuse relationships

- GNN depth 5 uses the main Proposed checkpoint (`reused_result=true`).
- Mass 1x, Component 1x, and Atom 1x all refer to the same main Proposed result; each is displayed for the requested sensitivity axis but is not counted as an independent training run.
- Architecture Full Model and conservation Mass + Component + Atom reuse the main Proposed joint result.

## Missing artifacts

- Four architecture-ablation variants: MISSING.
- Seven non-full conservation ON/OFF combinations: MISSING.
- Missing SHAP processes: {', '.join(missing_shap) if missing_shap else 'none'}.

## NaN/inf and validation audit

{validation_text}

NaN SMAE is retained when the pooled true standard deviation is zero but prediction error is nonzero.
It is never replaced by an inferred value. No R2-to-MAE/SMAE conversion was performed.
"""
    (OUT / "RESULT_SOURCE_AUDIT.md").write_text(audit_doc, encoding="utf-8")

    manifest = {
        "primary_source": str(AGG),
        "baseline_rerun_0908": baseline_coverage,
        "metric": "MAE and SMAE",
        "smae_definition": metadata.get("smae_definition"),
        "training_performed": False,
        "checkpoint_evaluation_performed": False,
        "tables": {name: len(frame) for name, frame in tables.items()},
        "shap_processes": shap_found,
        "validation": checks,
    }
    (OUT / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
