from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _process_number(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"cannot parse process id: {value!r}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preview SHAP flowsheet coordinates")
    parser.add_argument("--coordinates", default="data/process_overall_img/unit_coordinates.csv")
    parser.add_argument("--canonical-nodes", default="data/reference/canonical_nodes.csv")
    parser.add_argument("--image-root", default="data/process_overall_img")
    parser.add_argument("--preview-root", default="outputs/final_paper/shap_coordinate_preview")
    args = parser.parse_args()

    coordinates = pd.read_csv(PROJECT_ROOT / args.coordinates)
    canonical = pd.read_csv(PROJECT_ROOT / args.canonical_nodes)
    required = {"process_id", "node_name", "x", "y", "coordinate_source"}
    missing_columns = sorted(required.difference(coordinates.columns))
    if missing_columns:
        raise ValueError(f"coordinate columns missing: {missing_columns}")
    coordinates["process_number"] = coordinates["process_id"].map(_process_number)
    canonical["process_number"] = canonical["process_id"].map(_process_number)
    duplicate = coordinates.duplicated(["process_number", "node_name"], keep=False)
    if duplicate.any():
        raise ValueError(
            "duplicate coordinates: "
            + str(coordinates.loc[duplicate, ["process_number", "node_name"]].to_dict("records"))
        )

    preview_root = PROJECT_ROOT / args.preview_root
    preview_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    palette = {"image_verified": "#00bcd4", "boundary_anchor": "#ff9800", "canonical_only": "#e91e63"}
    for process_id in range(1, 11):
        expected = canonical.loc[canonical["process_number"].eq(process_id), "node_name"].astype(str)
        selected = coordinates.loc[coordinates["process_number"].eq(process_id)].copy()
        actual = set(selected["node_name"].astype(str))
        missing = sorted(set(expected).difference(actual))
        extra = sorted(actual.difference(set(expected)))
        image_path = PROJECT_ROOT / args.image_root / f"{process_id}번.png"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
            background = image.copy()
        out_of_bounds = selected.loc[
            selected["x"].lt(0) | selected["x"].ge(width)
            | selected["y"].lt(0) | selected["y"].ge(height),
            ["node_name", "x", "y"],
        ].to_dict("records")
        status = "completed" if not missing and not extra and not out_of_bounds else "failed"
        reports.append({
            "process_id": process_id,
            "status": status,
            "image_width": width,
            "image_height": height,
            "canonical_nodes": len(expected),
            "coordinate_rows": len(selected),
            "missing_nodes": missing,
            "extra_nodes": extra,
            "out_of_bounds": out_of_bounds,
        })

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(background)
        for source, group in selected.groupby("coordinate_source"):
            ax.scatter(
                group["x"], group["y"], s=75,
                c=palette.get(str(source), "#8bc34a"), edgecolors="black", linewidths=0.6,
                label=str(source), zorder=3,
            )
        for row in selected.itertuples(index=False):
            ax.text(
                float(row.x) + 5, float(row.y) - 5, str(row.node_name), fontsize=7,
                color="black", bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1},
                zorder=4,
            )
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"Process {process_id}: canonical node coordinate audit")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(preview_root / f"coordinate_preview_P{process_id:02d}.png", dpi=150)
        plt.close(fig)

    payload = {
        "status": "completed" if all(row["status"] == "completed" for row in reports) else "failed",
        "coordinate_file": str((PROJECT_ROOT / args.coordinates).resolve()),
        "preview_root": str(preview_root.resolve()),
        "processes": reports,
    }
    report_path = preview_root / "coordinate_validation.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
