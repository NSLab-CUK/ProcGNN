#!/usr/bin/env python3
"""Create individually titled P01--P10 process-flowsheet panels.

The original process-flow diagrams remain untouched.  This script makes
publication-ready copies with a centred top subtitle, from ``(a) P01`` through
``(j) P10``, on one common canvas aspect ratio.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PROCESS_IDS = tuple(range(1, 11))
PANEL_LETTERS = tuple("abcdefghij")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="data/process_overall_img")
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "flowsheet_panels_p01_p10_subtitles_top"
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--figure-width", type=float, default=12.0)
    parser.add_argument(
        "--title-size",
        type=float,
        default=48.0,
        help="Subtitle size in points (48 pt is twice the original 24 pt).",
    )
    parser.add_argument(
        "--title-space",
        type=float,
        default=1.30,
        help="Reserved top-title height in inches.",
    )
    parser.add_argument(
        "--process-ids",
        type=int,
        nargs="+",
        default=list(PROCESS_IDS),
        help="Process IDs to render; defaults to all P01--P10.",
    )
    return parser.parse_args()


def _find_process_image(image_dir: Path, process_id: int) -> Path:
    matches = [
        path
        for path in sorted(image_dir.glob("*.png"))
        if (found := re.match(r"^(\d+)", path.stem)) and int(found.group(1)) == process_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one source image for P{process_id:02d}; found {matches}"
        )
    return matches[0]


def _render_panel(
    source: Path,
    destination: Path,
    subtitle: str,
    *,
    figure_width: float,
    common_image_aspect: float,
    title_size: float,
    title_space: float,
    dpi: int,
) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        # Keep the original PFD aspect ratio.  The common image region is
        # sized using the tallest of all ten source diagrams, so every saved
        # panel has an identical outer canvas without stretching any PFD.
        image_height_inches = figure_width * common_image_aspect
        figure_height = image_height_inches + title_space
        figure = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
        axis = figure.add_axes([0.01, 0.0, 0.98, image_height_inches / figure_height])
        axis.imshow(image, interpolation="lanczos", aspect="equal")
        axis.set_anchor("C")
        axis.axis("off")
        figure.text(
            0.5,
            1.0 - title_space / (2.0 * figure_height),
            subtitle,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="normal",
            color="#20252B",
        )
        figure.savefig(destination.with_suffix(".png"), dpi=dpi, facecolor="white")
        figure.savefig(destination.with_suffix(".pdf"), facecolor="white")
        figure.savefig(destination.with_suffix(".svg"), facecolor="white")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    requested = tuple(dict.fromkeys(args.process_ids))
    unknown = sorted(set(requested).difference(PROCESS_IDS))
    if unknown:
        raise ValueError(f"Unsupported process IDs: {unknown}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        process_id: _find_process_image(image_dir, process_id)
        for process_id in PROCESS_IDS
    }
    image_aspects: list[float] = []
    for source in source_paths.values():
        with Image.open(source) as image:
            width, height = image.size
            image_aspects.append(height / width)
    common_image_aspect = max(image_aspects)

    rows: list[str] = [
        "# Titled process flowsheet panels",
        "",
        "Each panel is a copy of its original PFD with a centred top subtitle.",
        "All ten panels use the same outer canvas aspect ratio; source PFDs retain",
        "their native aspect ratio and are centred with white padding where needed.",
        "The source diagrams were not modified.",
        "",
        "| Panel | Process | Source |",
        "|---|---|---|",
    ]
    for process_id in requested:
        letter = PANEL_LETTERS[process_id - 1]
        source = source_paths[process_id]
        subtitle = f"({letter}) P{process_id:02d}"
        stem = output_dir / f"flowsheet_panel_{letter}_P{process_id:02d}"
        _render_panel(
            source,
            stem,
            subtitle,
            figure_width=args.figure_width,
            common_image_aspect=common_image_aspect,
            title_size=args.title_size,
            title_space=args.title_space,
            dpi=args.dpi,
        )
    for process_id, letter in zip(PROCESS_IDS, PANEL_LETTERS):
        source = source_paths[process_id]
        rows.append(f"| ({letter}) P{process_id:02d} | P{process_id:02d} | `{source.as_posix()}` |")
    (output_dir / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"rendered={len(requested)}; output={output_dir}")


if __name__ == "__main__":
    main()
