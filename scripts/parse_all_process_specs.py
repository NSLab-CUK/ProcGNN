from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from process_graph import parse_process_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing Process*_Adjacency_Matrix.xlsx files",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save parsed process specs as json",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("Process*_Adjacency_Matrix.xlsx"))

    if not files:
        raise FileNotFoundError(f"No process xlsx files found in {input_dir}")

    parsed = []
    for fp in files:
        process_id = fp.stem.replace("_Adjacency_Matrix", "")
        spec = parse_process_file(str(fp), process_id=process_id)
        parsed.append(spec)

        num_nodes = len(spec.node_names)
        num_edges = len(spec.edge_index[0])
        num_decoder = {k: len(v) for k, v in spec.decoder_spec.items()}
        print(f"[OK] {process_id}: nodes={num_nodes}, edges={num_edges}, decoder={num_decoder}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in parsed], f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()