"""
demo_objective.py
objective.evaluate() 를 10개 공정에 대해 Aspen ground truth 스트림으로 실행해
GA 목적함수 출력(LCOH, specific_co2)과 지표·제약을 보여준다.
GNN 예측본이 준비되면 stream_table_from_aspen_csv 대신 GNN 예측 dict 만 넣으면 된다.

    python demo_objective.py            # 10개 공정, case 1 요약
    python demo_objective.py P03 5      # 공정 3, case 5, 상세
"""
import os
import sys
from pathlib import Path

import costing_pipeline as cp
import objective as obj

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAM_DIR = Path(
    os.environ.get("SMR_STREAM_DIR", PROJECT_ROOT / "data" / "steady state_june")
)


def streams(n, cid):
    matches = sorted(STREAM_DIR.glob(f"{n}.Process_Streams*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No Aspen stream CSV for process {n} under {STREAM_DIR}. "
            "Set SMR_STREAM_DIR to the directory containing '<process>.Process_Streams*.csv'."
        )
    return cp.stream_table_from_aspen_csv(
        str(matches[0]), cid)


def summary(cid=1):
    print(f"{'proc':5}{'LCOH':>8}{'specCO2':>9}{'H2 kt/yr':>10}{'FCI M$':>8}"
          f"{'thEff':>7}{'S/C':>6}  feasible / violated")
    for n in range(1, 11):
        r = obj.evaluate(streams(n, cid), f"P{n:02d}")
        o, m = r["objectives"], r["indicators"]
        v = [c for c, cv in r["constraints"].items() if not cv["ok"]]
        print(f"P{n:02d}{o['lcoh_usd_per_kg']:8.2f}{o['specific_co2_kg_per_kg_h2']:9.1f}"
              f"{m['h2_production_kg_h']*8000/1e6:10.1f}{m['fci_usd']/1e6:8.0f}"
              f"{m['thermal_efficiency']:7.2f}{m['sc_ratio']:6.2f}  {r['feasible']} {v}")


def detail(pid, cid):
    r = obj.evaluate(streams(int(pid[1:]), cid), pid)
    print(f"\n=== {pid}  (case {cid}) ===")
    print(" objectives :", {k: round(v, 3) for k, v in r["objectives"].items()})
    print(" feasible   :", r["feasible"])
    print(" constraints:")
    for c, cv in r["constraints"].items():
        mark = "ok " if cv["ok"] else "VIOL"
        print(f"   {mark} {c:26s} violation={cv['violation']:.3f}")
    print(" indicators :")
    for k in ("h2_production_kg_h", "ch4_conversion", "h2_yield_mol_mol", "h2_purity",
              "sc_ratio", "specific_co2_kg_per_kg_h2", "specific_energy_MJ_per_kg_h2",
              "thermal_efficiency", "min_approach_temp_K", "fci_usd", "tci_usd",
              "opex_usd_per_year", "lcoh_usd_per_kg", "lcoh_gross_usd_per_kg",
              "coproduct_credit_usd_per_year", "furnace_duty_kW"):
        if k in r["indicators"]:
            print(f"   {k:32s} {r['indicators'][k]:,.3f}")
    print(" roles used :", r["indicators"]["roles_used"])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        detail(sys.argv[1].upper(), int(sys.argv[2]) if len(sys.argv) >= 3 else 1)
    else:
        summary(1)
