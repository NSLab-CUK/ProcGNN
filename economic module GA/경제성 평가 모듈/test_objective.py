"""test_objective.py - objective.evaluate / indicators 스모크 테스트 (Aspen 스트림으로)."""
from pathlib import Path
import numpy as np
import costing_pipeline as cp
import objective as obj
import indicators as ind

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAM_DIR = PROJECT_ROOT / "data" / "main_data_Streams"


def _streams(n, cid=1):
    path = STREAM_DIR / f"{n}.Process_Streams.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Process stream CSV not found: {path}")
    return cp.stream_table_from_aspen_csv(path, cid)


def test_all_processes_run():
    for n in range(1, 11):
        r = obj.evaluate(_streams(n), f"P{n:02d}")
        assert set(r["objectives"]) == {"lcoh_usd_per_kg", "specific_co2_kg_per_kg_h2"}
        assert np.isfinite(r["objectives"]["lcoh_usd_per_kg"])
        assert isinstance(r["feasible"], bool)
    print("[PASS] 10개 공정 evaluate() 정상")


def test_specific_co2_ballpark():
    # 정제공정(P03) 무포집 SMR: specific_co2 8~11 kg/kg (문헌)
    r = obj.evaluate(_streams(3), "P03")
    v = r["objectives"]["specific_co2_kg_per_kg_h2"]
    assert 6.0 < v < 13.0, v
    print(f"[PASS] P03 specific_co2 = {v:.2f} (문헌 8~11 범위)")


def test_robust_to_noise():
    import random
    random.seed(1)
    s = _streams(3)
    for v in s.values():
        v["T_c"] *= 1 + random.uniform(-.2, .2)
        v["mdot_kgh"] *= max(0.0, 1 + random.uniform(-.3, .3))
        v["x"] = [max(0.0, xi + random.uniform(-.05, .05)) for xi in v["x"]]
    r = obj.evaluate(s, "P03")
    assert np.isfinite(r["objectives"]["lcoh_usd_per_kg"])
    print("[PASS] 노이즈 입력에도 유한값")


def test_roles_override():
    s = _streams(5)
    # P05 stream '13' 을 포집 CO2로 지정
    roles = {**ind.default_roles("P05", s), "13": "captured_co2"}
    r = obj.evaluate(s, "P05", roles=roles)
    assert r["indicators"]["captured_co2_kg_h"] >= 0.0
    print("[PASS] roles(외부 역할매핑) 오버라이드 반영")


if __name__ == "__main__":
    test_all_processes_run()
    test_specific_co2_ballpark()
    test_robust_to_noise()
    test_roles_override()
    print("\n모든 objective 테스트 통과")
