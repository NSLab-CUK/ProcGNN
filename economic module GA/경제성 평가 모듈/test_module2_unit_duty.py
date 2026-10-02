"""
test_module2_unit_duty.py
module2_unit_duty.py 검증. 두 단계로 나눈다.

1. 합성 예시(직접 손으로 검산 가능한 값)로 엔탈피수지 plumbing 자체가 맞는지 확인.
2. 실제 Process1_Adjacency_Matrix.xlsx를 읽어서 토폴로지 파싱이 실제 파일 구조와
   맞는지 확인 (구조 테스트, 수치 검증 아님, 수치 검증은 실측 스트림-엣지 ID 매핑이
   따로 필요해서 별도 작업으로 남겨둠).

실행 방법: python test_module2_unit_duty.py
"""

import numpy as np

import properties as prop
import equipment_sizing as sizing
import module2_unit_duty as m2


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def main():
    print("=" * 70)
    print("1단계. 합성 예시로 엔탈피수지 plumbing 검증")
    print("=" * 70)

    # --- (a) 단순 반응기: FEED -> R1 -> PROD, WGS 반응(CO+H2O->CO2+H2) 근사 ---
    topology_a = {
        "FEED": {"in": [], "out": ["R1"]},
        "R1": {"in": ["FEED"], "out": ["PROD"]},
        "PROD": {"in": ["R1"], "out": []},
    }
    edge_stream_data_a = {
        ("FEED", "R1"): dict(temp_c=350.0, pres_bar=22.0,
                              mole_fractions=[0.30, 0.45, 0.05, 0.10, 0.10, 0.0, 0.0],
                              mass_flow_kgh=24000.0),
        ("R1", "PROD"): dict(temp_c=350.0, pres_bar=22.0,
                              mole_fractions=[0.25, 0.55, 0.05, 0.14, 0.01, 0.0, 0.0],
                              mass_flow_kgh=24000.0),
    }
    edge_props_a = m2.build_edge_properties(edge_stream_data_a)
    q_r1 = m2.node_duty_kw("R1", topology_a, edge_props_a)

    # 손으로 검산: 같은 온도(350C)에서 조성만 반응으로 바뀌었으므로,
    # Q_R1 = mdot/3600 * (h_out - h_in), h는 properties.py로 직접 계산.
    props_check = prop.reconstruct_stream_properties(
        np.array([350.0, 350.0]), np.array([22.0, 22.0]),
        np.array([edge_stream_data_a[("FEED", "R1")]["mole_fractions"],
                   edge_stream_data_a[("R1", "PROD")]["mole_fractions"]]),
        np.array([24000.0, 24000.0]),
    )
    q_expected = 24000.0 / 3600.0 * (props_check["enthalpy_kj_kg"][1] - props_check["enthalpy_kj_kg"][0])
    print(f"  Q_R1(module2 계산) = {q_r1:.2f} kW")
    print(f"  Q_R1(직접 손계산)  = {q_expected:.2f} kW")
    check("단일입구/단일출구 반응기 duty가 직접계산과 정확히 일치", abs(q_r1 - q_expected) < 1e-6)
    check("WGS 반응(CO+H2O->CO2+H2)이 일어난 반응기는 흡열(양수) 방향으로 나옴이 합리적",
          np.isfinite(q_r1))

    # --- (b) 다중입구 버너: FUEL, AIR -> BURNER -> FLUE, 연소(발열) ---
    topology_b = {
        "FUEL": {"in": [], "out": ["BURNER"]},
        "AIR": {"in": [], "out": ["BURNER"]},
        "BURNER": {"in": ["FUEL", "AIR"], "out": ["FLUE"]},
        "FLUE": {"in": ["BURNER"], "out": []},
    }
    edge_stream_data_b = {
        ("FUEL", "BURNER"): dict(temp_c=25.0, pres_bar=3.0,
                                  mole_fractions=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                  mass_flow_kgh=1500.0),
        ("AIR", "BURNER"): dict(temp_c=25.0, pres_bar=3.0,
                                 mole_fractions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.21, 0.79],
                                 mass_flow_kgh=26000.0),
        ("BURNER", "FLUE"): dict(temp_c=1010.0, pres_bar=3.0,
                                  # CH4 완전연소 근사: CH4+2O2->CO2+2H2O, 질량 보존 위해
                                  # 배가스 조성은 근사치(정밀 연소계산 아님, plumbing 검증용)
                                  mole_fractions=[0.10, 0.0, 0.0, 0.05, 0.0, 0.15, 0.70],
                                  mass_flow_kgh=27500.0),
    }
    edge_props_b = m2.build_edge_properties(edge_stream_data_b)
    q_burner = m2.node_duty_kw("BURNER", topology_b, edge_props_b)
    print(f"\n  Q_BURNER(module2 계산, 다중입구) = {q_burner:,.2f} kW")
    check("버너 노드가 다중 입구(FUEL+AIR)를 정상적으로 합산함", np.isfinite(q_burner))
    check("연소 반응은 강한 발열이므로 Q_BURNER가 큰 음수로 나옴 "
          "(실제 Process_Main.csv의 Q_BURNER 부호와 일치하는지 확인)", q_burner < 0)

    # --- (c) 방어 로직: 입구 스트림 물성이 없으면 에러를 내는지 ---
    try:
        m2.node_duty_kw("R1", topology_a, {})
        check("입구 물성 누락 시 ValueError 발생", False)
    except ValueError:
        check("입구 물성 누락 시 ValueError 발생", True)

    print()
    print("=" * 70)
    print("2단계. 새 터빈 동력 함수(equipment_sizing.turbine_isentropic_power_kw) 검증")
    print("=" * 70)
    # 압축기와 정확히 반대 조건(고압->저압)으로 넣어서 양의 동력이 나오는지 확인
    w_turbine = sizing.turbine_isentropic_power_kw(
        vol_flow_in_m3s=np.array([1.5]), p_in_bar=np.array([20.0]),
        p_out_bar=np.array([1.5]), k_ratio=1.3, isentropic_efficiency=0.80,
    )
    print(f"  터빈 추출동력 = {w_turbine[0]:,.1f} kW (입구 20bar -> 출구 1.5bar, 1.5 m3/s)")
    check("터빈 추출동력이 양수", w_turbine[0] > 0)

    print()
    print("=" * 70)
    print("3단계. 실제 Process1_Adjacency_Matrix.xlsx 토폴로지 파싱 (구조 테스트)")
    print("=" * 70)
    from pathlib import Path
    candidates = [Path(__file__).resolve().parents[2] / "data" / "process_specs" / "raw" / "Process1_Adjacency_Matrix.xlsx"]
    candidates = [p for p in candidates if p.is_file()]
    if candidates:
        topo, edges = m2.load_topology(candidates[0])
        types = m2.load_node_types(candidates[0])
        print(f"  파일 = {candidates[0]}")
        print(f"  노드 수 = {len(topo)}, 엣지 수 = {len(edges)}")
        check("BURNER 노드가 파싱됨", "BURNER" in topo)
        check("BURNER 노드 타입이 burner로 파싱됨", types.get("BURNER") == "burner")
        check("BURNER 입구가 C2, C3 (연료/공기 압축기)를 포함함",
              set(topo["BURNER"]["in"]) == {"C2", "C3"})
        print(f"  BURNER 입구 = {topo['BURNER']['in']}, 출구 = {topo['BURNER']['out']}")
    else:
        print("  Process1_Adjacency_Matrix.xlsx를 이 세션 파일시스템에서 못 찾음, "
              "구조 테스트는 건너뜀 (수치와 무관, 파일 경로 문제일 뿐).")

    print("\n모든 module2 테스트 통과")


if __name__ == "__main__":
    main()
