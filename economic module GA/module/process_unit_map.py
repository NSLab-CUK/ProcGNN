"""
process_unit_map.py
10개 SMR 공정(P01~P10) 각 노드를 경제성평가 파이프라인의
  (1) equipment_sizing 사이징 함수
  (2) turton_costing 장비군 키(K1/K2/K3, F_BM)
에 매핑하는 표. README "다음 작업 #3".

근거
  - 위상(노드/엣지): DataSet/Steady state/인접행렬/Process{n}_Adjacency_Matrix.xlsx
  - 반응기 역할 판정: steady state_june/{n}.Process_Main*.csv 의 T_R*, Q_R* 중앙값
      개질기        : T ~ 830~910 degC, Q > 0 (흡열)
      HT-WGS(중온)  : T ~ 310~410 degC, Q <= 0
      LT-WGS        : T ~ 225~250 degC, Q < 0
      연소기(burner): T 미보고 or 1010 degC, Q << 0 (대량 발열)
  - 분리기 역할: 인접행렬에서 SEP1은 항상 OUT_PROD(H2)로 나감 -> PSA.
                 F1은 항상 SEP1/OUT_H2O 로 나감 -> 응축수 녹아웃 드럼.

카테고리 -> equipment_sizing 함수 / turton 키
  compressor      : compressor_isentropic_power_kw   -> "compressor_centrifugal"   (동력 kW)
  pump            : pump_power_kw                    -> "pump_centrifugal"          (동력 kW)
  turbine         : turbine_isentropic_power_kw      -> "turbine_expander"          (동력 kW)
  process_hx      : heat_exchanger_area_m2 (U=850)   -> "heat_exchanger_floating_head" (면적 m2)
  utility_cooler  : heat_exchanger_area_m2 (U=850)   -> "heat_exchanger_floating_head" (면적 m2)
                    * 냉각수측 온도 25->40 degC 가정으로 LMTD 계산
  fired_heater    : burner_fired_heater_capacity_kw / |Q| -> "fired_heater_nonreactive" (흡수듀티 kW)
  flash_vessel    : vertical_vessel_sizing           -> "process_vessel_vertical"    (부피 m3)
  psa             : psa_bed_sizing                   -> "process_vessel_vertical" x n_beds (부피 m3)
                    + psa_adsorbent_* (별도 CAPEX/OPEX 라인)
  wgs_reactor     : wgs_reactor_sizing               -> "process_vessel_vertical"    (부피 m3)
                    + catalyst_* (별도 CAPEX/OPEX 라인)
  reformer        : 개별 계상 안 함. burner와 합쳐 "개질로 1기"로 통합
                    (fired_heater_nonreactive, 용량 = max(Σ개질 흡열, eta*버너 연소열))
  none            : 비용 없음 (믹서 MIX, 분배기 SP, 공급/제품 노드)

노드 -> 입·출구 스트림 번호 매핑은 process_stream_map.py 에 있다 (10개 공정 전부,
Aspen COM으로 각 .bkp/.apw를 headless로 열어 block.Ports에서 직접 추출).
노드명은 인접행렬(GNN 그래프) 기준으로 맞췄다. Aspen bkp의 블록명이 다른 경우:
  P01 MIX3 = bkp B2 / P06 R3_BURNER = bkp R3 / P09 MIX1 = bkp B2 / P10 BURNER = bkp BN.
P04 MIX2는 GNN 그래프에만 있고 Aspen에는 없다(BURNER가 FUEL/AIR를 직접 받음) -> 비용 없음.
"""

from __future__ import annotations

# 카테고리별 기본 연결 (equipment_sizing 함수명, turton 키)
CATEGORY_TO_COSTING = {
    "compressor":      ("compressor_isentropic_power_kw", "compressor_centrifugal"),
    "pump":            ("pump_power_kw",                   "pump_centrifugal"),
    "turbine":         ("turbine_isentropic_power_kw",     "turbine_expander"),
    "process_hx":      ("heat_exchanger_area_m2",          "heat_exchanger_floating_head"),
    "utility_cooler":  ("heat_exchanger_area_m2",          "heat_exchanger_floating_head"),
    "fired_heater":    ("burner_fired_heater_capacity_kw", "fired_heater_nonreactive"),
    "flash_vessel":    ("vertical_vessel_sizing",          "process_vessel_vertical"),
    "psa":             ("psa_bed_sizing",                  "process_vessel_vertical"),
    "wgs_reactor":     ("wgs_reactor_sizing",              "process_vessel_vertical"),
    "reformer":        (None,                              "fired_heater_nonreactive"),
    "none":            (None,                              None),
}
# "reformer" 노드(개질기 R1, P1은 R2도)와 burner 노드는 costing_pipeline에서
# "개질로(reformer furnace) 1기"로 통합 코스팅한다 (fired_heater_nonreactive,
# 용량 = max(Σ개질 흡열, eta_furnace * 버너 연소열)). 이 10개 공정의 버너는
# 전부 SMR 개질로 자체이므로 별도 계상하지 않는다. HEAT 노드(피드 예열기)는
# 별도로 자기 듀티로 코스팅한다.

# 반응기 kind 인자 (wgs_reactor_sizing 용)
#   "HT_WGS" : T >= 300 degC 인 shift 반응기 (중온 단일단 포함)
#   "LT_WGS" : T < 300 degC 인 shift 반응기

# 공정별 노드 -> (카테고리, 비고). 노드명은 인접행렬 헤더 그대로.
PROCESS_UNIT_MAP = {
    "P01": {
        "R1": ("reformer", "예비개질 631C, Q>0 (노에 통합)"),
        "R2": ("reformer", "주개질 898C, Q>0 (노에 통합)"),
        "R3": ("wgs_reactor", "HT-WGS 370C  kind=HT_WGS"),
        "R4": ("wgs_reactor", "LT-WGS 245C  kind=LT_WGS"),
        "BURNER": ("fired_heater", "|Q_BURNER| * eta_furnace, 100MW 초과분 병렬"),
        "HX1": ("process_hx", ""), "HX2": ("process_hx", "버너 배가스->exhaust"),
        "HX3": ("process_hx", ""), "HX4": ("process_hx", ""), "HX5": ("process_hx", ""),
        "C1": ("compressor", ""), "C2": ("compressor", "연료 압축"),
        "C3": ("compressor", "공기 압축"),
        "P1": ("pump", ""), "P2": ("pump", ""), "P3": ("pump", ""),
        "T1": ("turbine", ""), "T2": ("turbine", ""), "T3": ("turbine", "->RESTEAM"),
        "MIX1": ("none", ""), "MIX2": ("none", ""), "MIX3": ("none", ""),
        "SP1": ("none", ""), "SP2": ("none", ""), "SP3": ("none", ""),
    },
    "P02": {
        "R1": ("reformer", "개질 900C, Q>0 (R3 연소기에 통합)"),
        "R2": ("wgs_reactor", "LT-WGS 230C  kind=LT_WGS"),
        "R3": ("fired_heater", "연소기(T 미보고, Q_R3~-23MW). |Q_R3|*eta_furnace"),
        "HX1": ("process_hx", "->EXHAUST"), "HX2": ("process_hx", ""),
        "HX4": ("process_hx", ""),
        "COOL1": ("utility_cooler", ""), "COOL2": ("utility_cooler", ""),
        "COOL4": ("utility_cooler", ""),
        "C1": ("compressor", ""), "C2": ("compressor", ""),
        "C3": ("compressor", ""), "C4": ("compressor", ""),
        "P1": ("pump", ""),
        "SP1": ("none", ""), "MIX1": ("none", ""), "MIX2": ("none", ""),
    },
    "P03": {
        "R1": ("reformer", "개질 873C, Q>0 (BURNER에 통합)"),
        "R2": ("wgs_reactor", "WGS 401C  kind=HT_WGS"),
        "BURNER": ("fired_heater", "|Q_BURNER|*eta_furnace"),
        "HX1": ("process_hx", ""), "HX2": ("process_hx", ""), "HX3": ("process_hx", "->OUT_CO2"),
        "HX4": ("process_hx", ""), "HX5": ("process_hx", ""),
        "F1": ("flash_vessel", "응축수 녹아웃 (->SEP1, OUT_H2O)"),
        "SEP1": ("psa", "H2 정제 PSA (->OUT_PROD, 테일가스->C1 재압축)"),
        "C1": ("compressor", "PSA 테일가스 재압축"), "C2": ("compressor", ""),
        "P1": ("pump", ""),
        "MIX1": ("none", ""), "MIX2": ("none", ""),
    },
    "P04": {
        "R1": ("reformer", "개질 906C, Q>0 (BURNER에 통합)"),
        "R2": ("wgs_reactor", "HT-WGS 397C  kind=HT_WGS"),
        "R3": ("wgs_reactor", "LT-WGS 240C  kind=LT_WGS"),
        "BURNER": ("fired_heater", "연료+공기 연소. |Q_BURNER|*eta_furnace"),
        "HX1": ("process_hx", "->EXHAUST"), "HX2": ("process_hx", ""), "HX3": ("process_hx", ""),
        "COOL1": ("utility_cooler", "->SEP1"),
        "SEP1": ("psa", "H2 정제 PSA (->OUT_PROD, OUT_stream)"),
        "C1": ("compressor", ""), "P1": ("pump", ""),
        "MIX1": ("none", ""), "MIX2": ("none", ""),
    },
    "P05": {
        "R1": ("reformer", "개질 907C, Q_R1~+44MW. 버너 없음 -> 개질로가 이 흡열을 감당"),
        "R2": ("wgs_reactor", "WGS 366C  kind=HT_WGS"),
        "HEAT1": ("fired_heater", "피드(CH4) 예열기 ~1.5MW. 개질로와 별개."),
        "HX1": ("process_hx", ""), "HX2": ("process_hx", ""),
        "COOL1": ("utility_cooler", "->F1"),
        "F1": ("flash_vessel", "응축수 녹아웃"),
        "SEP1": ("psa", "H2 정제 PSA (->OUT_PROD, OUT_CO2)"),
        "C1": ("compressor", ""), "P1": ("pump", ""),
        "MIX1": ("none", ""),
    },
    "P06": {
        "R1": ("reformer", "개질 875C, Q>0 (R3_BURNER에 통합)"),
        "R2": ("wgs_reactor", "WGS 314C  kind=HT_WGS"),
        "R3_BURNER": ("fired_heater", "|Q_R3_BURNER|~-254MW *eta_furnace, 100MW 초과분 병렬"),
        "HEAT1": ("fired_heater", "보조가열 ~12MW  |Q_HEAT1|"),
        "HX1": ("process_hx", "->EXHAUST"), "HX2": ("process_hx", ""),
        "COOL1": ("utility_cooler", "->F1"),
        "F1": ("flash_vessel", "응축수 녹아웃"),
        "SEP1": ("psa", "H2 정제 PSA (->OUT_PROD, OUT_CO2)"),
        "C1": ("compressor", ""), "C2": ("compressor", ""), "C3": ("compressor", ""),
        "P1": ("pump", ""),
        "MIX1": ("none", ""), "MIX2": ("none", ""),
    },
    "P07": {
        "R1": ("reformer", "개질 840C, Q_R1~+22MW (HEAT1에 통합)"),
        "R2": ("wgs_reactor", "HT-WGS 376C  kind=HT_WGS"),
        "R3": ("wgs_reactor", "LT-WGS 229C  kind=LT_WGS"),
        "HEAT1": ("fired_heater", "개질 직전 가열기 ~10MW. 개질로와 별개."),
        "HEAT2": ("fired_heater", "피드 예열 ~21MW  |Q_HEAT2|"),
        "HX1": ("process_hx", ""), "HX2": ("process_hx", ""),
        "COOL1": ("utility_cooler", "->F1"),
        "F1": ("flash_vessel", "응축수 녹아웃 (->OUT_PROD 직결, PSA 없음)"),
        "C1": ("compressor", ""), "P1": ("pump", ""),
        "MIX1": ("none", ""),
    },
    "P08": {
        "R1": ("reformer", "개질 851C, Q_R1~+27MW (HEAT2에 통합)"),
        "R2": ("wgs_reactor", "HT-WGS 375C  kind=HT_WGS"),
        "R3": ("wgs_reactor", "LT-WGS 230C  kind=LT_WGS"),
        "HEAT1": ("fired_heater", "용수/피드 가열 ~30MW  |Q_HEAT1|"),
        "HEAT2": ("fired_heater", "개질 직전 가열기 ~4MW. 개질로와 별개."),
        "COOL1": ("utility_cooler", ""), "COOL2": ("utility_cooler", ""),
        "COOL3": ("utility_cooler", "->F1"),
        "F1": ("flash_vessel", "응축수 녹아웃"),
        "SEP1": ("psa", "H2 정제 PSA (->OUT_PROD, OUT_CO2)"),
        "C1": ("compressor", ""),
        "MIX1": ("none", ""),
    },
    "P09": {
        "R1": ("reformer", "개질 878C, Q>0 (BURNER에 통합)"),
        "R2": ("wgs_reactor", "WGS 281C  kind=HT_WGS (단일단, 중온)"),
        "BURNER": ("fired_heater", "|Q_Burner|*eta_furnace, 100MW 초과분 병렬"),
        "HX1": ("process_hx", ""), "HX2": ("process_hx", ""), "HX3": ("process_hx", ""),
        "HX4": ("process_hx", ""), "HX5": ("process_hx", "->OUT_PROD"),
        "HX6": ("process_hx", "->OUT_CO2"),
        "SEP1": ("psa", "H2 정제 PSA (->HX5->OUT_PROD, 테일->MIX2->BURNER)"),
        "C1": ("compressor", ""), "C2": ("compressor", ""), "C3": ("compressor", ""),
        "P1": ("pump", ""),
        "MIX1": ("none", ""), "MIX2": ("none", ""),
    },
    "P10": {
        "R1": ("reformer", "개질 871C, Q>0 (BURNER에 통합)"),
        "R2": ("wgs_reactor", "HT-WGS 385C  kind=HT_WGS"),
        "BURNER": ("fired_heater", "|Q_BURNER|*eta_furnace"),
        "HX1": ("process_hx", ""), "HX2": ("process_hx", ""), "HX3": ("process_hx", ""),
        "HX4": ("process_hx", "->OUT_CO2"), "HX5": ("process_hx", ""),
        "HX7": ("process_hx", ""), "HX8": ("process_hx", ""),
        "COOL1": ("utility_cooler", "->F1"),
        "F1": ("flash_vessel", "응축수 녹아웃"),
        "SEP1": ("psa", "H2 정제 PSA (->OUT_PROD, 테일->HX7)"),
        "C1": ("compressor", ""), "C2": ("compressor", ""),
        "P1": ("pump", ""),
        "MIX1": ("none", ""), "MIX2": ("none", ""),
    },
}

# WGS 반응기 노드 -> kind 인자
WGS_REACTOR_KIND = {
    ("P01", "R3"): "HT_WGS", ("P01", "R4"): "LT_WGS",
    ("P02", "R2"): "LT_WGS",
    ("P03", "R2"): "HT_WGS",
    ("P04", "R2"): "HT_WGS", ("P04", "R3"): "LT_WGS",
    ("P05", "R2"): "HT_WGS",
    ("P06", "R2"): "HT_WGS",
    ("P07", "R2"): "HT_WGS", ("P07", "R3"): "LT_WGS",
    ("P08", "R2"): "HT_WGS", ("P08", "R3"): "LT_WGS",
    ("P09", "R2"): "HT_WGS",
    ("P10", "R2"): "HT_WGS",
}


def cost_bearing_units(process_id):
    """해당 공정에서 비용이 잡히는 (노드, 카테고리) 리스트."""
    m = PROCESS_UNIT_MAP[process_id]
    return [(n, cat) for n, (cat, _note) in m.items()
            if cat not in ("none", "reformer")]


def costing_plan(process_id):
    """
    파이프라인이 바로 쓸 수 있는 형태로 두 매핑을 합친다.

    yields dict(node, category, sizing_fn, turton_key, streams, wgs_kind, note)
      - sizing_fn / turton_key : CATEGORY_TO_COSTING
      - streams : process_stream_map.PROCESS_STREAM_MAP[pid][node] (포트->스트림)
      - wgs_kind : wgs_reactor 노드면 "HT_WGS"/"LT_WGS", 아니면 None
    """
    from process_stream_map import PROCESS_STREAM_MAP
    pid_s = process_id if process_id in PROCESS_STREAM_MAP else "P%02d" % int(process_id[1:])
    umap = PROCESS_UNIT_MAP[process_id]
    smap = PROCESS_STREAM_MAP[pid_s]
    for node, (cat, note) in umap.items():
        if cat in ("none", "reformer"):
            continue
        fn, key = CATEGORY_TO_COSTING[cat]
        yield {
            "node": node, "category": cat, "sizing_fn": fn, "turton_key": key,
            "streams": smap.get(node),
            "wgs_kind": WGS_REACTOR_KIND.get((process_id, node)),
            "note": note,
        }


def summary_counts():
    """공정별 카테고리 개수 요약 (검토용)."""
    from collections import Counter
    out = {}
    for pid, m in PROCESS_UNIT_MAP.items():
        c = Counter(cat for cat, _ in m.values())
        out[pid] = {k: v for k, v in c.items() if k not in ("none",)}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(summary_counts(), indent=2, ensure_ascii=False))
