"""
objective.py
GA(다목적 최적화) 인터페이스. GA_코드구축_프롬프트.md 모듈 4.

    evaluate(streams, process_id, ...) -> {
        'objectives'  : {lcoh_usd_per_kg, specific_co2_kg_per_kg_h2}   # 둘 다 최소화
        'indicators'  : 공정지표 + 경제 요약
        'constraints' : {name: {'ok': bool, 'violation': float>=0}}
        'feasible'    : 모든 제약 만족 여부
    }

streams 스키마는 costing_pipeline / indicators 와 동일한 10D:
    {stream_name: {"T_c","P_bar","x"(7 몰분율,합1),"mdot_kgh"}}

GNN 대리모델 추론은 이 모듈 밖에서 한다(의존성 주입). GA 루프:
    운전조건 -> (대리모델) -> streams dict -> evaluate(streams, pid) -> objectives

roles: 외부 스트림 역할매핑표 {stream_name: 'product'|'atmospheric'|'captured_co2'|'condensate'}.
    안 넘기면 indicators.default_roles 휴리스틱 사용(Output 노드 스트림 중
    PROD=product, RE*/STEAM=condensate, 나머지=atmospheric).

속도: Aspen COM 호출 없음. 한 번 평가에 수 ms (properties 재구성 + ~15 유닛 + Turton).

제약(constraints)
    각 항목 {'ok': bool, 'violation': float>=0}. violation 은 GA 페널티로 바로 쓸 수 있다
    (만족 시 0). limits 인자로 한계값을 덮어쓴다. None 인 한계는 그 제약을 끈다.
    수지 잔차(질량/원소/에너지)는 대리모델 학습단계에서 이미 강제되므로 여기서
    제약으로 검사하지 않는다.
"""

from __future__ import annotations

import numpy as np

import costing_pipeline as cost
import indicators as ind
from process_stream_map import PROCESS_STREAM_MAP
from process_unit_map import PROCESS_UNIT_MAP

# H2 정제 공정(제품 H2 ~100%) — 스트림역할매핑_분석보고 §2.1
PURIFIED_PROCESSES = {"P03", "P04", "P06", "P08", "P09", "P10"}
SYNGAS_PROCESSES = {"P01", "P02", "P05", "P07"}

# 제약 기본 한계값. limits 인자로 공정별·논문별 조정. None = 그 제약 미적용.
DEFAULT_LIMITS = {
    "sc_ratio_min": 2.5,                  # coking 방지 (Ni 촉매, 문헌 통상 2.5~3.5)
    "reformer_outlet_temp_max_C": 950.0,  # 개질관 야금 한계 (공정가스 출구 기준)
    "h2_purity_min": None,                # None -> PURIFIED_PROCESSES 는 0.99 자동 적용
    "product_co_frac_max": None,          # H2 용도별(연료전지 <1e-5). 기본 미적용
    "min_approach_temp_min_K": 5.0,       # 열교환기 핀치
    "max_stream_temp_C": 1300.0,          # 물리적 상한 sanity
}


def _reformer_outlet_temp(pid, streams):
    temps = []
    for n, (c, _) in PROCESS_UNIT_MAP[pid].items():
        if c != "reformer":
            continue
        r = PROCESS_STREAM_MAP[pid].get(n, {})
        out_s = r.get("prod") or r.get("vapor_out")
        if out_s and str(out_s) in streams:
            temps.append(streams[str(out_s)]["T_c"])
    return max(temps) if temps else np.nan


def _check(name, ok, violation):
    return name, {"ok": bool(ok), "violation": float(max(0.0, violation))}


def _pid(process_id):
    return process_id if process_id in PROCESS_STREAM_MAP else "P%02d" % int(process_id[1:])


def evaluate(streams, process_id, prices=None, roles=None, limits=None,
             plant_life_years=cost.PLANT_LIFE_YEARS, heater_fuel_mode="NG",
             opex_method="itemized"):
    """한 공정·한 케이스(스트림 테이블) -> GA 목적함수 dict.

    opex_method : "itemized" (기본, Peters&Timmerhaus 항목별 = 유지보수 4%FCI +
                  세금보험 2%FCI + 인건비 + 간접비 60%, 변동비 별도)
                  또는 "turton_com" (Turton COM_d, sensitivity 비교용).
    """
    prices = prices or cost.REFERENCE_PRICES
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    pid = _pid(process_id)

    roles = roles or ind.default_roles(pid, streams)
    metrics = ind.all_indicators(pid, streams, roles)
    econ = cost.evaluate_process(pid, streams, prices,
                                 plant_life_years=plant_life_years,
                                 heater_fuel_mode=heater_fuel_mode,
                                 opex_method=opex_method)

    objectives = {
        "lcoh_usd_per_kg": econ["lcoh_usd_per_kg"],
        "specific_co2_kg_per_kg_h2": metrics["specific_co2_kg_per_kg_h2"],
    }

    # --- 제약 ---
    cons = {}

    # 입력 sanity (대리모델이 대체로 보장하지만 재확인)
    bad_flow = any(s["mdot_kgh"] < -1e-6 for s in streams.values())
    bad_frac = any(min(s["x"]) < -1e-3 or abs(sum(s["x"]) - 1.0) > 5e-2
                   for s in streams.values() if s["mdot_kgh"] > 1e-6)
    hot = max((s["T_c"] for s in streams.values()), default=np.nan)
    cons.update([_check("stream_sanity", (not bad_flow) and (not bad_frac)
                        and (not np.isfinite(hot) or hot <= limits["max_stream_temp_C"]),
                        1.0 if (bad_flow or bad_frac) else
                        max(0.0, (hot - limits["max_stream_temp_C"]) if np.isfinite(hot) else 0.0))])

    sc = metrics["sc_ratio"]
    if limits["sc_ratio_min"] is not None:
        cons.update([_check("sc_ratio_min", np.isfinite(sc) and sc >= limits["sc_ratio_min"],
                            (limits["sc_ratio_min"] - sc) if np.isfinite(sc) else 1e3)])

    t_ref = _reformer_outlet_temp(pid, streams)
    if limits["reformer_outlet_temp_max_C"] is not None and np.isfinite(t_ref):
        cons.update([_check("reformer_outlet_temp_max",
                            t_ref <= limits["reformer_outlet_temp_max_C"],
                            t_ref - limits["reformer_outlet_temp_max_C"])])

    purity_min = limits["h2_purity_min"]
    if purity_min is None and pid in PURIFIED_PROCESSES:
        purity_min = 0.99
    if purity_min is not None:
        p = metrics["h2_purity"]
        cons.update([_check("h2_purity_min", p >= purity_min, purity_min - p)])

    if limits["product_co_frac_max"] is not None:
        co = metrics["product_co_frac"]
        cons.update([_check("product_co_max", co <= limits["product_co_frac_max"],
                            co - limits["product_co_frac_max"])])

    mat = metrics["min_approach_temp_K"]
    if limits["min_approach_temp_min_K"] is not None and np.isfinite(mat):
        cons.update([_check("min_approach_temp", mat >= limits["min_approach_temp_min_K"],
                            limits["min_approach_temp_min_K"] - mat)])

    h2 = metrics["h2_production_kg_h"]
    cons.update([_check("h2_production_positive", h2 > 1.0,
                        1.0 - h2 if h2 <= 1.0 else 0.0)])

    feasible = all(v["ok"] for v in cons.values())

    return {
        "process_id": pid,
        "objectives": objectives,
        "indicators": {
            **metrics,
            "fci_usd": econ["fci_usd"], "tci_usd": econ["tci_usd"],
            "opex_usd_per_year": econ["opex_usd_per_year"],
            "lcoh_gross_usd_per_kg": econ["lcoh_gross_usd_per_kg"],
            "coproduct_credit_usd_per_year": econ["coproduct_credit_usd_per_year"],
            "c_fuel_fired_usd_per_year": econ["c_fuel_fired_usd_per_year"],
            "c_cooling_water_usd_per_year": econ["c_cooling_water_usd_per_year"],
            "furnace_duty_kW": econ["furnace_duty_kW"],
            "reformer_outlet_temp_C": t_ref,
        },
        "constraints": cons,
        "feasible": feasible,
    }
