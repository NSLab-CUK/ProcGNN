"""
indicators.py
GNN 대리모델 스트림 예측값 -> 공정 지표 (GA_코드구축_프롬프트.md 모듈 2).

입력은 costing_pipeline 와 동일한 스트림 테이블(10D):
    {stream_name: {"T_c","P_bar","x"(7),"mdot_kgh"}}
그래프 구조(어느 스트림이 어느 유닛 입/출력인지)는 process_stream_map 에서 읽는다.

스트림 역할
    "Output 노드에 연결된 스트림"(계 경계를 넘는 출구)은
    process_stream_map.OUTPUT_STREAMS[pid] 로 제공된다. 그 중 무엇이
      product        : 수소/합성가스 제품
      atmospheric    : 대기 방출 (연소 배가스 flue_gas, PSA 테일가스 vent 등)
      captured_co2   : 포집되어 저장/판매되는 CO2
      condensate     : 응축수 / 수출 스팀
    인지는 "외부 스트림 역할매핑표"에서 판정한다 (이 모듈은 목록만 안다).
    roles 인자로 {stream_name: role} 을 넘기면 그대로 쓰고, 안 넘기면
    기본 휴리스틱(PROD=product, RE*/STEAM=condensate, 나머지=atmospheric)을 쓴다.

기준 상태: properties.py 와 동일 (298.15 K, 1 bar, 원소 기준 생성엔탈피 0점).
단위: 입력 T[degC], P[bar], mdot[kg/h]. 지표 단위는 각 함수 docstring 참조.
"""

from __future__ import annotations

import numpy as np

import properties as props
from process_stream_map import PROCESS_STREAM_MAP, OUTPUT_STREAMS

SPECIES = props.SPECIES                      # H2O,H2,CH4,CO2,CO,O2,N2
MW = props.MW_VEC                            # kg/kmol
I = {s: i for i, s in enumerate(SPECIES)}
LHV_H2 = props.LHV_H2                        # 120 MJ/kg
LHV_CH4 = props.LHV_CH4                      # 50 MJ/kg
LHV_CO_MJ_KG = 10.11

FEED_NAMES = {"CH4", "FUEL", "AIR", "WATER", "H2O", "STEAM"}
PROCESS_CH4_NAMES = {"CH4"}      # 개질기로 가는 원료 CH4 (버너 FUEL 제외)
FUEL_FEED_NAMES = {"FUEL"}       # 버너로 직접 가는 연료
PRODUCT_NAMES = {"PROD", "PRODUCT", "H2"}
CONDENSATE_NAMES = {"RE", "RESTEAM", "RESTEAM1"}


# ---------------------------------------------------------------------
# 스트림 -> 성분 몰유량 [kmol/h]
# ---------------------------------------------------------------------

def _mol_flows(s):
    """스트림 dict -> 성분별 몰유량 [kmol/h] (SPECIES 순서). 유량 0이면 0벡터."""
    m = s["mdot_kgh"]
    x = np.asarray(s["x"], dtype=float)
    if m <= 1e-8 or x.sum() <= 0:
        return np.zeros(7)
    m_mix = float(x @ MW)
    n_tot = m / m_mix
    return n_tot * x


def _pid(process_id):
    return process_id if process_id in PROCESS_STREAM_MAP else "P%02d" % int(process_id[1:])


def default_roles(process_id, streams):
    """역할매핑이 없을 때 쓰는 기본 휴리스틱. 외부 표가 있으면 그걸 우선."""
    pid = _pid(process_id)
    roles = {}
    for nm in OUTPUT_STREAMS.get(pid, []):
        u = nm.upper()
        if u in PRODUCT_NAMES:
            roles[nm] = "product"
        elif u in CONDENSATE_NAMES or "STEAM" in u or nm.upper().startswith("RE"):
            roles[nm] = "condensate"
        else:
            roles[nm] = "atmospheric"   # flue_gas / vent 로 보수적 가정 (포집이면 외부표로 덮어쓸 것)
    return roles


def _feed_totals(streams):
    """공급 스트림 성분 몰유량 합 [kmol/h] + 개별."""
    tot = np.zeros(7)
    per = {}
    for nm, s in streams.items():
        if nm.upper() in FEED_NAMES:
            f = _mol_flows(s)
            per[nm] = f
            tot += f
    return tot, per


def _product_stream(process_id, streams, roles):
    for nm, r in roles.items():
        if r == "product" and nm in streams:
            return nm, streams[nm]
    for c in PRODUCT_NAMES:
        if c in streams:
            return c, streams[c]
    return None, None


# ---------------------------------------------------------------------
# 생산성 / 수율
# ---------------------------------------------------------------------

def production_indicators(process_id, streams, roles=None):
    """
    h2_production   : 제품 스트림 H2 질량유량 [kg/h]
    ch4_conversion  : (공급 CH4 - 전 출구 CH4) / 공급 CH4   [-]
    h2_yield        : 제품 H2 몰유량 / 공급 CH4 몰유량       [mol/mol], 화학량론 상한 4
    carbon_efficiency : 제품으로 간 C 몰유량 / 공급 C 몰유량 [-]
    h2_purity       : 제품 H2 몰분율 [-]
    product_co      : 제품 CO 몰분율 [-]
    sc_ratio        : 공급 H2O 몰유량 / 공급 CH4 몰유량 [-]
    """
    pid = _pid(process_id)
    roles = roles or default_roles(pid, streams)
    feed_tot, _ = _feed_totals(streams)
    out_names = OUTPUT_STREAMS.get(pid, [])

    pname, pstream = _product_stream(pid, streams, roles)
    n_prod = _mol_flows(pstream) if pstream is not None else np.zeros(7)
    h2_kgh = n_prod[I["H2"]] * MW[I["H2"]]

    # 개질기로 가는 원료 CH4 (버너 FUEL 제외)
    proc_ch4 = sum(_mol_flows(streams[nm])[I["CH4"]]
                   for nm in streams if nm.upper() in PROCESS_CH4_NAMES)
    ch4_feed = feed_tot[I["CH4"]]  # 전체(원료+연료) CH4
    ch4_out = sum(_mol_flows(streams[nm])[I["CH4"]] for nm in out_names if nm in streams)
    ch4_conv = (ch4_feed - ch4_out) / ch4_feed if ch4_feed > 1e-9 else np.nan

    h2_yield = n_prod[I["H2"]] / proc_ch4 if proc_ch4 > 1e-9 else np.nan

    c_idx = [I["CH4"], I["CO2"], I["CO"]]
    c_feed = feed_tot[c_idx].sum()
    c_prod = n_prod[c_idx].sum()
    carbon_eff = c_prod / c_feed if c_feed > 1e-9 else np.nan  # 낮을수록 탄소가 제품 밖으로

    x = np.asarray(pstream["x"], dtype=float) if pstream is not None else np.zeros(7)
    h2o_feed = feed_tot[I["H2O"]]
    sc = h2o_feed / proc_ch4 if proc_ch4 > 1e-9 else np.nan

    return {
        "h2_production_kg_h": float(h2_kgh),
        "ch4_conversion": float(ch4_conv),
        "h2_yield_mol_mol": float(h2_yield),
        "carbon_efficiency": float(carbon_eff),
        "h2_purity": float(x[I["H2"]]),
        "product_co_frac": float(x[I["CO"]]),
        "sc_ratio": float(sc),
        "_product_stream": pname,
    }


# ---------------------------------------------------------------------
# 환경 (CO2 / CO 배출)
# ---------------------------------------------------------------------

def emission_indicators(process_id, streams, roles=None, h2_production_kg_h=None):
    """
    co2_emission    : role=="atmospheric" 인 Output 스트림들의 CO2 질량유량 합 [kg/h]
    co_emission     : 같은 스트림들의 CO 질량유량 합 [kg/h]
    captured_co2    : role=="captured_co2" 스트림들의 CO2 질량유량 합 [kg/h]
    specific_co2    : co2_emission / h2_production [kg CO2 / kg H2]
                      (무포집 SMR 문헌값 ~8~11; GA 다목적 최소화 대상)
    co2_capture_ratio : captured / (captured + emitted + 제품내 CO2)  [-]

    atmospheric / captured 판정은 roles 인자(외부 역할매핑표). 없으면 기본 휴리스틱.
    """
    pid = _pid(process_id)
    roles = roles or default_roles(pid, streams)

    co2_emit = co_emit = co2_capt = 0.0
    for nm, r in roles.items():
        if nm not in streams:
            continue
        n = _mol_flows(streams[nm])
        if r == "atmospheric":
            co2_emit += n[I["CO2"]] * MW[I["CO2"]]
            co_emit += n[I["CO"]] * MW[I["CO"]]
        elif r == "captured_co2":
            co2_capt += n[I["CO2"]] * MW[I["CO2"]]

    if h2_production_kg_h is None:
        h2_production_kg_h = production_indicators(pid, streams, roles)["h2_production_kg_h"]

    spec_co2 = co2_emit / h2_production_kg_h if h2_production_kg_h > 1e-6 else np.nan

    pname, pstream = _product_stream(pid, streams, roles)
    co2_in_prod = (_mol_flows(pstream)[I["CO2"]] * MW[I["CO2"]]) if pstream is not None else 0.0
    total_co2 = co2_emit + co2_capt + co2_in_prod
    capture_ratio = co2_capt / total_co2 if total_co2 > 1e-9 else np.nan

    return {
        "co2_emission_kg_h": float(co2_emit),
        "co_emission_kg_h": float(co_emit),
        "captured_co2_kg_h": float(co2_capt),
        "specific_co2_kg_per_kg_h2": float(spec_co2),
        "co2_capture_ratio": float(capture_ratio),
    }


# ---------------------------------------------------------------------
# 에너지
# ---------------------------------------------------------------------

def _h_kj_per_kg(streams):
    names = list(streams)
    T = np.array([streams[n]["T_c"] for n in names])
    X = np.array([streams[n]["x"] for n in names])
    M = np.array([streams[n]["mdot_kgh"] for n in names])
    ph = props.determine_phase(T, np.array([streams[n]["P_bar"] for n in names]), X, M)
    mmix = props.average_molar_mass(X)
    h = props.mixture_specific_enthalpy(T, X, ph, mmix)
    return {n: (0.0 if not np.isfinite(h[i]) else float(h[i])) for i, n in enumerate(names)}


def energy_indicators(process_id, streams, roles=None):
    """
    unit_duties_kW      : {유닛: Q [kW]}  (Q = Σ_out mdot*h - Σ_in mdot*h; +흡열/-발열)
    total_heat_input_kW : Σ max(0, Q)  (가열·흡열 반응 유닛)
    compressor_work_kW  : {압축기: W}  (엔탈피수지 기반 근사)
    total_shaft_work_kW : 압축기+펌프 - 터빈
    specific_energy_MJ_per_kg_h2 : (총 투입열 + 순 동력) / H2 질량유량
    thermal_efficiency  : (H2*LHV_H2) / (총 CH4 투입*LHV_CH4 + 순 동력)
    fuel_fraction       : 버너로 간 CH4 / 총 공급 CH4
    """
    pid = _pid(process_id)
    roles = roles or default_roles(pid, streams)
    smap = PROCESS_STREAM_MAP[pid]
    h = _h_kj_per_kg(streams)

    def hf(nm):  # 엔탈피 유량 [kW]
        nm = str(nm)
        s = streams.get(nm)
        return 0.0 if s is None else (s["mdot_kgh"] / 3600.0) * h.get(nm, 0.0)

    unit_duties = {}
    comp_work = {}
    for node, r in smap.items():
        k = r["kind"]
        if k == "hx":
            ins = [r["hot_in"], r["cold_in"]]
            outs = [r["hot_out"], r["cold_out"]]
        elif k in ("inout",):
            ins, outs = [r["feed"]], [r["prod"]]
        elif k in ("reactor_vl", "vessel"):
            ins = [r["feed"]]
            outs = [r.get("vapor_out"), r.get("liquid_out")]
        elif k == "burner":
            ins = list(r["feed"])
            outs = [r["prod"]]
        elif k == "sep":
            ins = [r["feed"]]
            outs = list(r["outlets"])
        else:
            continue
        q = sum(hf(o) for o in outs if o) - sum(hf(i) for i in ins if i)
        unit_duties[node] = q
        if node.startswith("C") and node[1:].isdigit():
            comp_work[node] = q  # 압축은 엔탈피 증가 ≈ 축일 (근사)

    total_heat_in = sum(max(0.0, q) for n, q in unit_duties.items()
                        if not (n.startswith(("C", "T", "P")) and n[1:].isdigit()))
    comp_pump_w = sum(q for n, q in unit_duties.items()
                      if (n.startswith(("C", "P")) and n[1:].isdigit()))
    turb_w = sum(-q for n, q in unit_duties.items()
                 if n.startswith("T") and n[1:].isdigit())
    net_shaft = comp_pump_w - turb_w

    feed_tot, feed_per = _feed_totals(streams)
    ch4_feed_kgh = feed_tot[I["CH4"]] * MW[I["CH4"]]
    # 버너로 간 CH4: burner 노드 feed 중 FUEL/CH4 계열
    fuel_ch4_kgh = 0.0
    for node, r in smap.items():
        if r["kind"] == "burner":
            for sname in r["feed"]:
                s = streams.get(str(sname))
                if s:
                    fuel_ch4_kgh += _mol_flows(s)[I["CH4"]] * MW[I["CH4"]]
    fuel_fraction = fuel_ch4_kgh / ch4_feed_kgh if ch4_feed_kgh > 1e-9 else 0.0

    prod = production_indicators(pid, streams, roles)
    h2_kgh = prod["h2_production_kg_h"]
    # 외부 투입 열 [kW] = 개질로 흡수듀티(costing_pipeline과 동일 정의) + HEAT 노드 듀티.
    from costing_pipeline import reformer_furnace_duty_kw
    furnace_kw = reformer_furnace_duty_kw(pid, streams)[0]
    heat_node_kw = sum(max(0.0, q) for n, q in unit_duties.items() if n.startswith("HEAT"))
    ext_heat_kw = furnace_kw + heat_node_kw
    ext_energy_kw = ext_heat_kw + max(0.0, net_shaft)
    spec_energy = (ext_energy_kw * 3.6 / h2_kgh) if h2_kgh > 1e-6 else np.nan  # kW/(kg/h)->MJ/kg

    # 열효율: 제품 H2 화학에너지 / (공급 CH4 화학에너지 + 외부 투입에너지)
    ch4_chem_kw = ch4_feed_kgh * LHV_CH4 * (1000.0 / 3600.0)
    denom = ch4_chem_kw + ext_energy_kw
    thermal_eff = (h2_kgh * LHV_H2 * (1000.0 / 3600.0)) / denom if denom > 1e-6 else np.nan

    return {
        "unit_duties_kW": {k: float(v) for k, v in unit_duties.items()},
        "total_heat_input_kW": float(total_heat_in),
        "compressor_work_kW": {k: float(v) for k, v in comp_work.items()},
        "total_shaft_work_kW": float(net_shaft),
        "specific_energy_MJ_per_kg_h2": float(spec_energy),
        "thermal_efficiency": float(thermal_eff),
        "fuel_fraction": float(fuel_fraction),
    }


# ---------------------------------------------------------------------
# 최소 접근온도
#   질량/원소/에너지 수지 잔차는 대리모델 학습단계에서 이미 강제되므로
#   여기서 별도 지표/제약으로 검사하지 않는다 (2026-09-03 결정).
# ---------------------------------------------------------------------

def min_approach_temp_K(process_id, streams):
    """전 process_hx 의 최소 접근온도 [K]. min(hot_in-cold_out, hot_out-cold_in) 의 최솟값.
    온도교차(음수)면 그 값이 그대로 최솟값이 되어 제약 위반으로 잡힌다."""
    pid = _pid(process_id)
    vals = []
    for node, r in PROCESS_STREAM_MAP[pid].items():
        if r["kind"] != "hx":
            continue
        try:
            hi = streams[str(r["hot_in"])]["T_c"]; ho = streams[str(r["hot_out"])]["T_c"]
            ci = streams[str(r["cold_in"])]["T_c"]; co = streams[str(r["cold_out"])]["T_c"]
        except KeyError:
            continue
        vals.append(min(hi - co, ho - ci))
    return float(min(vals)) if vals else np.nan


# ---------------------------------------------------------------------
# 통합
# ---------------------------------------------------------------------

def all_indicators(process_id, streams, roles=None):
    """모듈 2 지표 전체를 한 번에. roles 는 외부 스트림 역할매핑표(선택)."""
    pid = _pid(process_id)
    roles = roles or default_roles(pid, streams)
    prod = production_indicators(pid, streams, roles)
    emis = emission_indicators(pid, streams, roles,
                               h2_production_kg_h=prod["h2_production_kg_h"])
    ener = energy_indicators(pid, streams, roles)
    return {**prod, **emis, **ener,
            "min_approach_temp_K": min_approach_temp_K(pid, streams),
            "roles_used": roles,
            "output_streams": OUTPUT_STREAMS.get(pid, [])}
