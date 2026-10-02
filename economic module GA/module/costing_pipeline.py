"""
costing_pipeline.py
GNN 대리모델의 스트림 예측값 -> 경제성 지표(장비비, FCI, OPEX, LCOH) 변환 드라이버.

이 모듈은 "스트림 테이블 하나"를 받아 한 공정의 경제성 지표를 계산한다.
스트림 테이블의 스키마는 GNN 대리모델 출력과 동일한 10D:
    stream_name -> {"T_c": T[degC], "P_bar": P[bar],
                    "x": [H2O,H2,CH4,CO2,CO,O2,N2 몰분율(합 1)],
                    "mdot_kgh": 질량유량[kg/h]}
GNN 예측본이 아직 없으므로, 검증·시연은 Aspen Process_Streams CSV(동일 스키마,
대리모델이 재현하려는 ground truth)를 넣어서 한다.

===== GNN 예측본 연결 방법 (상대측이 수행) =====
  1. GNN이 예측한 (공정 pid, 케이스별) 스트림값을 위 스키마 dict로 만든다.
     스트림 이름은 process_stream_map.PROCESS_STREAM_MAP[pid] 에서 참조하는 이름과
     동일해야 한다 (해당 공정 Process_Streams CSV의 Stream_Name과 같음).
  2. evaluate_process(pid, streams, prices) 호출.
  3. 반환 dict의 "lcoh_usd_per_kg"(부산물 크레딧 반영)를 GA/RL 목적함수로 사용.
     (LCOH는 kg H2당 값이라 대형화 편향이 없다. 절대이익이 필요하면 NPV를 별도로.)
  블록->스트림 매핑, 유닛->장비 매핑, 사이징식, Turton 계수는 전부 이 저장소 안에
  고정돼 있으므로 상대측은 "스트림 dict 만들기 + 이 함수 호출" 만 하면 된다.
===============================================

파이프라인
    streams(10D)
      -> properties.reconstruct_stream_properties  (스트림별 rho, V_dot, h)
      -> process_unit_map.costing_plan(pid)        (노드 -> 장비카테고리 + 스트림)
      -> equipment_sizing.*                        (카테고리별 용량변수: kW / m2 / m3)
      -> turton_costing                            (구매비 -> 베어모듈 -> 그래스루트 = FCI)
      -> economics                                 (TCI, COM_d, LCOH)

가격 파라미터(전력·NG·인건비 등)는 기본값을 심지 않는다. prices dict로 명시적으로
넘겨야 하며, 참고용 예시는 economics.REFERENCE_PRICE_EXAMPLES / 이 파일 하단
REFERENCE_PRICES 에 있다 (그대로 쓰지 말고 최신 시세로 교체).

개질로 / 가열유닛 처리
    개질기(R1, P1은 R2도)와 버너는 "개질로(reformer furnace) 1기"로 통합 코스팅한다.
      furnace_duty = max( Σ개질 흡열,  eta_furnace * 버너 연소열 )
      -> fired_heater_nonreactive (100 MW 초과 시 병렬 셀), 개질기·버너 개별계상 없음.
    이 10개 공정의 버너는 전부 SMR 개질로 자체이므로(독립 보일러 없음) 이렇게 묶는다.
    HEAT 노드(피드 예열기 등)는 개질로와 별개로 자기 듀티(h_out-h_in)만큼 코스팅한다
    -> RL에서 예열기를 넣고/빼는 차이가 목적함수에 그대로 반영된다.
    진단: reformer_endothermic_kW / burner_combustion_kW / furnace_duty_kW 반환.

플래시 응축기
    flash(F1) 블록이 분리뿐 아니라 냉각(응축)까지 하면(블록 엔탈피수지로 판정,
    >100 kW) 응축기 HX 1기를 별도로 추가한다. 앞단에 COOL 노드가 있어 F1이 단열
    분리만 하는 공정(P05~P08,P10)은 COOL이 냉각기, F1이 드럼으로 각각 잡힌다.

Turton 유효구간 밖 처리
    용량이 상관식 범위를 벗어나면 외삽하지 않고 turton_costing.purchased_cost_ranged
    로 (상한 초과 -> 동일 장비 N기 병렬 / 하한 미만 -> 하한값 클램프) 처리한다.
    조정된 유닛은 range_adjusted_units 로 반환.

외부 공급 화력의 연료비 (heater_fuel_mode)
    개질로/HEAT 노드에 FUEL 스트림이 없는 공정(버너 없는 P05/P07/P08)은 그 열을
    만드는 연료가 어느 스트림에도 없다 -> (개질로 듀티 + HEAT 듀티)/eta_furnace 를
    NG 연료비로 c_rm 에 추가한다 (heater_fuel_mode="NG", 기본).
    "electric" 이면 그 듀티를 전력단가로 c_ut 에 넣는다 (e-SMR 옵션).
    버너 있는 공정은 연소연료가 이미 FUEL 스트림으로 c_rm 에 있으므로 추가 없음.

CO2: 10개 공정에 CO2 포집 전용 장비는 없다(OUT_CO2는 PSA 테일가스/연소 배가스).
    별도 계상 항목 없음. 탄소세를 넣으려면 economics.cost_of_manufacturing 의
    c_wt 인자에 (co2_emission * 탄소단가) 를 넘기면 된다 (현재 0).

플랜트 수명·건설기간 가정 (LCOH 스케줄):
    건설 1년(H2=0, CAPEX 전액) + 운전 PLANT_LIFE_YEARS년(H2·OPEX·크레딧 일정).
    r = economics.DISCOUNT_RATE (5%). PLANT_LIFE_YEARS 기본 20 (SMR TEA 통상값).

LCOH 정의 (부산물 크레딧 반영)
    LCOH = [ NPV(CAPEX) + NPV(OPEX - 부산물수익) ] / NPV(H2 생산량[kg])

    H2 생산량 = 제품 스트림(PROD)의 H2 질량유량 * 8,000 h/yr
               (합성가스 공정도 H2 질량만 분모에 넣어 10개 공정을 같은 잣대로 비교)
    부산물 수익(연간) =
      (a) 순 전력수출  : max(0, W_turbine - W_comp - W_pump) * h * elec_price
                         (순 소비면 크레딧 0, 대신 C_UT에 유틸리티비로 계상)
      (b) 합성가스 연료가치 : 제품 스트림의 비-H2 가연분(CH4, CO) 질량 * LHV
                         * (NG_price / NG_LHV)   -- 즉 NG 대비 에너지 등가 가격
      (c) 수출 스팀    : RESTEAM 등 * (스팀유효엔탈피/보일러효율) * (NG_price/NG_LHV)
    lcoh_gross (크레딧 없이) 도 함께 반환한다.

    LCOH는 kg H2당 값(intensive)이므로 규모 자체가 목적함수를 지배하지 않는다
    (절대이익 NPV와 달리 대형화 편향이 없음).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import properties as props
import equipment_sizing as sizing
import turton_costing as tc
import economics as ec
from process_unit_map import costing_plan, PROCESS_UNIT_MAP
from process_stream_map import PROCESS_STREAM_MAP  # noqa

SPECIES = props.SPECIES  # ["H2O","H2","CH4","CO2","CO","O2","N2"]
PLANT_LIFE_YEARS = 20
CONSTRUCTION_YEARS = 1
COOLING_WATER_IN_C = 25.0
COOLING_WATER_OUT_C = 40.0

MW = props.MW_VEC  # kg/kmol, SPECIES order

# 저위발열량 [MJ/kg] (부산물 연료가치 산정용). H2/CH4는 properties.py와 동일 출처.
LHV_MJ_PER_KG = {"H2": props.LHV_H2, "CH4": props.LHV_CH4, "CO": 10.11}
# 천연가스 저위발열량 [MJ/kg] (연료가치를 NG 단가로 환산할 때의 분모).
#   순 메탄 기준 50 MJ/kg, 실제 파이프라인 가스는 ~47~49. 48 채택.
NATURAL_GAS_LHV_MJ_PER_KG = 48.0
# 수출 스팀 크레딧: 보일러 대체 회피비용 = (스팀 엔탈피 - 급수 엔탈피)/보일러효율
#   * NG 단가[$/MJ]. 스팀 유효 엔탈피 ~2.6 MJ/kg (과열 감안), 보일러효율 0.85.
STEAM_EFFECTIVE_ENTHALPY_MJ_PER_KG = 2.6
BOILER_EFFICIENCY = 0.85


# =====================================================================
# 1. 스트림 테이블 로딩 / 물성 재구성
# =====================================================================

def stream_table_from_aspen_csv(streams_csv_path, case_id):
    """
    Aspen n.Process_Streams*.csv 에서 한 케이스(case_id)를 뽑아 표준 스트림 테이블로.
    반환: dict {stream_name: dict(T_c, P_bar, x(list7), mdot_kgh)}
    """
    df = pd.read_csv(streams_csv_path, dtype={"Stream_Name": str})
    sub = df[df["ID"] == case_id]
    if len(sub) == 0:
        sub = df[df["ID"].astype(float) == float(case_id)]
    out = {}
    fr_cols = [f"Frac_{s}" for s in SPECIES]
    for _, r in sub.iterrows():
        out[str(r["Stream_Name"])] = {
            "T_c": float(r["Temp"]), "P_bar": float(r["Pres"]),
            "x": [float(r[c]) for c in fr_cols],
            "mdot_kgh": float(r["Mass_Flow"]),
        }
    return out


def _reconstruct(streams):
    """스트림 테이블 -> {name: props dict} (rho, V_dot, h, phase ...)."""
    names = list(streams)
    T = np.array([streams[n]["T_c"] for n in names])
    P = np.array([streams[n]["P_bar"] for n in names])
    X = np.array([streams[n]["x"] for n in names])
    M = np.array([streams[n]["mdot_kgh"] for n in names])
    rp = props.reconstruct_stream_properties(T, P, X, M)
    per = {}
    for i, n in enumerate(names):
        per[n] = {
            "T_c": T[i], "P_bar": P[i], "x": X[i], "mdot_kgh": M[i],
            "rho": rp["density_kg_m3"][i],
            "vdot_m3s": rp["vol_flow_m3s"][i],
            "h_kj_kg": rp["enthalpy_kj_kg"][i],
            "m_mix": rp["m_mix_kg_per_kmol"][i],
            "phase": rp["phase"][i],
        }
    return per


def _H_flow_kW(s):
    """스트림 엔탈피 유량 [kW] = mdot[kg/s] * h[kJ/kg]."""
    return (s["mdot_kgh"] / 3600.0) * s["h_kj_kg"]


def _mass_frac_h2(s):
    xh2 = s["x"][SPECIES.index("H2")]
    return xh2 * MW[SPECIES.index("H2")] / max(s["m_mix"], 1e-9)


def _mol_flows_kmolh(s):
    """스트림 -> 성분 몰유량 [kmol/h] (SPECIES 순서). 유량 0이면 0벡터."""
    x = np.asarray(s["x"], dtype=float)
    m = s["mdot_kgh"]
    if m <= 1e-8 or x.sum() <= 0:
        return np.zeros(7)
    return (m / float(x @ MW)) * x


def reformer_furnace_duty_kw(process_id, streams):
    """
    개질로 흡수 듀티 [kW] = max( Σ개질 흡열,  eta_furnace * 버너 연소열 ).
    indicators.py 등에서 재사용. streams 는 10D dict.
    Returns (furnace_duty_kw, reformer_endo_kw, burner_combustion_kw, has_burner).
    """
    pid = process_id if process_id in PROCESS_STREAM_MAP else "P%02d" % int(process_id[1:])
    P = _reconstruct(streams)
    smap_all = PROCESS_STREAM_MAP.get(pid, {})
    endo = 0.0
    for n, (c, _n) in PROCESS_UNIT_MAP[pid].items():
        if c != "reformer":
            continue
        rs = smap_all.get(n)
        if not rs:
            continue
        try:
            f = P[str(rs["feed"])]
            o = P[str(rs.get("prod") or rs.get("vapor_out"))]
            endo += max(0.0, _H_flow_kW(o) - _H_flow_kW(f))
        except (KeyError, TypeError):
            pass
    burner_nodes = [n for n, (c, _) in PROCESS_UNIT_MAP[pid].items()
                    if c == "fired_heater" and smap_all.get(n, {}).get("kind") == "burner"]
    comb = 0.0
    for bn in burner_nodes:
        bs = smap_all[bn]
        try:
            comb += abs(sum(_H_flow_kW(P[str(s)]) for s in bs["feed"])
                        - _H_flow_kW(P[str(bs["prod"])]))
        except (KeyError, TypeError):
            pass
    return (max(endo, sizing.FURNACE_THERMAL_EFFICIENCY * comb),
            endo, comb, bool(burner_nodes))


# =====================================================================
# 2. 노드별 용량변수 계산
# =====================================================================

def _capacity(node, cat, smap, wgs_kind, P, extras):
    """
    한 노드의 (turton_key, capacity_value, aux) 반환.
    P: {stream_name: props}, extras: 부수 비용 라인 누적 dict.
    capacity_value 단위는 카테고리별 (kW / m2 / m3).
    """
    g = lambda name: P[str(name)]

    if cat == "compressor":
        s_in, s_out = g(smap["feed"]), g(smap["prod"])
        k = props.mixture_k_ratio(s_in["T_c"], s_in["x"])
        if not np.isfinite(k):
            k = 1.4
        w = sizing.compressor_isentropic_power_kw(
            s_in["vdot_m3s"], s_in["P_bar"], s_out["P_bar"],
            k_ratio=k, isentropic_efficiency=sizing.ASPEN_COMPRESSOR_ISENTROPIC_EFF)
        return "compressor_centrifugal", float(w), {"W_kW": float(w)}

    if cat == "turbine":
        s_in, s_out = g(smap["feed"]), g(smap["prod"])
        k = props.mixture_k_ratio(s_in["T_c"], s_in["x"])
        if not np.isfinite(k):
            k = 1.4
        w = sizing.turbine_isentropic_power_kw(
            s_in["vdot_m3s"], s_in["P_bar"], s_out["P_bar"],
            k_ratio=k, isentropic_efficiency=sizing.ASPEN_TURBINE_ISENTROPIC_EFF)
        return "turbine_expander", float(w), {"W_kW": float(w)}

    if cat == "pump":
        s_in, s_out = g(smap["feed"]), g(smap["prod"])
        eta = sizing.PUMP_ISENTROPIC_EFFICIENCY_BY_BLOCK.get(node, 0.75)
        w = sizing.pump_power_kw(s_in["vdot_m3s"], s_out["P_bar"] - s_in["P_bar"],
                                 pump_efficiency=eta)
        return "pump_centrifugal", float(w), {"W_kW": float(w)}

    if cat in ("process_hx", "utility_cooler"):
        if cat == "process_hx":
            ci, co = g(smap["cold_in"]), g(smap["cold_out"])
            hi, ho = g(smap["hot_in"]), g(smap["hot_out"])
            q_kw = abs(_H_flow_kW(co) - _H_flow_kW(ci))
            lmtd = sizing.log_mean_temperature_difference(
                hi["T_c"], ho["T_c"], ci["T_c"], co["T_c"])
        else:  # 1-sided cooler: process stream loses heat to cooling water
            pi, po = g(smap["feed"]), g(smap["prod"])
            q_kw = abs(_H_flow_kW(po) - _H_flow_kW(pi))
            lmtd = sizing.log_mean_temperature_difference(
                pi["T_c"], po["T_c"], COOLING_WATER_IN_C, COOLING_WATER_OUT_C)
        lmtd = float(np.atleast_1d(lmtd)[0])
        if not np.isfinite(lmtd) or lmtd < 1.0:
            lmtd = 1.0
        area = sizing.heat_exchanger_area_m2(q_kw, sizing.ASPEN_HEATX_U_W_M2K, lmtd)
        return "heat_exchanger_floating_head", float(np.atleast_1d(area)[0]), {
            "Q_kW": q_kw, "LMTD_K": lmtd}

    if cat == "fired_heater":
        # 여기 오는 것은 HEAT 노드(피드 예열기 등)뿐이다. burner 노드는
        # evaluate_process에서 개질기와 합쳐 "개질로 1기"로 처리하므로 루프에서 제외된다.
        # 각 HEAT는 "그 유닛이 실제로 공급하는 열"만큼만 코스팅 (heater 유무 차이가
        # RL 목적함수에 그대로 반영되도록).
        s_in, s_out = g(smap["feed"]), g(smap["prod"])
        q_duty_kw = abs(_H_flow_kW(s_out) - _H_flow_kW(s_in))
        return "fired_heater_nonreactive", q_duty_kw, {"duty_kW": q_duty_kw}

    if cat == "flash_vessel":
        fe = g(smap["feed"])
        vo = g(smap["vapor_out"])
        lo = g(smap["liquid_out"]) if smap.get("liquid_out") else None
        p_barg = max(vo["P_bar"] - 1.01325, 0.0)
        v = sizing.vertical_vessel_sizing(
            np.array([vo["vdot_m3s"]]),
            np.array([lo["rho"] if lo is not None else 997.0]),
            np.array([vo["rho"]]),
            pressure_barg=p_barg,
            liquid_vol_flow_m3s=np.array([lo["vdot_m3s"] if lo is not None else 0.0]))
        vol = float(np.atleast_1d(v["volume_m3"])[0])
        # 이 flash 블록 자체가 냉각(응축)까지 하는 경우(예 P03 F1, Q~-11.5MW):
        # 블록 엔탈피수지로 응축기 듀티를 구해 aux로 넘긴다 -> 루프에서 응축기 HX 1기 추가.
        h_out = _H_flow_kW(vo) + (_H_flow_kW(lo) if lo is not None else 0.0)
        condenser_duty_kw = _H_flow_kW(fe) - h_out          # +면 냉각(열 제거)
        return "process_vessel_vertical", vol, {
            "D_m": float(np.atleast_1d(v["diameter_m"])[0]),
            "L_m": float(np.atleast_1d(v["length_m"])[0]),
            "pressure_barg": p_barg,
            "condenser_duty_kw": condenser_duty_kw,
            "feed_T_c": fe["T_c"], "vap_T_c": vo["T_c"]}

    if cat == "psa":
        feed = g(smap["feed"])
        r = sizing.psa_bed_sizing(np.array([feed["vdot_m3s"]]))
        n_beds = float(np.atleast_1d(r["n_beds"])[0])
        vol_bed = float(np.atleast_1d(r["volume_per_bed_m3"])[0])
        ads_kg = float(np.atleast_1d(
            sizing.psa_adsorbent_inventory_kg(r["total_bed_volume_m3"]))[0])
        ads_cost = float(np.atleast_1d(
            sizing.psa_adsorbent_initial_cost_usd(ads_kg))[0])
        extras["adsorbent_initial_usd"] += ads_cost
        extras["adsorbent_replace_usd_yr"] += float(np.atleast_1d(
            sizing.psa_adsorbent_annual_replacement_cost_usd(ads_cost))[0])
        # n_beds개의 동일 용기 -> capacity는 총 흡착탑 부피, n_intrinsic=n_beds
        return "process_vessel_vertical", vol_bed * n_beds, {
            "n_intrinsic": n_beds, "D_m": float(np.atleast_1d(r["diameter_m"])[0]),
            "adsorbent_t": ads_kg / 1000.0}

    if cat == "wgs_reactor":
        s_in = g(smap["feed"])
        r = sizing.wgs_reactor_sizing(
            np.array([s_in["vdot_m3s"]]), s_in["T_c"], s_in["P_bar"], wgs_kind)
        v_vessel = float(np.atleast_1d(r["v_vessel_m3"])[0])
        cat_kg = float(np.atleast_1d(r["catalyst_mass_kg"])[0])
        price = (sizing.CATALYST_PRICE_HT_WGS_USD_PER_KG if wgs_kind == "HT_WGS"
                 else sizing.CATALYST_PRICE_LT_WGS_USD_PER_KG)
        c0 = float(np.atleast_1d(sizing.catalyst_initial_cost_usd(cat_kg, price))[0])
        extras["catalyst_initial_usd"] += c0
        extras["catalyst_replace_usd_yr"] += float(np.atleast_1d(
            sizing.catalyst_annual_replacement_cost_usd(c0))[0])
        return "process_vessel_vertical", v_vessel, {
            "catalyst_t": cat_kg / 1000.0, "wgs_kind": wgs_kind}

    raise ValueError(f"unknown category {cat}")


# =====================================================================
# 3. 공정 1개 평가
# =====================================================================

def evaluate_process(process_id, streams, prices, cepci_current=tc.CEPCI_CURRENT,
                      plant_life_years=PLANT_LIFE_YEARS, heater_fuel_mode="NG",
                      condensate_streams=None, opex_method="itemized"):
    """
    한 공정, 한 스트림 테이블(=한 케이스)의 경제성 지표를 계산한다.

    streams : dict {name: {T_c, P_bar, x[7], mdot_kgh}}  (또는 DataFrame는 호출측에서
              stream_table_from_aspen_csv 로 변환)
    prices  : dict {
        "electricity_usd_per_kwh", "natural_gas_usd_per_kg",
        "process_water_usd_per_kg", "operating_labor_usd_per_hr"
      }

    Returns dict(units=[...], capex=..., opex=..., lcoh=..., ...)
    """
    pid = process_id if process_id in PROCESS_UNIT_MAP else "P%02d" % int(process_id[1:])
    pid_s = pid if pid in PROCESS_STREAM_MAP else "P%02d" % int(pid[1:])
    smap_all = PROCESS_STREAM_MAP.get(pid_s, {})
    P = _reconstruct(streams)

    extras = {"adsorbent_initial_usd": 0.0, "adsorbent_replace_usd_yr": 0.0,
              "catalyst_initial_usd": 0.0, "catalyst_replace_usd_yr": 0.0}

    unit_rows = []
    bm_2001_total = [0.0]  # 리스트로 감싸 클로저에서 수정
    shaft_power_kw = {"compressor": 0.0, "pump": 0.0, "turbine": 0.0}

    def add_unit(node, cat, key, capv, aux):
        """용량변수 -> Turton 구간처리 -> 베어모듈 -> unit_rows / bm_2001_total 누적."""
        n_intrinsic = aux.get("n_intrinsic", 1.0)
        cp0_pu, n_rng, cap_pu, status = tc.purchased_cost_ranged(key, capv / n_intrinsic)
        cp0_pu, n_rng, cap_pu = float(cp0_pu), float(n_rng), float(cap_pu)
        n_units = n_intrinsic * n_rng
        if key in tc.TURTON_FBM_DIRECT:
            cbm_pu = float(np.atleast_1d(tc.bare_module_cost(key, np.array([cp0_pu])))[0])
        else:
            pbarg = aux.get("pressure_barg", None)
            if key.startswith("process_vessel") or key.startswith("reactor"):
                dm = (4.0 * cap_pu / (3.0 * np.pi)) ** (1.0 / 3.0)  # L/D=3 가정
            else:
                dm = aux.get("D_m", None)
            cbm_pu = float(np.atleast_1d(tc.bare_module_cost(
                key, np.array([cp0_pu]), pressure_barg=pbarg, diameter_m=dm))[0])
        cbm = cbm_pu * n_units
        bm_2001_total[0] += cbm
        if cat in shaft_power_kw:
            shaft_power_kw[cat] += aux.get("W_kW", 0.0)
        unit_rows.append({"node": node, "category": cat, "turton_key": key,
                          "capacity": capv, "n_units": n_units, "capacity_per_unit": cap_pu,
                          "cp0_2001usd": cp0_pu, "cbm_2001usd": cbm, "range_status": status,
                          **{k: v for k, v in aux.items() if k != "n_intrinsic"}})

    # --- 개질로(reformer furnace) 1기: 개질기 + 버너 통합 ---
    #   용량 = max( Σ개질 흡열, eta_furnace * 버너 연소열 )
    #   이 10개 공정의 버너는 전부 SMR 개질로 자체이므로 별도 계상하지 않는다.
    reformer_endo_kw = 0.0
    for n, (c, _note) in PROCESS_UNIT_MAP[pid].items():
        if c != "reformer":
            continue
        rs = smap_all.get(n)
        if not rs:
            continue
        try:
            f = P[str(rs["feed"])]
            o = P[str(rs.get("prod") or rs.get("vapor_out"))]
            reformer_endo_kw += max(0.0, _H_flow_kW(o) - _H_flow_kW(f))
        except (KeyError, TypeError):
            pass
    burner_comb_kw = 0.0
    burner_nodes = [n for n, (c, _) in PROCESS_UNIT_MAP[pid].items()
                    if c == "fired_heater" and smap_all.get(n, {}).get("kind") == "burner"]
    for bn in burner_nodes:
        bs = smap_all[bn]
        try:
            h_in = sum(_H_flow_kW(P[str(s)]) for s in bs["feed"])
            h_out = _H_flow_kW(P[str(bs["prod"])])
            burner_comb_kw += abs(h_in - h_out)
        except (KeyError, TypeError):
            pass
    furnace_duty_kw = max(reformer_endo_kw,
                          sizing.FURNACE_THERMAL_EFFICIENCY * burner_comb_kw)
    if furnace_duty_kw > 1.0:
        add_unit("(reformer furnace)", "reformer", "fired_heater_nonreactive",
                 furnace_duty_kw,
                 {"duty_kW": furnace_duty_kw, "reformer_endo_kW": reformer_endo_kw,
                  "burner_combustion_kW": burner_comb_kw})

    # 개질 촉매 (Ni/Al2O3): 개질기는 노로 코스팅하지만 촉매는 별도 소모품.
    for n, (c, _note) in PROCESS_UNIT_MAP[pid].items():
        if c != "reformer":
            continue
        rs = smap_all.get(n)
        if not rs:
            continue
        try:
            s_in = P[str(rs["feed"])]
            m_cat = float(np.atleast_1d(sizing.reformer_catalyst_mass_kg(
                np.array([s_in["vdot_m3s"]]), s_in["T_c"], s_in["P_bar"]))[0])
        except (KeyError, TypeError):
            continue
        c0 = m_cat * sizing.CATALYST_PRICE_REFORMER_USD_PER_KG
        extras["catalyst_initial_usd"] += c0
        extras["catalyst_replace_usd_yr"] += float(np.atleast_1d(
            sizing.catalyst_annual_replacement_cost_usd(c0))[0])

    # --- 나머지 유닛 (버너 제외) ---
    flash_nodes = []
    cooler_prod = {}   # {출구 스트림명: (node, feed 스트림명)}  -- utility_cooler 만
    for row in costing_plan(pid):
        node, cat, smap, wkind = row["node"], row["category"], row["streams"], row["wgs_kind"]
        if smap is None:
            unit_rows.append({"node": node, "category": cat, "error": "no stream map"})
            continue
        if cat == "fired_heater" and smap.get("kind") == "burner":
            continue  # 개질로에 통합됨
        key, capv, aux = _capacity(node, cat, smap, wkind, P, extras)
        add_unit(node, cat, key, capv, aux)
        if cat == "utility_cooler":
            cooler_prod[str(smap["prod"])] = (node, str(smap["feed"]))
        if cat == "flash_vessel":
            flash_nodes.append((node, smap, aux))

    # --- 응축기 처리 ---
    # flash(F1)에서 물이 응축되면서 나오는 잠열은 properties.py가 전상(前相)을
    # 전부 기상으로 보기 때문에 앞단 cooler 듀티에 안 잡히고 F1 지점에 나타난다.
    #   - F1 바로 앞에 utility_cooler 가 있으면: 그 cooler = 실제 응축기.
    #     cooler 듀티를 (cooler입구 -> F1 기/액 출구) 전체 엔탈피차로 다시 잡아
    #     면적/비용을 재계산한다 (별도 F1 응축기 안 만듦).
    #   - 앞단이 process_hx 뿐이면: F1 자체가 응축기 -> 별도 응축기 HX 1기 추가.
    for fnode, fsmap, faux in flash_nodes:
        cd = faux.get("condenser_duty_kw", 0.0)
        if cd <= 100.0:
            continue
        feed_name = str(fsmap["feed"])
        if feed_name in cooler_prod:
            cnode, cfeed = cooler_prod[feed_name]
            crow = next(u for u in unit_rows if u["node"] == cnode)
            real_duty = (_H_flow_kW(P[cfeed]) - _H_flow_kW(P[str(fsmap["vapor_out"])])
                         - (_H_flow_kW(P[str(fsmap["liquid_out"])])
                            if fsmap.get("liquid_out") else 0.0))
            real_duty = abs(real_duty)
            lmtd = sizing.log_mean_temperature_difference(
                P[cfeed]["T_c"], faux["vap_T_c"], COOLING_WATER_IN_C, COOLING_WATER_OUT_C)
            lmtd = max(float(np.atleast_1d(lmtd)[0]), 1.0)
            area = float(np.atleast_1d(sizing.heat_exchanger_area_m2(
                real_duty, sizing.ASPEN_HEATX_U_W_M2K, lmtd))[0])
            bm_2001_total[0] -= crow["cbm_2001usd"]
            unit_rows.remove(crow)
            add_unit(cnode, "utility_cooler", "heat_exchanger_floating_head", area,
                     {"Q_kW": real_duty, "LMTD_K": lmtd, "note": "condenser(+latent)"})
        else:
            lmtd = sizing.log_mean_temperature_difference(
                faux["feed_T_c"], faux["vap_T_c"], COOLING_WATER_IN_C, COOLING_WATER_OUT_C)
            lmtd = max(float(np.atleast_1d(lmtd)[0]), 1.0)
            area = float(np.atleast_1d(sizing.heat_exchanger_area_m2(
                cd, sizing.ASPEN_HEATX_U_W_M2K, lmtd))[0])
            add_unit(f"({fnode} condenser)", "utility_cooler",
                     "heat_exchanger_floating_head", area, {"Q_kW": cd, "LMTD_K": lmtd})

    bm_2001_total = bm_2001_total[0]

    # --- CAPEX ---
    grassroots_2001 = tc.grassroots_cost(np.array([bm_2001_total]))
    fci = float(tc.escalate_cepci(grassroots_2001, cepci_current))
    fci += extras["adsorbent_initial_usd"] + extras["catalyst_initial_usd"]
    tci = float(ec.total_capital_investment(fci))

    # --- OPEX (COM_d) ---
    hours = ec.OPERATING_HOURS_PER_YEAR
    shaft_net_kw = (shaft_power_kw["compressor"] + shaft_power_kw["pump"]
                    - shaft_power_kw["turbine"])
    elec_price = prices["electricity_usd_per_kwh"]
    # 순 소비면 유틸리티비(C_UT), 순 발전이면 0 (초과분은 아래 co-product 크레딧으로)
    c_ut = max(0.0, shaft_net_kw) * hours * elec_price
    net_elec_export_kw = max(0.0, -shaft_net_kw)

    ng_per_mj = prices["natural_gas_usd_per_kg"] / NATURAL_GAS_LHV_MJ_PER_KG  # $/MJ

    # 원료: 공급 CH4 + 버너 FUEL 스트림 = 구매 NG ; 용수 = WATER/H2O
    # 구매 원료: 개질 원료 CH4(스트림) + 신규 연료 FUEL(스트림, 계 경계 입력) + 용수
    ng_kgh = sum(s["mdot_kgh"] for nm, s in streams.items()
                 if nm.upper() in ("CH4", "FUEL"))
    water_kgh = sum(s["mdot_kgh"] for nm, s in streams.items()
                    if nm.upper() in ("WATER", "H2O"))
    c_rm = (ng_kgh * hours * prices["natural_gas_usd_per_kg"]
            + water_kgh * hours * prices["process_water_usd_per_kg"])

    # --- 개질로/HEAT 를 돌리는 데 필요한 "구매 보충연료" ---
    #   총 연료소요[LHV] = (개질로 흡수듀티 + HEAT 듀티) / eta_furnace
    #   이미 확보된 연료  = 버너에 들어가는 내부 스트림(테일가스·미반응가스·FUEL스트림)의
    #                       가연분(H2,CH4,CO) 발열량 합
    #   보충연료 = max(0, 총소요 - 확보)  -> NG 로 계상 (heater_fuel_mode="electric"면
    #             그 부족분을 전력가열로 보고 전력단가로 C_UT에)
    heat_node_duty_kw = sum(u.get("duty_kW", 0.0) for u in unit_rows
                            if u.get("category") == "fired_heater")
    total_fired_energy_kw = (furnace_duty_kw + heat_node_duty_kw) / sizing.FURNACE_THERMAL_EFFICIENCY
    burner_feed_lhv_kw = 0.0
    for bn in burner_nodes:
        for sname in smap_all[bn].get("feed", []):
            s = streams.get(str(sname))
            if not s:
                continue
            n = _mol_flows_kmolh(s)
            burner_feed_lhv_kw += sum(
                n[SPECIES.index(sp)] * MW[SPECIES.index(sp)] * LHV_MJ_PER_KG.get(sp, 0.0)
                for sp in ("H2", "CH4", "CO")) * (1000.0 / 3600.0)  # kmol/h*kg/kmol*MJ/kg -> kW
    supplemental_fuel_kw = max(0.0, total_fired_energy_kw - burner_feed_lhv_kw)

    c_fuel_fired = c_heat_electric = 0.0
    if heater_fuel_mode == "electric":
        c_heat_electric = supplemental_fuel_kw * sizing.FURNACE_THERMAL_EFFICIENCY \
            * hours * elec_price   # 전기가열은 ~100%효율이므로 흡수듀티 기준
        c_ut += c_heat_electric
    else:
        c_fuel_fired = supplemental_fuel_kw * hours * 3.6 * ng_per_mj  # kW*h*3.6 = MJ/yr
        c_rm += c_fuel_fired

    # 냉각수 유틸리티비: utility_cooler(COOL 노드 + F1 응축기)가 제거하는 열.
    #   공정-공정 HX(process_hx)는 내부 회수라 유틸리티 소비 없음.
    #   c = Σduty[kW] * (3600*h/1e6) [GJ/yr] * 냉각수단가[$/GJ]. (냉수/냉동 아님)
    cooling_duty_kw = sum(u.get("Q_kW", 0.0) for u in unit_rows
                          if u.get("category") == "utility_cooler")
    c_cooling = (cooling_duty_kw * (3600.0 * hours / 1.0e6)
                 * prices.get("cooling_water_usd_per_gj", 0.35))
    c_ut += c_cooling

    # 폐수처리비 (C_WT): 계 밖으로 배출되는 응축수.
    #   SMR 공정 응축수는 용존 CO2·미량 유기물 함유 -> 스트립/처리 필요.
    #   condensate_streams 로 스트림명 리스트를 넘기면 그걸 쓰고(objective.py가
    #   역할매핑의 role=="condensate" 를 넘김), 안 넘기면 계 경계 출구 중
    #   H2O 몰분율 >0.9 인 것(과열 스팀 수출 RESTEAM 은 제외되도록 액상 판정).
    from process_stream_map import OUTPUT_STREAMS
    if condensate_streams is None:
        outs = OUTPUT_STREAMS.get(pid, [])
        condensate_streams = [nm for nm in outs
                              if nm in P and P[nm]["x"][SPECIES.index("H2O")] > 0.9
                              and P[nm]["phase"] == props.PHASE_LIQUID]
    cond_kgh = sum(streams[nm]["mdot_kgh"] for nm in condensate_streams if nm in streams)
    c_wastewater = (cond_kgh / 1000.0) * hours * prices.get("wastewater_usd_per_m3", 0.5)
    # 탄소세 (선택): prices["co2_tax_usd_per_kg"] 주면 대기방출 CO2에 부과.
    #   대기방출량은 여기서 계산 안 하므로(indicators 소관) prices["co2_emission_kgh"] 로 받는다.
    c_carbon = (prices.get("co2_emission_kgh", 0.0) * hours
                * prices.get("co2_tax_usd_per_kg", 0.0))
    c_wt = c_wastewater + c_carbon

    n_units = sum(1 for r in unit_rows if "error" not in r)
    n_ol = float(ec.operating_labor_headcount(0, n_units))
    c_ol = n_ol * hours * prices["operating_labor_usd_per_hr"]

    replace_yr = extras["adsorbent_replace_usd_yr"] + extras["catalyst_replace_usd_yr"]
    c_variable = c_ut + c_rm + c_wt + replace_yr

    if opex_method == "turton_com":
        # 방법 A: Turton COM_d (판관비·R&D 마크업 포함). 촉매재생은 그 밖에 가산.
        com_d = float(ec.cost_of_manufacturing(fci, c_ol, c_ut, c_rm, c_wt_usd_per_year=c_wt))
        opex_annual = com_d + replace_yr
        fixed_breakdown = {"COM_d_fci_term": ec.COM_FCI_COEFF * fci,
                           "COM_d_labor_term": ec.COM_LABOR_COEFF * c_ol}
    else:
        # 방법 B (기본): 항목별. 유지보수 4%FCI + 세금보험 2%FCI + 인건비 + 간접비 60%
        opex_annual, fixed_breakdown = ec.itemized_opex(fci, c_ol, c_variable)
        opex_annual = float(opex_annual)
        fixed_breakdown = {k: float(v) for k, v in fixed_breakdown.items()}

    # --- 제품 스트림: H2(분모) + 부산물 크레딧 ---
    prod_name = next((c for c in ("PROD", "PRODUCT", "H2") if c in P), None)
    # ng_per_mj 는 위에서 계산됨

    h2_kg_yr = np.nan
    syngas_fuel_credit_yr = 0.0
    if prod_name is not None:
        sp = P[prod_name]
        mdot_s = sp["mdot_kgh"] / 3600.0  # kg/s
        xs = sp["x"]
        mmix = max(sp["m_mix"], 1e-9)
        # 성분별 질량분율 = x_i * MW_i / M_mix
        wfrac = {s: xs[SPECIES.index(s)] * MW[SPECIES.index(s)] / mmix
                 for s in ("H2", "CH4", "CO")}
        h2_kg_yr = mdot_s * wfrac["H2"] * hours * 3600.0
        # 합성가스의 비-H2 가연분(CH4, CO)은 연료가치로 크레딧 (NG 단가 대비 에너지환산)
        fuel_mj_s = sum(mdot_s * wfrac[s] * LHV_MJ_PER_KG[s] for s in ("CH4", "CO"))
        syngas_fuel_credit_yr = fuel_mj_s * (hours * 3600.0) * ng_per_mj

    # 수출 스팀 크레딧 (P01 RESTEAM 등): 보일러 회피비용
    steam_kgh = sum(s["mdot_kgh"] for nm, s in streams.items()
                    if "RESTEAM" in nm.upper() or nm.upper() == "STEAM_EXPORT")
    steam_credit_yr = (steam_kgh * hours
                       * (STEAM_EFFECTIVE_ENTHALPY_MJ_PER_KG / BOILER_EFFICIENCY)
                       * ng_per_mj)

    # 순 전력수출 크레딧
    elec_credit_yr = net_elec_export_kw * hours * elec_price

    coproduct_credit_yr = syngas_fuel_credit_yr + steam_credit_yr + elec_credit_yr

    # --- LCOH (부산물 크레딧 차감) ---
    #   LCOH = [ NPV(CAPEX) + NPV(OPEX - 부산물수익) ] / NPV(H2 생산량)
    nyr = CONSTRUCTION_YEARS + plant_life_years
    capex_sched = np.zeros(nyr); capex_sched[0] = tci
    opex_net_sched = np.concatenate(
        [[0.0], np.full(plant_life_years, opex_annual - coproduct_credit_yr)])
    h2_sched = np.concatenate([[0.0], np.full(plant_life_years, h2_kg_yr)])
    lcoh = (float(ec.lcoh(capex_sched, opex_net_sched, h2_sched))
            if np.isfinite(h2_kg_yr) and h2_kg_yr > 0 else np.nan)
    lcoh_gross = (float(ec.lcoh(capex_sched,
                                np.concatenate([[0.0], np.full(plant_life_years, opex_annual)]),
                                h2_sched))
                  if np.isfinite(h2_kg_yr) and h2_kg_yr > 0 else np.nan)

    return {
        "process_id": pid,
        "units": unit_rows,
        "bare_module_2001usd": bm_2001_total,
        "fci_usd": fci, "tci_usd": tci,
        "opex_usd_per_year": opex_annual,
        "opex_method": opex_method,
        "opex_fixed_breakdown_usd_per_year": fixed_breakdown,
        "opex_variable_usd_per_year": c_variable,
        "catalyst_adsorbent_replace_usd_per_year": replace_yr,
        "c_ut": c_ut, "c_rm": c_rm, "c_ol": c_ol, "c_wt": c_wt,
        "c_fuel_fired_usd_per_year": c_fuel_fired,
        "c_heat_electric_usd_per_year": c_heat_electric,
        "c_cooling_water_usd_per_year": c_cooling,
        "cooling_duty_kW": cooling_duty_kw,
        "total_fired_energy_kW": total_fired_energy_kw,
        "burner_internal_fuel_kW": burner_feed_lhv_kw,
        "supplemental_fuel_kW": supplemental_fuel_kw,
        "shaft_power_kW": shaft_power_kw,
        "net_electric_export_kW": net_elec_export_kw,
        "coproduct_credit_usd_per_year": coproduct_credit_yr,
        "credit_breakdown_usd_per_year": {
            "electricity": elec_credit_yr, "syngas_fuel": syngas_fuel_credit_yr,
            "steam": steam_credit_yr},
        "extras_usd": extras,
        "h2_kg_per_year": h2_kg_yr,
        "lcoh_usd_per_kg": lcoh,               # 부산물 크레딧 반영 (대표값)
        "lcoh_gross_usd_per_kg": lcoh_gross,   # 크레딧 없이
        # 개질로 진단
        "reformer_endothermic_kW": reformer_endo_kw,
        "burner_combustion_kW": burner_comb_kw,
        "furnace_duty_kW": furnace_duty_kw,
        "range_adjusted_units": [u["node"] for u in unit_rows
                                 if u.get("range_status") in ("split", "clamped_low")],
    }


# =====================================================================
# 참고용 가격 (그대로 쓰지 말 것)
# =====================================================================

REFERENCE_PRICES = {
    # 글로벌 평균 기준 (2026-09-02 확정). 논문에서 출처와 함께 명시할 것.
    "electricity_usd_per_kwh": 0.08,      # 사용자 지정. 전력 소비/수출 공통 단가로 사용.
    "natural_gas_usd_per_kg": 0.30,       # 글로벌 평균 ~6 $/GJ_HHV (Henry Hub/TTF/JKM 혼합,
                                          #   IEA·World Bank Commodity Markets 2015-2024).
                                          #   NG HHV ~52 MJ/kg -> 0.30 $/kg. (Moon 2025의
                                          #   US-only 0.218 대비 글로벌 반영해 상향)
    "process_water_usd_per_kg": 1.5e-3,   # 글로벌 산업용수 ~1.5 $/m3
    "cooling_water_usd_per_gj": 0.35,     # 냉각수 제거열당 (Turton 유틸리티 표준 ~0.35 $/GJ)
    "wastewater_usd_per_m3": 0.5,         # 응축수 처리 (재순환 설계면 0)
    # "co2_tax_usd_per_kg": 0.0,          # 탄소세 (넣으려면 co2_emission_kgh 도 함께)
    "operating_labor_usd_per_hr": 35.0,   # OECD 로디드 운전원 임금 근사. 비OECD면 하향.
}
