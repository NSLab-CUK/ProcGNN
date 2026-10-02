"""
equipment_sizing.py
properties.py가 계산한 체적유량(Vol_Flow)을 turton_costing.py의 장비별 용량변수
(압축기/펌프는 동력 kW, 열교환기는 면적 m2, 베셀/반응기는 부피 m3)로 변환하는 모듈.

배경 (지난 논의 결론)
    Turton 표준 상관식 자체는 체적유량을 직접 회귀변수로 쓰지 않는다. 압축기/펌프는
    동력(kW), 열교환기는 전열면적(m2), 베셀/반응기는 부피(m3)를 쓴다. 그러나 그 동력과
    부피를 구하는 계산 자체가 체적유량을 반드시 거쳐야 하므로, 체적유량은 이 코스팅
    파이프라인 전체를 관통하는 공통 허브 변수다. 이 모듈이 그 변환을 전담한다.

        properties.py (V_dot)
            -> 압축기/펌프: 등엔트로피 압축일/유량*양정 공식으로 동력 산출
            -> 열교환기   : Q/(U*LMTD)로 면적 산출 (V_dot은 유속/압손 점검에만 간접 사용)
            -> 베셀/PSA   : Souders-Brown식으로 직경, 체류시간으로 길이/부피 산출
            -> 반응기     : 체류시간 * V_dot 으로 부피 산출
        -> turton_costing.purchased_cost_2001usd(equipment_key, capacity_value)

GNN 대리모델은 열부하(Q), 압축기/펌프 동력, 반응기 체류시간을 직접 출력하지 않으므로
이 모듈의 함수들은 모두 "가정값 또는 별도 산정값을 받는" 인터페이스로 설계했다.
열부하 Q 자체는 Module 2(공정 지표)의 unit_duties가 엔탈피수지로 계산해 넘겨주는
값을 그대로 받는다 (GA_코드구축_프롬프트.md 기존 계획과 연결).
"""

from __future__ import annotations

import numpy as np

# =====================================================================
# 0. Aspen Plus 실측 스펙 (2026-09-02 확인)
# =====================================================================
#
# 확인 방법: steady state_june 데이터셋의 최신 apw 파일
#   "DataSet/Steady state/steady state_june/아스펜/1 SMR.apw" (Aspen Plus V14)
#   을 열어 둔 상태에서 win32com으로 Running Object Table의 파일 모니커에 붙어
#   (GetActiveObject("Apwn.Document")는 V14에서 실패 -> ROT 파일 모니커 사용)
#   각 블록의 Input/Output 트리를 읽었다. apw 파일은 읽기 전용으로만 조회했다.
#
# --- 열교환기 총괄전열계수 U -------------------------------------------------
#   HeatX 블록 HX1~HX5는 전부 Shortcut 모드이고, 다음이 "고정 입력"돼 있다:
#       U_OPTION   = 'PHASE'          (구간별 상수 U)
#       U_REF_VALUE = 850.0           단위 Watt/sqm-K  (SMR 유닛셋)
#       HOT_FILM / COLD_FILM = 미지정, GEOMETRY = 'NO'
#   즉 필름계수 상관식이나 기하구조 기반 자동계산이 아니라 사용자가 850을
#   직접 박아 넣은 값이다. Aspen 자체 출력으로 역검증:
#       HX1: 21.844 MW /(670.109 m2 * 38.350 K) = 850.0000
#       HX2: 82.012 MW /(596.505 m2 * 161.750 K) = 850.0000
#       HX3: 11.976 MW /(52.792 m2 * 266.875 K)  = 850.0000
#       HX4:  8.388 MW /(129.800 m2 * 76.030 K)  = 850.0000
#       HX5:  8.506 MW /(322.865 m2 * 30.995 K)  = 850.0000
#   또 hx_implied_u.csv 에서 HX1_U_implied = 850.00 (표준편차 0.00, 50케이스).
#   => HX2~HX5의 역산 U가 850을 벗어나 보이는 것(평균 1315/899/646/374)은
#      실제 U가 달라서가 아니라, 다구간(상변화 포함) 열교환기의 유효 LMTD를
#      말단 온도만으로 재구성할 때 생기는 오차가 U로 흡수된 것이다.
#      실제 Aspen 스펙은 5개 모두 정확히 850이다. (HX1은 단일구간이라
#      말단 LMTD == 실제 LMTD 여서 역산이 정확히 850으로 떨어진다.)
ASPEN_HEATX_U_W_M2K = 850.0

# --- 펌프 효율 -------------------------------------------------------------
#   Pump 블록 P1/P2/P3는 효율 입력이 전부 비어 있다
#   (EFF, DEFF, EFF_COEF1..4, EFF_DATA, CURVES, USER_CURVES 전부 미지정 =
#    성능곡선 참조도 아니고 사용자 고정값도 아님).
#   Aspen이 유량 의존 내부 기본 상관식으로 효율을 자동 산정하며(CEFF로 출력),
#   따라서 케이스마다 값이 달라 "단일 고정 스펙값"이 존재하지 않는다.
#     - "1 SMR.apw" 단일 케이스 Aspen CEFF : P1 0.659, P2 0.581, P3 0.433
#     - 프로젝트 50케이스 역산 효율(실측 W_P로 역산): P1 0.570, P2 0.465, P3 0.375
#   대리모델이 서비스하는 모집단(50케이스 DOE)에 맞춰 역산 평균값을 대표 효율로
#   채택한다. Aspen 단일 케이스 CEFF는 크기 순서(P1>P2>P3)와 대략적 수준을
#   교차확인해 준다(약 0.09 차이는 케이스 간 산포 + 단일 표본의 운전점 차이).
PUMP_ISENTROPIC_EFFICIENCY_BY_BLOCK = {
    "P1": 0.570,
    "P2": 0.465,
    "P3": 0.375,
}

# --- 압축기 / 터빈 효율 --------------------------------------------------
#   Compressor 블록 C1/C2/C3, Turbine 블록 T1/T2/T3 모두
#       TYPE = 'ISENTROPIC', SEFF = 0.75 (고정 사용자 입력), EFF_MECH = 1.0,
#       성능곡선 미사용(CURVE_REG='NONE', CALC_EFF='NO').
#   => 압축기/터빈은 등엔트로피 효율 0.75 가 실제 Aspen 스펙이다.
#      (터빈도 0.80 이 아니라 0.75)
ASPEN_COMPRESSOR_ISENTROPIC_EFF = 0.75
ASPEN_TURBINE_ISENTROPIC_EFF = 0.75

# =====================================================================
# 1. 압축기 / 펌프 동력 (Turton capacity_param = power_kw)
# =====================================================================

def compressor_isentropic_power_kw(vol_flow_in_m3s, p_in_bar, p_out_bar,
                                    k_ratio=1.4, isentropic_efficiency=0.75):
    """
    압축기 축동력 [kW], 이상기체 등엔트로피 압축일 공식.

        W_isentropic = (k/(k-1)) * P_in * V_dot_in * [ (P_out/P_in)^((k-1)/k) - 1 ]
        W_shaft      = W_isentropic / eta_isentropic

    입력 체적유량은 반드시 흡입(inlet) 조건의 체적유량이어야 한다
    (properties.py의 압축기 상류측 스트림에서 계산한 vol_flow_m3s를 그대로 사용).

    Parameters
    ----------
    vol_flow_in_m3s : 흡입측 체적유량 [m3/s]
    p_in_bar, p_out_bar : 흡입/토출 압력 [bar]
    k_ratio : 비열비 Cp/Cv. 공정가스 혼합물 근사 기본값 1.4(공기 근사).
              성분별 정밀 비열비가 필요하면 properties.py의 Cp 다항식으로 교체할 것.
    isentropic_efficiency : 등엔트로피 효율. 기본 0.75.
              2026-09-02 Aspen 조회로 확인: C1/C2/C3 블록의 SEFF = 0.75 고정
              입력값이므로 이 기본값이 실제 시뮬레이션 스펙과 일치한다
              (ASPEN_COMPRESSOR_ISENTROPIC_EFF 참조).

    Returns
    -------
    power_kw : 축동력 [kW]
    """
    v_in = np.asarray(vol_flow_in_m3s, dtype=float)
    p_in_pa = np.asarray(p_in_bar, dtype=float) * 1.0e5
    p_out_pa = np.asarray(p_out_bar, dtype=float) * 1.0e5

    pressure_ratio = p_out_pa / p_in_pa
    exponent = (k_ratio - 1.0) / k_ratio

    w_isentropic_w = (k_ratio / (k_ratio - 1.0)) * p_in_pa * v_in * (
        pressure_ratio**exponent - 1.0
    )
    w_shaft_w = w_isentropic_w / isentropic_efficiency
    return w_shaft_w / 1000.0  # W -> kW


def turbine_isentropic_power_kw(vol_flow_in_m3s, p_in_bar, p_out_bar,
                                 k_ratio=1.4, isentropic_efficiency=0.75):
    """
    터빈(팽창기) 축동력 [kW], 이상기체 등엔트로피 팽창일 공식.
    압축기 공식(compressor_isentropic_power_kw)의 부호를 뒤집은 형태다.

        W_isentropic = (k/(k-1)) * P_in * V_dot_in * [ 1 - (P_out/P_in)^((k-1)/k) ]
        W_shaft      = W_isentropic * eta_isentropic   (팽창기는 효율을 곱함, 압축기는 나눔)

    P_out < P_in 조건에서 대괄호 항이 양수가 되어 양의 추출동력이 나온다.
    인접행렬 노드타입 "turbine"에 대응하는 유닛(예 배가스 팽창 터빈, 스팀터빈)에 사용.

    Parameters
    ----------
    vol_flow_in_m3s : 흡입측(터빈 입구) 체적유량 [m3/s]
    p_in_bar, p_out_bar : 입구/출구 압력 [bar], p_out_bar < p_in_bar 이어야 정상
    k_ratio : 비열비, 기본 1.4 (자리표시 가정값)
    isentropic_efficiency : 등엔트로피 효율, 기본 0.75.
              2026-09-02 Aspen 조회로 확인: T1/T2/T3 블록의 SEFF = 0.75 고정
              입력값이다 (이전 기본값 0.80은 실제 스펙과 달라 0.75로 교체함,
              ASPEN_TURBINE_ISENTROPIC_EFF 참조).

    Returns
    -------
    power_kw : 추출 축동력 [kW], 양수
    """
    v_in = np.asarray(vol_flow_in_m3s, dtype=float)
    p_in_pa = np.asarray(p_in_bar, dtype=float) * 1.0e5
    p_out_pa = np.asarray(p_out_bar, dtype=float) * 1.0e5

    pressure_ratio = p_out_pa / p_in_pa
    exponent = (k_ratio - 1.0) / k_ratio

    w_isentropic_w = (k_ratio / (k_ratio - 1.0)) * p_in_pa * v_in * (
        1.0 - pressure_ratio**exponent
    )
    w_shaft_w = w_isentropic_w * isentropic_efficiency
    return w_shaft_w / 1000.0  # W -> kW


def pump_power_kw(vol_flow_m3s, delta_p_bar, pump_efficiency=0.75):
    """
    펌프 축동력 [kW].

        W_hydraulic = V_dot * dP
        W_shaft     = W_hydraulic / eta_pump

    Parameters
    ----------
    vol_flow_m3s : 액체 체적유량 [m3/s]
    delta_p_bar : 토출-흡입 압력차 [bar]
    pump_efficiency : 펌프 효율. 기본 0.75는 사양 미상 펌프의 일반 가정값이다.
        1번 공정의 P1/P2/P3는 Aspen에서 효율을 지정하지 않아 유량 의존 내부
        상관식으로 자동 산정되며(고정값도 곡선참조도 아님), 대표값은
        PUMP_ISENTROPIC_EFFICIENCY_BY_BLOCK 에 블록별로 정리해 두었다
        (P1 0.570 / P2 0.465 / P3 0.375). 해당 블록을 계산할 때는 그 값을
        명시적으로 넘길 것.
    """
    v = np.asarray(vol_flow_m3s, dtype=float)
    dp_pa = np.asarray(delta_p_bar, dtype=float) * 1.0e5
    w_hydraulic_w = v * dp_pa
    return (w_hydraulic_w / pump_efficiency) / 1000.0  # kW


# =====================================================================
# 2. 열교환기 면적 (Turton capacity_param = area_m2)
# =====================================================================

def heat_exchanger_area_m2(duty_kw, overall_u_w_m2k, lmtd_k):
    """
    열교환기 전열면적 [m2].

        A = Q / (U * LMTD)

    Parameters
    ----------
    duty_kw : 열부하 |Q| [kW] (Module 2의 unit_duties에서 산출된 값을 사용,
              부호는 절대값으로 넣을 것)
    overall_u_w_m2k : 총괄열전달계수 [W/(m2*K)]. 이 함수는 값을 받아만 계산하며
              기본값을 내부에 두지 않는다 (잘못된 U 가정을 감추지 않기 위함).
              1번 공정의 HX1~HX5는 Aspen에서 U를 850 W/m2K로 고정 입력해 두었으므로
              (2026-09-02 확인, 필름계수 자동계산 아님) 이 공정에는
              ASPEN_HEATX_U_W_M2K (=850.0) 를 넘기면 된다.
    lmtd_k : 로그평균온도차 [K]

    Returns
    -------
    area_m2
    """
    q_w = np.abs(np.asarray(duty_kw, dtype=float)) * 1000.0
    u = np.asarray(overall_u_w_m2k, dtype=float)
    lmtd = np.asarray(lmtd_k, dtype=float)
    return q_w / (u * lmtd)


def log_mean_temperature_difference(t_hot_in, t_hot_out, t_cold_in, t_cold_out):
    """
    로그평균온도차 LMTD [K] (병류/향류 공용, 온도차 부호가 같은 정상적인 경우).
    delta_T1, delta_T2가 같으면(등온 상변화 등) 산술평균으로 대체한다 (0/0 방지).
    """
    d1 = np.asarray(t_hot_in, dtype=float) - np.asarray(t_cold_out, dtype=float)
    d2 = np.asarray(t_hot_out, dtype=float) - np.asarray(t_cold_in, dtype=float)

    # d1/d2 <= 0 (온도 교차/핀치) 또는 d1~d2 인 곳은 산술평균으로 대체 (log 발산 방지)
    ratio = np.divide(d1, d2, out=np.ones_like(d1 * d2), where=(d2 != 0))
    safe = (d2 != 0) & (ratio > 0) & (~np.isclose(d1, d2))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lmtd = (d1 - d2) / np.log(np.where(safe, ratio, np.e))
    return np.where(safe, log_lmtd, (d1 + d2) / 2.0)


# =====================================================================
# 3. 베셀 / 분리기 / PSA 사이징 (Turton capacity_param = volume_m3)
# =====================================================================

# GPSA Engineering Data Book: 수평 메시패드가 있는 수직 드럼의 Souders-Brown K [m/s]
# 대 게이지압력 [barg]. (웹 조회 2026-09-02, GPSA Engineering Data Book 인용 자료.
# en.wikipedia.org/wiki/Souders-Brown_equation 및 GPSA 발췌 자료 교차확인.)
#   0 barg -> 0.107 / 7 -> 0.107 / 21 -> 0.101 / 42 -> 0.092 / 63 -> 0.083 / 105 -> 0.065
# 규칙: 7 barg 초과 시 7 bar당 K를 0.003 낮춘다. 메시패드가 없으면 K를 절반으로.
_GPSA_KSB_BARG = np.array([0.0, 7.0, 21.0, 42.0, 63.0, 105.0])
_GPSA_KSB_VAL = np.array([0.107, 0.107, 0.101, 0.092, 0.083, 0.065])


def souders_brown_k_factor(pressure_barg, has_mesh_pad=True):
    """
    운전 게이지압력에 따른 Souders-Brown K [m/s]를 GPSA 표로 구한다.

    표 범위(0~105 barg)는 선형보간하고, 105 barg 초과는 GPSA 규칙
    (7 bar당 -0.003)으로 외삽하되 하한 0.03 m/s로 자른다.
    has_mesh_pad=False (패드 없는 수직 분리기)면 K를 0.5배 한다.
    """
    p = np.asarray(pressure_barg, dtype=float)
    k = np.interp(p, _GPSA_KSB_BARG, _GPSA_KSB_VAL)
    over = p > 105.0
    if np.any(over):
        k = np.where(over,
                     np.maximum(0.065 - 0.003 * (p - 105.0) / 7.0, 0.03),
                     k)
    if not has_mesh_pad:
        k = k * 0.5
    return k


def souders_brown_max_vapor_velocity(rho_liquid_kg_m3, rho_vapor_kg_m3, k_sb=0.107):
    """
    Souders-Brown식으로 허용 최대 증기유속 [m/s]을 계산한다.

        v_max = k_sb * sqrt( (rho_L - rho_V) / rho_V )

    k_sb는 souders_brown_k_factor()로 운전압력에 맞게 구해서 넘기는 것을 권장한다.
    기본값 0.107은 상압/저압(<=7 barg) + 메시패드 조건의 GPSA 값이다.
    """
    rho_l = np.asarray(rho_liquid_kg_m3, dtype=float)
    rho_v = np.asarray(rho_vapor_kg_m3, dtype=float)
    return k_sb * np.sqrt(np.maximum(rho_l - rho_v, 0.0) / rho_v)


def vertical_vessel_sizing(vapor_vol_flow_m3s, rho_liquid_kg_m3, rho_vapor_kg_m3,
                            pressure_barg=None, k_sb=None, has_mesh_pad=True,
                            length_to_diameter=3.0,
                            liquid_vol_flow_m3s=0.0, liquid_holdup_min=5.0):
    """
    플래시드럼 / 응축수 녹아웃드럼 / 기액 분리기 등 수직 압력용기 사이징.
    (PSA는 흡착층 사이징이 필요하므로 psa_bed_sizing()을 쓸 것.)

    1. Souders-Brown식으로 허용 증기유속 v_max를 구한다.
       k_sb를 직접 주지 않으면 pressure_barg로 souders_brown_k_factor()를 호출한다
       (GPSA 압력보정). pressure_barg도 없으면 0 barg(=0.107)로 처리한다.
    2. 단면적 = 증기 체적유량 / v_max 로부터 증기부 직경 D를 구한다.
    3. 액상 홀드업 부피 V_hold = 액상 체적유량 * 홀드업시간 을 더해준다.
       liquid_holdup_min 기본 5분(half-full 기준)은 Turton 5th ed. §11 및
       Svrcek & Monnery (1993) Chem. Eng. Prog. 89(10), 41 의 기액분리기 통상값이다
       (liquid_vol_flow_m3s=0이면 홀드업 항은 0, 즉 기존 동작과 동일).
    4. 길이는 max(L/D 관행값(Turton 수직베셀 기본 3), 홀드업 액주높이 + 증기 이탈여유
       1.5*D)로 잡는다. L/D가 5를 넘으면 수평 드럼으로 전환하는 것이 맞다(경고만).
    5. 부피 = (pi/4) * D^2 * L

    Returns
    -------
    dict(diameter_m, length_m, volume_m3, k_sb, v_max_m_s)
    """
    v_dot = np.asarray(vapor_vol_flow_m3s, dtype=float)

    if k_sb is None:
        p_barg = 0.0 if pressure_barg is None else pressure_barg
        k_sb = souders_brown_k_factor(p_barg, has_mesh_pad)
    v_max = souders_brown_max_vapor_velocity(rho_liquid_kg_m3, rho_vapor_kg_m3, k_sb)

    area_m2 = v_dot / np.maximum(v_max, 1e-8)
    diameter_m = np.sqrt(4.0 * area_m2 / np.pi)

    v_hold_m3 = np.asarray(liquid_vol_flow_m3s, dtype=float) * (liquid_holdup_min * 60.0)
    l_holdup = v_hold_m3 / np.maximum((np.pi / 4.0) * diameter_m**2, 1e-9)
    length_m = np.maximum(length_to_diameter * diameter_m,
                          l_holdup + 1.5 * diameter_m)
    volume_m3 = (np.pi / 4.0) * diameter_m**2 * length_m

    prefer_horizontal = (length_m / np.maximum(diameter_m, 1e-9)) > 5.0

    return {"diameter_m": diameter_m, "length_m": length_m, "volume_m3": volume_m3,
            "k_sb": k_sb, "v_max_m_s": v_max, "prefer_horizontal": prefer_horizontal}


# ---------------------------------------------------------------------
# 3-b. PSA 흡착층 사이징
# ---------------------------------------------------------------------
#
# 10개 공정 중 P3~P10의 SEP1은 Aspen에서 Sep(이상 성분분리) 블록으로 모델링돼
# 있으나 실장비는 H2 정제용 PSA다. PSA는 기액분리가 아니라 흡착층이므로
# Souders-Brown(vertical_vessel_sizing) 기준을 쓰면 흡착층을 크게 과소평가한다.
# 흡착단계 공탑속도 상한으로 직경을 정하는 별도 식을 쓴다.
#
# 설계 상수 (모두 출처 명시, screening 수준):
#   u_feed  흡착단계 공탑속도 0.35 m/s. H2 PSA에서 조기유동화(u_mf) 이하로
#           유지되는 보고범위 0.2~0.5 m/s의 중앙값. 이 값으로 흡착탑 직경을 정한다.
#           출처: Ruthven, Farooq & Knaebel (1994), "Pressure Swing Adsorption", VCH;
#                 Sircar, S. & Golden, T.C. (2000), Sep. Sci. Technol. 35(5), 667.
#   L/D     흡착탑 세장비 2.5. 산업용 H2 PSA 흡착탑 통상 D 1~4.5 m, L 3~10 m.
#           출처: Sircar & Golden (2000).
#   N_beds  기본 4베드 (Skarstrom + 압력평형 단계 포함, 99.99% 순도 / 75~90% 회수
#           를 내는 실용적 최소 구성). 베드당 직경이 D_max(수송/샵제작 한계
#           4.5 m)를 넘으면 베드 수를 늘린다 (Air Products Polybed류 4~16베드).
#           출처: Sircar & Golden (2000); Turton 5th ed. (대형 용기 제작한계).
#   순간 급기 베드 수 = 1 (4베드 Skarstrom 기준). 이 1기가 전량 공급을 감당하도록
#           보수적으로 사이징한다.
#   흡착제  적층베드(활성탄 + 제올라이트 5A/13X) 벌크밀도 ~700 kg/m3.
#           단가 ~5 $/kg (2020년대, 활성탄 2~3 + 제올라이트 5~10 혼합 근사).
#           출처: Sircar & Golden (2000); 벤더 시세 근사.
#   흡착제 초기충전은 별도 CAPEX 라인, 재생/교체는 별도 OPEX 라인으로 넣는다
#   (psa_adsorbent_inventory / psa_adsorbent_annual_replacement_cost 참조).
#   Turton process_vessel_vertical 상관식은 용기 자체만 커버한다.
PSA_FEED_SUPERFICIAL_VELOCITY_M_S = 0.35
PSA_BED_LENGTH_TO_DIAMETER = 2.5
PSA_MIN_N_BEDS = 4
PSA_MAX_BED_DIAMETER_M = 4.5
PSA_N_BEDS_ON_FEED = 1
PSA_ADSORBENT_BULK_DENSITY_KG_M3 = 700.0
PSA_ADSORBENT_PRICE_USD_PER_KG = 5.0
# 재생비: Lee et al. (2021) Appl. Sci. 11, 6021 각주 - 3개월마다 10%를 원가의 40%로
# 교체 -> 연간 0.10*(4/yr)*0.40 = 0.16 * (초기 흡착제비) /yr.
PSA_ADSORBENT_ANNUAL_REPLACEMENT_FRACTION = 0.16


def psa_bed_sizing(feed_vapor_vol_flow_m3s,
                    superficial_velocity_m_s=PSA_FEED_SUPERFICIAL_VELOCITY_M_S,
                    length_to_diameter=PSA_BED_LENGTH_TO_DIAMETER,
                    beds_per_train=PSA_MIN_N_BEDS,
                    max_bed_diameter_m=PSA_MAX_BED_DIAMETER_M):
    """
    H2 PSA 흡착탑 사이징 (흡착압력·온도 조건 기준).

    구성: beds_per_train기(기본 4) 1조를 1트레인으로 보고, 각 트레인에서 항상
    1기가 급기(adsorption)를 받는 4베드 Skarstrom 구성을 기준으로 한다.
    유량이 커서 베드 직경이 max_bed_diameter_m를 넘으면 동일 트레인을 병렬 증설한다.

        A_feed   = V_dot_feed / u                       # 급기 총 단면적
        D_1bed   = sqrt(4 A_feed / pi)                  # 트레인 1개(급기 1기) 가정
        n_trains = max(1, ceil( (D_1bed / D_max)^2 ))
        D_bed    = D_1bed / sqrt(n_trains)              # <= D_max
        L_bed    = (L/D) * D_bed
        n_beds   = beds_per_train * n_trains
        V_total  = n_beds * (pi/4) D_bed^2 L_bed

    feed_vapor_vol_flow_m3s
        PSA 공급가스의 흡착압력·온도 조건 체적유량 [m3/s] (properties.py로 PSA
        상류 스트림에서 계산).

    비용 연결
        베드 1기를 turton_costing.process_vessel_vertical(부피 m3)로 코스팅 후
        n_beds배. 흡착제 비용은 psa_adsorbent_* 로 별도.

    Returns
    -------
    dict(diameter_m, length_m, volume_per_bed_m3, total_bed_volume_m3,
         n_trains, n_beds)
    """
    v_dot = np.asarray(feed_vapor_vol_flow_m3s, dtype=float)

    area_feed = v_dot / superficial_velocity_m_s
    d_1bed = np.sqrt(4.0 * area_feed / np.pi)
    n_trains = np.maximum(1.0, np.ceil((d_1bed / max_bed_diameter_m) ** 2))
    diameter_m = d_1bed / np.sqrt(n_trains)
    length_m = length_to_diameter * diameter_m
    volume_per_bed_m3 = (np.pi / 4.0) * diameter_m**2 * length_m
    n_beds = beds_per_train * n_trains
    total_bed_volume_m3 = n_beds * volume_per_bed_m3

    return {"diameter_m": diameter_m, "length_m": length_m,
            "volume_per_bed_m3": volume_per_bed_m3,
            "total_bed_volume_m3": total_bed_volume_m3,
            "n_trains": n_trains, "n_beds": n_beds}


def psa_adsorbent_inventory_kg(total_bed_volume_m3,
                                bulk_density_kg_m3=PSA_ADSORBENT_BULK_DENSITY_KG_M3):
    """PSA 흡착제 초기충전 질량 [kg] = 총 흡착탑 부피 * 벌크밀도."""
    return np.asarray(total_bed_volume_m3, dtype=float) * bulk_density_kg_m3


def psa_adsorbent_initial_cost_usd(adsorbent_kg,
                                    price_usd_per_kg=PSA_ADSORBENT_PRICE_USD_PER_KG):
    """흡착제 초기충전 비용 [US$] (별도 CAPEX 라인). CEPCI 보정 대상 아님(현재가 기준)."""
    return np.asarray(adsorbent_kg, dtype=float) * price_usd_per_kg


def psa_adsorbent_annual_replacement_cost_usd(
        adsorbent_initial_cost_usd,
        annual_fraction=PSA_ADSORBENT_ANNUAL_REPLACEMENT_FRACTION):
    """연간 흡착제 재생/교체 비용 [US$/year] (별도 OPEX 라인). Lee et al. (2021) 근거."""
    return np.asarray(adsorbent_initial_cost_usd, dtype=float) * annual_fraction


# =====================================================================
# 4. 반응기 부피 (Turton capacity_param = volume_m3, 화염가열로는 duty_kw)
# =====================================================================

def reactor_volume_from_residence_time(vol_flow_m3s, residence_time_s):
    """
    체류시간 기반 반응기 부피 [m3].

        V_reactor = V_dot * tau

    residence_time_s는 촉매 종류/공정 조건에 따라 문헌값을 확보해 지정해야 한다
    (이 함수는 값을 받아 계산만 한다). GHSV로 주어진 경우는 아래
    reactor_volume_from_ghsv()를 쓰는 것이 낫다.
    """
    v = np.asarray(vol_flow_m3s, dtype=float)
    tau = np.asarray(residence_time_s, dtype=float)
    return v * tau


# ---------------------------------------------------------------------
# GHSV 기반 WGS 반응기(촉매층) 부피
# ---------------------------------------------------------------------
#
# 개질기(P1의 R1/R2 등)는 개별 반응기로 코스팅하지 않는다. 실제 SMR에서
# 개질관은 연소 노(furnace) 안에 들어 있으므로 노 하나로 통합해
# fired_heater_nonreactive(|Q_BURNER| 기준, burner_fired_heater_capacity_kw 참조)로
# 코스팅한다. 여기서 부피를 산정하는 대상은 WGS 반응기(HT/LT)뿐이다.
#
# GHSV 출처: Lee, S. et al. (2021). Scenario-Based Techno-Economic Analysis of
#   Steam Methane Reforming. Applied Sciences 11(13), 6021, Table 1.
#     - HT-WGS  : 1.0  m3/h per kg-cat   (Lee의 원출처 [33] Haryanto et al. 2009)
#     - LT-WGS  : 6000 m3/h per m3-cat   (Lee의 원출처 [34]) -> = 6000 /h, 부피기준
#     - 개질기   : 2.0  m3/h per kg-cat  -> 개질기는 노로 통합 코스팅하므로 미사용
#
# GHSV의 표준 정의(Fogler; 촉매공학 교재)는 "표준상태(0 degC, 1 atm) 기준 부피유량
# / 촉매 부피". 대리모델 VolFlow_R*는 반응기 운전 T,P의 실제 체적유량이므로,
# 표준상태로 환산(이상기체)한 뒤 GHSV로 나눈다 (ghsv_basis="STP", 기본).
#
# 한계(논문에 명시): HT-WGS는 질량기준·LT-WGS는 부피기준 GHSV라 두 값이 서로
# 다른 관례로 보고된 것을 그대로 인용했다. 그 결과 케이스에 따라 LT-WGS 촉매층
# 부피가 HT-WGS보다 작게 나올 수 있는데(실제 SMR은 보통 LT가 더 큼), 이는
# 원출처 GHSV 관례 차이에서 오는 것으로 방법 자체의 한계다. 대리모델 예측값을
# 일관된 공개 상관식으로 변환하는 것이 목적이므로 값은 출처대로 둔다.
STANDARD_T_K = 273.15
STANDARD_P_BAR = 1.01325

GHSV_HT_WGS_M3H_PER_KG_CAT = 1.0
GHSV_LT_WGS_M3H_PER_M3_CAT = 6000.0
# 촉매 벌크밀도 [kg/m3] (촉매 질량/비용 산정용).
#   HT-WGS Fe-Cr : ~1200 (통상 1100~1400, 중앙값)
#   LT-WGS Cu-Zn : ~1300 (통상 1200~1400)
# 출처: 상용 shift 촉매 벤더자료 근사. 정확값 확보 시 교체.
BULK_DENSITY_HT_WGS_KG_M3 = 1200.0
BULK_DENSITY_LT_WGS_KG_M3 = 1300.0
# 촉매층 -> 용기 부피 여유계수 (입구 분산기/출구 집수부/지지격자 공간).
REACTOR_VESSEL_ALLOWANCE = 1.25
# shift 촉매 단가 [US$/kg] (초기충전 CAPEX / 재생 OPEX 산정용). Fe-Cr~10, Cu-Zn~25
# 근사. 재생: Lee et al.(2021) 각주 - 3개월마다 10%를 원가 40%로 -> 0.16/yr.
CATALYST_PRICE_HT_WGS_USD_PER_KG = 10.0
CATALYST_PRICE_LT_WGS_USD_PER_KG = 25.0
CATALYST_ANNUAL_REPLACEMENT_FRACTION = 0.16

# 개질 촉매 (Ni/Al2O3). 개질기는 노로 코스팅하지만 촉매는 별도 소모품.
#   GHSV_reformer = 2.0 m3/h per kg-cat (Lee et al. 2021, Appl. Sci. 11, 6021, Table 1)
#   단가 ~20 $/kg, 재생주기 위 shift와 동일 근사.
GHSV_REFORMER_M3H_PER_KG_CAT = 2.0
CATALYST_PRICE_REFORMER_USD_PER_KG = 20.0


def reformer_catalyst_mass_kg(vol_flow_m3s, temp_c, pressure_bar, ghsv_basis="STP"):
    """개질 촉매 질량 [kg] = 표준상태 환산 유량 [m3/h] / GHSV_reformer."""
    if str(ghsv_basis).upper() == "STP":
        v_ref = standard_volumetric_flow_m3h(vol_flow_m3s, temp_c, pressure_bar)
    else:
        v_ref = np.asarray(vol_flow_m3s, dtype=float) * 3600.0
    return v_ref / GHSV_REFORMER_M3H_PER_KG_CAT


def standard_volumetric_flow_m3h(vol_flow_m3s, temp_c, pressure_bar):
    """운전조건 체적유량 [m3/s] -> 표준상태(0 degC, 1 atm) 체적유량 [m3/h] (이상기체)."""
    v_act_m3h = np.asarray(vol_flow_m3s, dtype=float) * 3600.0
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    p_bar = np.asarray(pressure_bar, dtype=float)
    return v_act_m3h * (p_bar / STANDARD_P_BAR) * (STANDARD_T_K / t_k)


def wgs_reactor_sizing(vol_flow_m3s, temp_c, pressure_bar, kind,
                        ghsv_basis="STP", vessel_allowance=REACTOR_VESSEL_ALLOWANCE):
    """
    GHSV 기반 WGS 반응기(단열 충전층) 사이징.

        V_ref[m3/h] = 표준상태 환산 체적유량 (ghsv_basis="STP", 기본)
                      또는 운전조건 그대로     (ghsv_basis="ACTUAL")
        HT-WGS : m_cat[kg]   = V_ref / GHSV_mass ;  V_cat[m3] = m_cat / bulk_density(HT)
        LT-WGS : V_cat[m3]   = V_ref / GHSV_vol
                 m_cat[kg]   = V_cat * bulk_density(LT)
        V_vessel = V_cat * vessel_allowance

    Parameters
    ----------
    vol_flow_m3s : 반응기 통과 체적유량 [m3/s] (대리모델 VolFlow_R*, 운전조건).
    temp_c, pressure_bar : 반응기 운전 온도/압력 (표준상태 환산용, GNN 예측값).
    kind : "HT_WGS" 또는 "LT_WGS".
    ghsv_basis : "STP"(기본, 표준 GHSV 정의) 또는 "ACTUAL".

    비용 연결
        V_vessel_m3 를 turton_costing.process_vessel_vertical(부피 m3)로 코스팅
        (충전층 단열반응기 = 촉매가 채워진 수직 압력용기; reactor_jacketed_agitated는
        교반/재킷형이라 부적합). 촉매 초기충전/재생 비용은 catalyst_* 로 별도.

    Returns
    -------
    dict(v_catalyst_m3, v_vessel_m3, catalyst_mass_kg)
    """
    if str(ghsv_basis).upper() == "STP":
        v_ref_m3h = standard_volumetric_flow_m3h(vol_flow_m3s, temp_c, pressure_bar)
    else:
        v_ref_m3h = np.asarray(vol_flow_m3s, dtype=float) * 3600.0

    k = str(kind).upper().replace("-", "_")
    if k in ("HT_WGS", "HTWGS"):
        mass_cat_kg = v_ref_m3h / GHSV_HT_WGS_M3H_PER_KG_CAT
        v_cat_m3 = mass_cat_kg / BULK_DENSITY_HT_WGS_KG_M3
    elif k in ("LT_WGS", "LTWGS"):
        v_cat_m3 = v_ref_m3h / GHSV_LT_WGS_M3H_PER_M3_CAT
        mass_cat_kg = v_cat_m3 * BULK_DENSITY_LT_WGS_KG_M3
    else:
        raise ValueError(f"kind must be HT_WGS or LT_WGS, got {kind!r}")

    return {"v_catalyst_m3": v_cat_m3,
            "v_vessel_m3": v_cat_m3 * vessel_allowance,
            "catalyst_mass_kg": mass_cat_kg}


def catalyst_initial_cost_usd(catalyst_mass_kg, price_usd_per_kg):
    """촉매 초기충전 비용 [US$] (별도 CAPEX 라인, 현재가 기준)."""
    return np.asarray(catalyst_mass_kg, dtype=float) * price_usd_per_kg


def catalyst_annual_replacement_cost_usd(
        catalyst_initial_cost_usd,
        annual_fraction=CATALYST_ANNUAL_REPLACEMENT_FRACTION):
    """연간 촉매 재생/교체 비용 [US$/year] (별도 OPEX 라인). Lee et al. (2021) 근거."""
    return np.asarray(catalyst_initial_cost_usd, dtype=float) * annual_fraction


# 하위호환: 기존 이름 유지
def reactor_volume_from_ghsv(vol_flow_m3s, temp_c, pressure_bar, kind, ghsv_basis="STP",
                              bulk_density_kg_m3=None):
    """[deprecated] wgs_reactor_sizing()['v_vessel_m3']를 쓸 것."""
    return wgs_reactor_sizing(vol_flow_m3s, temp_c, pressure_bar, kind,
                              ghsv_basis)["v_vessel_m3"]


# ---------------------------------------------------------------------
# 버너 / 개질 노(furnace) 용량
# ---------------------------------------------------------------------
#
# BURNER 블록(P1,P3,P4,P9,P10; P6는 R3_BURNER)은 개질 노의 연소부다.
# "버너 = 노 1개로 통합" 방침(사용자 결정 2026-09-02): 개질기 반응기(R1/R2)는
# 별도 코스팅하지 않고, 노 전체를 fired_heater_nonreactive로 코스팅한다.
#
# 대리모델의 Q_BURNER는 연소 블록의 엔탈피 변화 = "연소 발열량"이다(단위 Watt,
# 부호 음수 확인됨). Turton fired_heater 상관식의 용량변수는 "공정이 흡수하는
# 열부하"이므로, 노 열효율 eta_furnace를 곱해 흡수듀티로 환산한다.
#   duty_absorbed = eta_furnace * |Q_BURNER|
# eta_furnace 기본 0.88: 대류부 열회수를 포함한 개질로 통상 열효율(0.85~0.92).
#   출처: Turton et al. 5th ed. §11 fired heater; SMR 개질로 설계 통상값.
#
# Turton fired_heater_nonreactive 용량범위는 1,000~100,000 kW (=100 MW).
# 1번 공정 흡수듀티는 상한을 넘는 케이스가 많으므로(Q_BURNER 중앙값 ~153 MW,
# 최대 ~266 MW), 동일 노 N기 병렬로 나눠 각 <=100 MW가 되게 한 뒤 코스팅한다
# (대형 SMR 개질로는 실제로 복수 셀/패스로 구성).
FIRED_HEATER_MAX_DUTY_KW = 100000.0
FURNACE_THERMAL_EFFICIENCY = 0.88


def burner_fired_heater_capacity_kw(burner_combustion_duty_kw,
                                     furnace_thermal_efficiency=FURNACE_THERMAL_EFFICIENCY,
                                     split_into_max_units=True):
    """
    BURNER 연소 발열량 [kW](부호 무관, 절대값)를 fired_heater_nonreactive
    용량변수(공정 흡수 열부하 [kW])로 변환한다.

        duty_absorbed = furnace_thermal_efficiency * |Q_BURNER|

    split_into_max_units=True 면 Turton 상한(100 MW) 초과 시 동일 노를
    ceil(duty/100MW) 기로 나눈 "1기당 흡수듀티"와 "기수"를 반환한다.
    비용 연결: per_unit_duty_kw 로 turton_costing.purchased_cost_2001usd(
    "fired_heater_nonreactive", ...) 계산 후 n_units배.

    Returns
    -------
    dict(per_unit_duty_kw, n_units, total_absorbed_duty_kw, combustion_duty_kw)
    """
    q_comb = np.abs(np.asarray(burner_combustion_duty_kw, dtype=float))
    q_abs = furnace_thermal_efficiency * q_comb
    if not split_into_max_units:
        return {"per_unit_duty_kw": q_abs, "n_units": np.ones_like(q_abs),
                "total_absorbed_duty_kw": q_abs, "combustion_duty_kw": q_comb}
    n_units = np.maximum(np.ceil(np.maximum(q_abs / FIRED_HEATER_MAX_DUTY_KW, 1e-9)), 1.0)
    return {"per_unit_duty_kw": q_abs / n_units, "n_units": n_units,
            "total_absorbed_duty_kw": q_abs, "combustion_duty_kw": q_comb}
