"""
properties.py
GNN 대리모델 출력(T, P, 7성분 몰분율, 질량유량)으로부터 경제성/공정지표 계산에 필요한
물성을 재구성하는 모듈. GA_코드구축_프롬프트.md의 "모듈 1"에 해당한다.

계산 항목 (요청 문서의 1~7번 순서 그대로 구현)
    1. 평균분자량 M_mix = sum(x_i * M_i)
    2. 총 몰유량 n_dot = m_dot / M_mix
    3. 성분별 몰유량 n_dot_i = n_dot * x_i
    4. 상 판정 (기상 / 액상), Antoine식 + Raoult 법칙 수준의 간이 기액평형
    5. 밀도 rho (기상은 이상기체, 액상은 Kell(1975) 물 밀도 상관식)
    6. 체적유량 V_dot = m_dot / rho
    7. 비엔탈피 h(T, P, x) (이상기체 Shomate식 + 응축잠열)

설계 원칙
    - 모든 함수는 NumPy 배열 입력을 받아 N개 스트림을 한 번에 벡터화 처리한다.
    - 입력 단위는 대리모델 원출력과 동일하게 T[degC], P[bar], m_dot[kg/h]를 그대로 받는다.
      함수 내부에서는 SI 단위(K, Pa, kg/s)로 변환해 계산하고, 반환값은 각 함수의
      docstring에 명시된 단위로 돌려준다.
    - 질량유량이 0에 가까운 비활성 스트림은 조성이 정의되지 않으므로 모든 함수에서
      0/NaN으로 안전하게 처리한다 (0으로 나누기 금지).

화학종 순서는 프로젝트 전체에서 고정
    H2O, H2, CH4, CO2, CO, O2, N2
"""

from __future__ import annotations

import numpy as np

# =====================================================================
# 0. 단위 변환 유틸
# =====================================================================

def celsius_to_kelvin(t_c):
    """degC -> K"""
    return np.asarray(t_c, dtype=float) + 273.15


def bar_to_pa(p_bar):
    """bar -> Pa"""
    return np.asarray(p_bar, dtype=float) * 1.0e5


def kg_per_h_to_kg_per_s(m_dot_kgh):
    """kg/h -> kg/s"""
    return np.asarray(m_dot_kgh, dtype=float) / 3600.0


def kg_per_s_to_kg_per_h(m_dot_kgs):
    """kg/s -> kg/h"""
    return np.asarray(m_dot_kgs, dtype=float) * 3600.0


# =====================================================================
# 1. 화학종 상수 테이블
# =====================================================================

# 고정 성분 순서 (프로젝트 전체 공통)
SPECIES = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]

# 분자량 [g/mol] = [kg/kmol]. 출처 IUPAC 표준원자량 기반 통상값.
MOLAR_MASS = {  # kg/kmol
    "H2O": 18.015,
    "H2": 2.016,
    "CH4": 16.043,
    "CO2": 44.010,
    "CO": 28.010,
    "O2": 32.000,
    "N2": 28.014,
}
MW_VEC = np.array([MOLAR_MASS[s] for s in SPECIES])  # kg/kmol, SPECIES 순서 고정

R_GAS = 8.314462618  # J/(mol*K) = kJ/(kmol*K), 범용기체상수

# 표준생성엔탈피 [kJ/mol], 298.15 K, 1 bar, 원소 기준 (NIST/CODATA 통상값)
ENTHALPY_FORMATION_298 = {  # kJ/mol
    "H2O": -241.826,   # 기상 기준 (수증기)
    "H2": 0.0,
    "CH4": -74.87,
    "CO2": -393.522,
    "CO": -110.527,
    "O2": 0.0,
    "N2": 0.0,
}

# 저위발열량 LHV [MJ/kg] (연료가치 계산용, GA_코드구축_프롬프트.md 명시값)
LHV_H2 = 120.0   # MJ/kg
LHV_CH4 = 50.0   # MJ/kg

# 물의 표준증발잠열 [kJ/mol], 100 degC, 1 atm 기준 통상값
WATER_HEAT_OF_VAPORIZATION_373K = 40.65  # kJ/mol

# 이상기체 비열 Cp [J/(mol*K)] 다항식 계수 (Shomate 형태 간략화, NIST Webbook 근사)
# Cp/R = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4  (T in K, 300~1500K 근사 범위)
# 참고용 근사 계수. 정밀 반응공정 열역학이 필요하면 NASA 7항 다항식으로 교체 권장.
SHOMATE_CP_COEFF = {
    # a0,     a1,        a2,          a3,           a4
    "H2O": (4.395, -4.186e-3, 1.405e-5, -1.564e-8, 0.632e-11),
    # 2026-09-01 수정: 기존 a3=0.083e-6 계수는 623K 이상에서 발산해 1000K에서
    # Cp를 약 720 J/(mol*K)로 계산했다 (실제값 30.2, 24배 과대). NIST WebBook
    # H2(g) JANAF 표(298~1000K, 8개 점)를 np.polyfit으로 2차 재적합해 대체했다.
    # 재적합 최대오차 0.124 J/(mol*K) (298K 기준 28.95 vs 실측 28.84).
    "H2":  (3.4621261, 2.6948142e-5, 1.3580677e-7, 0.0, 0.0),
    "CH4": (1.702, 9.081e-3, -2.164e-6, 0.0, 0.0),
    "CO2": (3.259, 1.356e-3, 1.502e-5, -2.374e-8, 1.056e-11),
    "CO":  (3.912, -3.913e-3, 1.182e-5, -1.302e-8, 0.515e-11),
    "O2":  (3.630, -1.794e-3, 6.582e-6, -6.010e-9, 1.901e-12),
    "N2":  (3.539, -0.261e-3, 0.007e-6, 0.157e-9, -0.099e-12),
}

# Antoine식 계수, 물 (NIST Webbook, log10(P/bar) = A - B/(T[K]+C))
# 구간별로 다르며, 값은 NIST가 원 문헌(Bridgeman&Aldrich 1964, Liu&Lindsay 1970)에서
# 재계산해 게재한 표준값이다.
WATER_ANTOINE_RANGES = [
    # (T_min_K, T_max_K, A, B, C)
    (273.0, 303.0, 5.40221, 1838.675, -31.737),
    (304.0, 333.0, 5.20389, 1733.926, -39.485),
    (334.0, 363.0, 5.07680, 1659.793, -45.854),
    (363.0, 573.0, 3.55959, 643.748, -198.043),  # 379-573K 계수를 363K까지 확장 적용
]


def water_saturation_pressure_bar(t_k):
    """
    물의 포화증기압 [bar]을 Antoine식으로 계산한다.
    입력 온도 범위를 벗어나면 가장 가까운 구간의 계수를 그대로 사용한다
    (외삽이므로 정밀도가 떨어질 수 있음을 감안할 것).

    log10(P/bar) = A - B / (T[K] + C)
    출처 NIST Chemistry WebBook, Water, Antoine Equation Parameters
    """
    t_k = np.asarray(t_k, dtype=float)
    p_sat = np.full_like(t_k, np.nan, dtype=float)

    ranges = WATER_ANTOINE_RANGES
    for lo, hi, a, b, c in ranges:
        mask = (t_k >= lo) & (t_k <= hi) & np.isnan(p_sat)
        if np.any(mask):
            p_sat[mask] = 10.0 ** (a - b / (t_k[mask] + c))

    # 범위를 벗어난 값은 가장 가까운 구간으로 외삽 처리
    below = t_k < ranges[0][0]
    if np.any(below):
        lo, hi, a, b, c = ranges[0]
        p_sat[below] = 10.0 ** (a - b / (t_k[below] + c))
    above = t_k > ranges[-1][1]
    if np.any(above):
        lo, hi, a, b, c = ranges[-1]
        p_sat[above] = 10.0 ** (a - b / (t_k[above] + c))

    return p_sat


# =====================================================================
# 2. 평균분자량, 몰유량 (요청 문서 1~3번)
# =====================================================================

def average_molar_mass(mass_fractions_or_mole_fractions, basis="mole"):
    """
    평균분자량 M_mix [kg/kmol] 계산.

    GNN 대리모델의 Frac_* 출력은 몰분율(mole fraction)이다 (softmax 출력, 합이 1).
    basis="mole" 이면 표준 몰분율 가중 평균식을 사용한다.

        M_mix = sum_i (x_i * M_i)

    Parameters
    ----------
    mass_fractions_or_mole_fractions : (N, 7) array
        SPECIES 순서(H2O,H2,CH4,CO2,CO,O2,N2)의 몰분율. 각 행의 합이 1이어야 한다.
    basis : {"mole"}
        현재는 몰분율 입력만 지원 (대리모델 출력 스키마 기준).

    Returns
    -------
    M_mix : (N,) array, kg/kmol
    """
    x = np.asarray(mass_fractions_or_mole_fractions, dtype=float)
    if basis != "mole":
        raise NotImplementedError("현재는 mole fraction 입력만 지원합니다.")
    return x @ MW_VEC


def total_molar_flow(mass_flow_kgh, m_mix_kg_per_kmol):
    """
    총 몰유량 n_dot [kmol/h] = m_dot[kg/h] / M_mix[kg/kmol]

    질량유량이 0에 가까운 비활성 스트림(전체의 약 5.6%)은 0을 반환한다
    (0으로 나누기 방지, GA_코드구축_프롬프트.md 결측 처리 요구사항 반영).
    """
    m_dot = np.asarray(mass_flow_kgh, dtype=float)
    m_mix = np.asarray(m_mix_kg_per_kmol, dtype=float)

    n_dot = np.zeros_like(m_dot)
    active = m_dot > 1e-8  # 원본 데이터 정책: 유량은 정확히 0 이거나 1 kg/h 초과
    n_dot[active] = m_dot[active] / m_mix[active]
    return n_dot


def component_molar_flows(n_dot_total_kmolh, mole_fractions):
    """
    성분별 몰유량 n_dot_i [kmol/h] = n_dot_total * x_i

    Returns
    -------
    (N, 7) array, SPECIES 순서
    """
    n_dot = np.asarray(n_dot_total_kmolh, dtype=float).reshape(-1, 1)
    x = np.asarray(mole_fractions, dtype=float)
    return n_dot * x


# =====================================================================
# 3. 상 판정 (요청 문서 4번)
# =====================================================================

# 상 판정 결과 코드
PHASE_VAPOR = 0
PHASE_LIQUID = 1
PHASE_INACTIVE = -1  # 유량이 0인 비활성 스트림


def determine_phase(temp_c, pres_bar, mole_fractions, mass_flow_kgh,
                     water_liquid_threshold=0.98):
    """
    간이 기액상 판정 (Antoine식 + Raoult 법칙 수준).

    설계 배경 (GA_코드구축_프롬프트.md 원문 요구사항)
        우리 데이터에는 몰분율 1.000에 가까운 순수 물 스트림(응축수)이 다수 있다.
        상 판정을 생략하면 체적유량이 30~500배 오차가 나거나 음수가 나올 수 있으므로,
        정밀도보다 "상을 틀리지 않는 것"을 우선한다.

    판정 로직
        1. 질량유량이 0에 가까운 비활성 스트림은 PHASE_INACTIVE.
        2. H2, CH4, CO, O2, N2는 이 공정의 운전 T, P 범위에서 임계점보다 훨씬 위에
           있는 초임계 기체로 간주해 항상 기상 성분으로 취급한다 (Raoult 법칙 대상에서 제외).
        3. H2O 몰분율이 매우 높은 스트림(water_liquid_threshold 이상, 기본 0.98)에
           대해서만 Antoine식으로 물의 포화증기압 P_sat(T)을 구해 실제 압력 P와 비교한다.
           P > P_sat(T) 이면 그 온도에서 응축이 일어날 수 있는 상태이므로 액상으로 판정한다.
        4. H2O 비중이 낮은 스트림(개질가스, 배가스 등)은 항상 기상으로 판정한다.
           (CO2도 이 공정 T, P 범위에서는 초임계이므로 별도 상 판정 대상에서 제외)

    이 판정은 "정밀 VLE"가 아니라 "물 응축 여부를 놓치지 않기 위한 안전장치"이다.
    두 상이 실제로 공존하는 스트림(기액 공존)은 액상으로 보수적으로 분류해
    체적유량을 과대추정하지 않도록 한다.

    Returns
    -------
    phase : (N,) int array, PHASE_VAPOR / PHASE_LIQUID / PHASE_INACTIVE
    """
    t_k = celsius_to_kelvin(temp_c)
    p_bar = np.asarray(pres_bar, dtype=float)
    x = np.asarray(mole_fractions, dtype=float)
    m_dot = np.asarray(mass_flow_kgh, dtype=float)

    n = len(m_dot)
    phase = np.full(n, PHASE_VAPOR, dtype=int)

    inactive = m_dot <= 1e-8
    phase[inactive] = PHASE_INACTIVE

    x_h2o = x[:, SPECIES.index("H2O")]
    water_dominant = (x_h2o >= water_liquid_threshold) & (~inactive)

    if np.any(water_dominant):
        p_sat = water_saturation_pressure_bar(t_k[water_dominant])
        is_condensed = p_bar[water_dominant] > p_sat
        idx = np.where(water_dominant)[0]
        phase[idx[is_condensed]] = PHASE_LIQUID

    return phase


# =====================================================================
# 4. 밀도 (요청 문서 5번)
# =====================================================================

def vapor_density_ideal_gas(temp_c, pres_bar, m_mix_kg_per_kmol, z_factor=1.0):
    """
    기상 밀도 [kg/m3], 이상기체 상태방정식.

        rho = P * M_mix / (Z * R * T)

    z_factor를 1이 아닌 배열로 넘기면 압축인자 보정이 가능하도록 인터페이스를
    분리해 두었다 (추후 Peng-Robinson 등 실제기체 상태방정식으로 교체 가능).
    이 공정의 운전압력이 최대 약 40 bar 수준이므로 1차 근사로는 이상기체로 충분하다.
    """
    t_k = celsius_to_kelvin(temp_c)
    p_pa = bar_to_pa(pres_bar)
    m_mix = np.asarray(m_mix_kg_per_kmol, dtype=float)
    z = np.asarray(z_factor, dtype=float)

    # R_GAS 단위 J/(mol*K) = kJ/(kmol*K) 이므로 P[Pa]*M[kg/kmol] / (Z*R*T) 는
    # Pa*kg/kmol / (kJ/(kmol*K)*K) = Pa*kg/kmol / (kJ/kmol) 이 되어 단위를 맞추려면
    # R을 J/(kmol*K)로 스케일해야 한다. R_GAS[J/(mol*K)] * 1000 = R[J/(kmol*K)].
    r_j_per_kmol_k = R_GAS * 1000.0
    return p_pa * m_mix / (z * r_j_per_kmol_k * t_k)


def liquid_water_density(temp_c):
    """
    액상(응축수) 밀도 [kg/m3], Kell(1975) 물 밀도 상관식, 1 atm 기준.
    유효범위 -30 ~ 150 degC. 압력에 의한 액체 밀도 변화는 이 압력범위(~수십 bar)에서
    무시할 수 있는 수준이라 별도 보정하지 않는다.

        rho(T) = (((((a*T + b)*T + c)*T + d)*T + e)*T + f) / (1 + g*T),  T in degC

    출처 Kell, G. S. (1975), J. Chem. Eng. Data 20(1), 97-105.
    """
    t = np.asarray(temp_c, dtype=float)
    a = -2.8054253e-10
    b = 1.0556302e-7
    c = -4.6170461e-5
    d = -7.9870401e-3
    e = 16.945176
    f = 999.83952
    g = 0.01687985

    numerator = ((((a * t + b) * t + c) * t + d) * t + e) * t + f
    denominator = 1.0 + g * t
    return numerator / denominator


def mixture_density(temp_c, pres_bar, mole_fractions, m_mix_kg_per_kmol, phase):
    """
    상 판정 결과에 따라 기상은 이상기체식, 액상은 Kell 물 밀도식을 적용한
    혼합 밀도 [kg/m3]를 반환한다. 비활성 스트림은 NaN을 반환한다 (0으로 나누기 방지).
    """
    n = len(np.asarray(temp_c))
    rho = np.full(n, np.nan, dtype=float)

    is_vapor = phase == PHASE_VAPOR
    if np.any(is_vapor):
        rho[is_vapor] = vapor_density_ideal_gas(
            np.asarray(temp_c)[is_vapor],
            np.asarray(pres_bar)[is_vapor],
            np.asarray(m_mix_kg_per_kmol)[is_vapor],
        )

    is_liquid = phase == PHASE_LIQUID
    if np.any(is_liquid):
        # 액상으로 판정된 스트림은 water_liquid_threshold 이상 H2O이므로
        # 순수 물 밀도식으로 근사한다 (혼합액체 물성은 이 공정 범위에서 불필요).
        rho[is_liquid] = liquid_water_density(np.asarray(temp_c)[is_liquid])

    return rho


# =====================================================================
# 5. 체적유량 (요청 문서 6번, 이번 요청의 핵심 산출물)
# =====================================================================

def volumetric_flow(mass_flow_kgh, density_kg_per_m3):
    """
    체적유량 V_dot [m3/h] = m_dot[kg/h] / rho[kg/m3]

    밀도가 NaN이거나 0인 비활성 스트림은 0을 반환한다 (0으로 나누기 방지).
    """
    m_dot = np.asarray(mass_flow_kgh, dtype=float)
    rho = np.asarray(density_kg_per_m3, dtype=float)

    v_dot = np.zeros_like(m_dot)
    valid = (~np.isnan(rho)) & (rho > 1e-8) & (m_dot > 1e-8)
    v_dot[valid] = m_dot[valid] / rho[valid]
    return v_dot


def volumetric_flow_m3_per_s(mass_flow_kgh, density_kg_per_m3):
    """체적유량 [m3/s]. 압축기/펌프 동력 계산 등 SI 단위가 필요한 곳에서 사용."""
    return volumetric_flow(mass_flow_kgh, density_kg_per_m3) / 3600.0


# =====================================================================
# 6. 비엔탈피 (요청 문서 7번)
# =====================================================================
#
# 기준 상태 (반드시 이 기준으로 문서화할 것)
#     298.15 K, 1 bar, 각 화학종의 표준 생성엔탈피(원소 기준)를 0점으로 삼는다.
#     즉 h(298.15K, 1bar) = 표준생성엔탈피(원소 기준 정의). 물은 기상 표준생성엔탈피를
#     0점으로 사용하고, 응축이 일어나는 경우 증발잠열만큼을 추가로 빼서 액상 엔탈피를 만든다.

def component_molar_enthalpy_kj_per_mol(species, temp_k):
    """
    성분 하나의 몰당 비엔탈피 [kJ/mol], 이상기체 기준.

        h_i(T) = dHf298_i + Integral_298^T Cp_i(T') dT'

    Cp는 SHOMATE_CP_COEFF의 4차 다항식(Cp/R = a0+a1T+a2T^2+a3T^3+a4T^4)을 적분한다.
    """
    t = np.asarray(temp_k, dtype=float)
    a0, a1, a2, a3, a4 = SHOMATE_CP_COEFF[species]
    t0 = 298.15

    def cp_integral(tt):
        # Integral (a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4) dT, 결과 단위 K (R 곱 전)
        return (a0 * tt + a1 * tt**2 / 2 + a2 * tt**3 / 3
                + a3 * tt**4 / 4 + a4 * tt**5 / 5)

    delta_integral = cp_integral(t) - cp_integral(t0)  # R 배수 상태, 단위 K
    sensible_kj_per_mol = R_GAS * delta_integral / 1000.0  # J->kJ

    return ENTHALPY_FORMATION_298[species] + sensible_kj_per_mol


def mixture_k_ratio(temp_c, mole_fractions):
    """
    혼합물의 비열비 k = Cp_mix/Cv_mix (이상기체 가정, Cv = Cp - R).

    equipment_sizing.py의 compressor_isentropic_power_kw / turbine_isentropic_power_kw
    는 기본값 k_ratio=1.4(공기 기준)를 쓰지만, 실제로는 스트림 조성에 따라 크게
    달라진다 (예: 25도C 기준 CH4 k=1.311, 공기 k=1.403). 이 함수는 SHOMATE_CP_COEFF
    다항식(component_molar_enthalpy_kj_per_mol과 동일한 데이터)을 그대로 이용해
    각 스트림 조성/온도에 맞는 k값을 구해, 압축기/터빈 동력식에 넣을 수 있게 한다.

        Cp_i(T)/R = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4   (SHOMATE_CP_COEFF)
        Cp_mix(T) = sum_i( x_i * Cp_i(T) )                  [몰분율 가중]
        Cv_mix(T) = Cp_mix(T) - R                            (이상기체)
        k(T)      = Cp_mix(T) / Cv_mix(T)

    2026-09-01 검증 결과 (1번 공정 50케이스, june 데이터, 실측 Density 사용):
        C1(CH4 압축기) 역산효율: k_ratio=1.4 고정 시 0.861±0.012 (실제 가정값 0.75
        대비 +15%p 이상 차이) -> 성분별 k 적용 시 0.788±0.005로 개선 (0.75에 근접).
        C2(FUEL 압축기): 0.777±0.010 -> 0.755±0.003 (사실상 일치).
        C3(AIR 압축기, 원래 공기라 k=1.4에 가까움): 0.751±0.0004로 변화 거의 없음
        (원래도 거의 정확했던 케이스이므로 이 결과는 이 함수가 "공기처럼 k=1.4에
        가까운 조성에서는 원래 값을 그대로 재현한다"는 sanity check 역할도 한다).
        즉, 이 함수가 동력식의 형태 자체가 맞다는 것과, k_ratio 기본값 가정이
        오차의 상당 부분을 차지했다는 것을 함께 뒷받침한다.

    Parameters
    ----------
    temp_c : (N,) array, degC. 등엔트로피 계산 관례상 압축/팽창기 "입구" 온도를 쓴다.
    mole_fractions : (N,7) array, SPECIES 순서, 각 행 합 1

    Returns
    -------
    k_ratio : (N,) array, 무차원. Cp_mix가 R 이하로 내려가는 비정상 입력
        (예: 몰분율이 전부 0인 빈 스트림)에 대해서는 np.nan을 반환한다.
    """
    t_k = celsius_to_kelvin(temp_c)
    t = np.atleast_1d(np.asarray(t_k, dtype=float))
    x = np.atleast_2d(np.asarray(mole_fractions, dtype=float))

    cp_over_r_species = np.zeros((t.shape[0], len(SPECIES)))
    for j, sp in enumerate(SPECIES):
        a0, a1, a2, a3, a4 = SHOMATE_CP_COEFF[sp]
        cp_over_r_species[:, j] = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4

    cp_over_r_mix = np.sum(x * cp_over_r_species, axis=1)  # Cp_mix / R, 무차원
    cv_over_r_mix = cp_over_r_mix - 1.0  # Cv = Cp - R

    k = np.full(cp_over_r_mix.shape, np.nan)
    valid = cv_over_r_mix > 1e-6
    k[valid] = cp_over_r_mix[valid] / cv_over_r_mix[valid]

    if np.isscalar(temp_c) or np.ndim(temp_c) == 0:
        return float(k[0])
    return k


def mixture_specific_enthalpy(temp_c, mole_fractions, phase, m_mix_kg_per_kmol):
    """
    혼합물 비엔탈피 [kJ/kg], 이상기체 Shomate 적분 + 응축 시 잠열 반영.

    기상 스트림
        h_mix = sum_i( x_i * h_i(T) ) / M_mix   [kJ/kg]

    액상(응축수)으로 판정된 스트림
        기상 기준 엔탈피에서 100 degC 기준 물 증발잠열(40.65 kJ/mol)을 추가로 빼서
        근사한다. 응축 온도에 따른 잠열 보정(Clausius-Clapeyron 등)은 하지 않는
        1차 근사이며, 정밀 계산이 필요하면 IAPWS-IF97 등으로 교체할 것.

    Returns
    -------
    h_mix : (N,) array, kJ/kg
    """
    t_k = celsius_to_kelvin(temp_c)
    x = np.asarray(mole_fractions, dtype=float)
    m_mix = np.asarray(m_mix_kg_per_kmol, dtype=float)
    n = x.shape[0]

    h_mix = np.full(n, np.nan, dtype=float)
    valid = m_mix > 1e-8
    if not np.any(valid):
        return h_mix

    h_species = np.zeros((n, len(SPECIES)))
    for j, sp in enumerate(SPECIES):
        h_species[:, j] = component_molar_enthalpy_kj_per_mol(sp, t_k)

    h_molar_mix_kj_per_kmol = np.sum(x * h_species, axis=1) * 1000.0  # kJ/mol -> kJ/kmol
    h_mass_vapor = np.full(n, np.nan)
    h_mass_vapor[valid] = h_molar_mix_kj_per_kmol[valid] / m_mix[valid]  # kJ/kg

    h_mix[valid] = h_mass_vapor[valid]

    is_liquid = (phase == PHASE_LIQUID) & valid
    if np.any(is_liquid):
        x_h2o = x[:, SPECIES.index("H2O")]
        latent_kj_per_kmol = (WATER_HEAT_OF_VAPORIZATION_373K * 1000.0) * x_h2o
        h_mix[is_liquid] = h_mix[is_liquid] - (
            latent_kj_per_kmol[is_liquid] / m_mix[is_liquid]
        )

    return h_mix


# =====================================================================
# 7. 일괄 처리 파이프라인
# =====================================================================

def reconstruct_stream_properties(temp_c, pres_bar, mole_fractions, mass_flow_kgh):
    """
    GNN 대리모델의 10D 원출력을 받아 물성 재구성 전 항목을 한 번에 계산한다.

    Parameters
    ----------
    temp_c : (N,) array, degC
    pres_bar : (N,) array, bar
    mole_fractions : (N,7) array, SPECIES 순서, 각 행 합 1
    mass_flow_kgh : (N,) array, kg/h

    Returns
    -------
    dict of (N,) or (N,7) arrays:
        m_mix_kg_per_kmol, n_dot_total_kmolh, n_dot_species_kmolh (N,7),
        phase, density_kg_m3, vol_flow_m3h, vol_flow_m3s, enthalpy_kj_kg
    """
    x = np.asarray(mole_fractions, dtype=float)

    m_mix = average_molar_mass(x)
    n_dot = total_molar_flow(mass_flow_kgh, m_mix)
    n_dot_i = component_molar_flows(n_dot, x)
    phase = determine_phase(temp_c, pres_bar, x, mass_flow_kgh)
    rho = mixture_density(temp_c, pres_bar, x, m_mix, phase)
    v_dot_h = volumetric_flow(mass_flow_kgh, rho)
    v_dot_s = v_dot_h / 3600.0
    h_mix = mixture_specific_enthalpy(temp_c, x, phase, m_mix)

    return {
        "m_mix_kg_per_kmol": m_mix,
        "n_dot_total_kmolh": n_dot,
        "n_dot_species_kmolh": n_dot_i,
        "phase": phase,
        "density_kg_m3": rho,
        "vol_flow_m3h": v_dot_h,
        "vol_flow_m3s": v_dot_s,
        "enthalpy_kj_kg": h_mix,
    }
