"""
turton_costing.py
Turton et al., "Analysis, Synthesis, and Design of Chemical Processes"
(module costing technique)을 이용한 장비비/CAPEX 산정 엔진.

지난 논의에서 확정한 원칙
    - 여러 경제성평가 방법론 중 Turton method를 대표 방법으로 채택한다.
    - CEPCI는 2025년 기준(가정값 800)으로 보정한다. Chemical Engineering지가
      2024년 9월부터 CEPCI를 유료 구독제로 전환해 공개 최신값을 독립적으로
      확인할 수 없었다. 마지막으로 확인 가능한 공개값은 2023년 연평균 797.9,
      2024년 6월(월간) 798.8이며, 800은 이 추세선을 2025년으로 연장한 가정값이다.
      랩실에 구독 접근권이 있다면 CEPCI_CURRENT 상수만 실제 값으로 교체하면 된다.

핵심 수식 (Turton 표준, 아래 세 식이 이 모듈의 뼈대)
    1. 구매비(상압, 탄소강 기준)
        log10(C_p0) = K1 + K2*log10(A) + K3*(log10(A))^2
       A는 장비 종류별 용량변수(동력 kW, 면적 m2, 부피 m3 등)이며 아래
       TURTON_EQUIPMENT_TABLE에 등록되어 있다.

    2. 베어모듈비용(설치비 포함, 재질/압력 보정)
        C_BM = C_p0 * F_BM               (F_BM = B1 + B2*F_M*F_P, 장비군에 따라 다름)
       압축기처럼 재질/압력 보정 없이 고정 F_BM을 쓰는 장비군도 있다 (TURTON_FBM_DIRECT).

    3. 총모듈비용/그래스루트 비용
        C_TM  = 1.18 * sum(C_BM)          (예비비 + 계약자수수료 관행계수)
        C_GR  = C_TM + 0.5 * sum(C_BM)    (신설 플랜트 부대설비 관행계수)

    모든 원가는 CEPCI 397(2001년 5~9월 평균)을 기준연도로 하므로, 실제 사용
    시점 비용으로 환산하려면 escalate_cepci()로 보정한다.

중요한 한계 (반드시 문서화하고 랩실에 전달할 것)
    - K1,K2,K3 7종(압축기, 펌프, 열교환기 3종, 화염가열로, 베셀 2종, 반응기, 터빈/
      팽창기)은 복수의 공개 출처(교재 부록 발췌본)로 교차 확인했다. 터빈/팽창기
      (turbine_expander)는 Turton 원서 Table A.1을 직접 확인하지 못해 Lemmens
      (2016, Energies 9(7), 485, Table 3, doi.org/10.3390/en9070485)가 전사한
      'Expander' 행(원출처 Turton et al. 4th ed. Table A.1)을 인용했고, 같은 표의
      Compressor 값이 이 파일 기존 값과 정확히 일치함을 교차확인 근거로 삼았다.
      용량 대역(capacity_range_kw)만은 원서에서 직접 확인하지 못해 압축기 대역을
      잠정 차용했다 (turbine_expander 항목의 note, range_verified 필드 참조).
    - 베어모듈 계수(B1,B2, 재질계수 F_M, 압력계수 F_P)는 일부만 독립적으로
      재확인했다. 확인되지 않은 값은 TURTON_BM_FACTORS/TURTON_FBM_DIRECT에
      VERIFIED=False로 표시해 두었으니, 논문/코드에 쓰기 전 반드시 원서 Table
      A.3, A.4로 대조할 것 (Turton 5th edition 기준). turbine_expander의
      F_BM=2.8은 압축기 값을 유사장비 유추로 잠정 차용한 것으로 특히 대조가
      필요하다.
"""

from __future__ import annotations

import numpy as np

# =====================================================================
# 경제성 기준 가정 (2025-09 확정)
# =====================================================================

CEPCI_BASE_2001 = 397.0     # Turton 원가 데이터의 기준 CEPCI (2001년 5~9월 평균)
CEPCI_CURRENT = 800.0       # 2025년 기준 가정값. 출처 주석 참조 (모듈 docstring)
CEPCI_CURRENT_YEAR_LABEL = "2025년 기준(가정)"

TOTAL_MODULE_FACTOR = 1.18       # 예비비 + 계약자수수료 관행계수 (Turton)
GRASSROOTS_EXTRA_FACTOR = 0.5    # 신설 플랜트 부대설비 관행계수 (Turton)


def escalate_cepci(cost_base_2001, cepci_current=CEPCI_CURRENT,
                    cepci_base=CEPCI_BASE_2001):
    """
    CEPCI 물가보정. Cost_now = Cost_2001기준 * (CEPCI_now / CEPCI_2001)
    """
    return np.asarray(cost_base_2001, dtype=float) * (cepci_current / cepci_base)


# =====================================================================
# 1. Turton 구매비 상관식 테이블 (K1, K2, K3)
# =====================================================================
#
# 각 항목: capacity_unit, capacity_range, (K1,K2,K3)
# 출처: Turton, Bailie, Whiting, Shaeiwitz, "Analysis, Synthesis, and Design
#       of Chemical Processes" Appendix A (Table A.1), 2001년 5~9월 장비업체
#       설문 기준 데이터. 복수의 공개 발췌 자료로 교차 확인한 값이다.

TURTON_EQUIPMENT_TABLE = {
    "compressor_centrifugal": {
        "K1": 2.2897, "K2": 1.3604, "K3": -0.1027,
        "capacity_param": "power_kw",
        "capacity_range_kw": (450, 3000),
        "note": "원심/축류/왕복 압축기 공용 상관식 (Turton Table A.1)",
    },
    "pump_centrifugal": {
        "K1": 3.3892, "K2": 0.0536, "K3": 0.1538,
        "capacity_param": "power_kw",
        "capacity_range_kw": (1, 300),
        "note": "원심펌프 축동력 기준 (Turton Table A.1)",
    },
    "heat_exchanger_floating_head": {
        "K1": 4.8306, "K2": -0.8509, "K3": 0.3187,
        "capacity_param": "area_m2",
        "capacity_range_m2": (10, 1000),
    },
    "heat_exchanger_fixed_tube": {
        "K1": 4.3247, "K2": -0.3030, "K3": 0.1634,
        "capacity_param": "area_m2",
        "capacity_range_m2": (10, 1000),
    },
    "heat_exchanger_u_tube": {
        "K1": 4.1884, "K2": -0.2503, "K3": 0.1974,
        "capacity_param": "area_m2",
        "capacity_range_m2": (10, 1000),
    },
    "fired_heater_nonreactive": {
        "K1": 7.3488, "K2": -1.1666, "K3": 0.2028,
        "capacity_param": "duty_kw",
        "capacity_range_kw": (1000, 100000),
        "note": "개질기(reformer)를 비반응성 화염가열로로 근사할 때 사용",
    },
    "process_vessel_horizontal": {
        "K1": 3.5565, "K2": 0.3776, "K3": 0.0905,
        "capacity_param": "volume_m3",
        "capacity_range_m3": (0.1, 628),
    },
    "process_vessel_vertical": {
        "K1": 3.4974, "K2": 0.4485, "K3": 0.1074,
        "capacity_param": "volume_m3",
        "capacity_range_m3": (0.3, 520),
        "note": "플래시드럼, PSA 베셀, 분리기 등 수직 압력용기 근사에 사용",
    },
    "reactor_jacketed_agitated": {
        "K1": 4.1052, "K2": -0.4680, "K3": -0.0005,
        "capacity_param": "volume_m3",
        "capacity_range_m3": (0.1, 35),
        "note": "WGS 반응기 등 교반/재킷형 반응기 근사 (개질기 자체는 fired_heater 권장)",
    },
    "turbine_expander": {
        "K1": 2.2476, "K2": 1.4965, "K3": -0.1618,
        "capacity_param": "power_kw",
        "capacity_range_kw": (450, 3000),
        "range_verified": False,
        "note": (
            "인접행렬 노드타입 turbine(공정가스/스팀 팽창터빈, 예 T1~T3)에 적용. "
            "K1/K2/K3 자체는 Turton 원서 Table A.1의 'Expander' 항목이며, 아래 2차 "
            "출처로 확인함: Lemmens, S. (2016). Cost Engineering Techniques and Their "
            "Applicability for Cost Estimation of Organic Rankine Cycle Systems. "
            "Energies, 9(7), 485, Table 3. https://doi.org/10.3390/en9070485 "
            "-> 이 논문이 Table 3에서 'Expander' K1=2.2476, K2=1.4965, K3=-0.1618을 "
            "[13] Turton, R.; Bailie, R.C.; Whiting, W.B.; Shaeiwitz, J.A.; "
            "Bhattacharyya, D. (2013), Analysis, Synthesis, and Design of Chemical "
            "Processes, 4th ed., Pearson 인용으로 명시함. 같은 Table 3의 Compressor "
            "행(K1=2.2897,K2=1.3604,K3=-0.1027)이 이 파일의 compressor_centrifugal "
            "값과 정확히 일치해 이 논문의 표 전사 신뢰성을 교차확인함. "
            "capacity_param을 power_kw로 둔 것은 Turton 표에서 압축기 등 회전동력 "
            "장비군이 공통으로 동력을 용량변수로 쓰는 관행을 따른 것이며, 원서에서 "
            "'Expander' 행의 단위 표기를 직접 확인하지는 못했다 (range_verified=False, "
            "capacity_range_kw는 압축기 대역을 잠정 차용한 값이므로 원서 Table A.1로 "
            "재대조 권장). "
            "주의: 이 값을 채택하기 전 DOE/NETL 'Process Equipment Cost Estimation "
            "Final Report'(H.P. Loh, 2002년 1월, osti.gov/servlets/purl/797810/)의 "
            "가스터빈 발전기 세트 원가자료(1,000/50,000/370,000 HP)로 별도 상관식을 "
            "자체 회귀했었으나, 동일 kW대에서 이 Turton Expander식보다 약 3배 높은 "
            "값을 준다는 것을 확인함 (예 745.7kW에서 NETL유도식 $485,369 vs 이 "
            "Turton식 $163,000, 2001 CEPCI 397 기준). 이는 NETL 자료가 연소기/발전기가 "
            "포함된 완전한 가스터빈 발전 패키지 원가인 반면, Turton의 Expander는 "
            "연소기·발전기가 없는 단순 공정가스 팽창(letdown)용 터빈이기 때문으로 "
            "판단되며, 본 SMR 공정의 T1~T3(공정 내부 팽창터빈)에는 장비 범위가 더 "
            "가까운 이 Turton Expander식을 채택함."
        ),
    },
}


def purchased_cost_2001usd(equipment_key, capacity_value):
    """
    Turton 상관식으로 2001년 기준(CEPCI 397) 구매비 [US$]를 계산한다.

        log10(C_p0) = K1 + K2*log10(A) + K3*(log10(A))^2

    capacity_value 단위는 TURTON_EQUIPMENT_TABLE[equipment_key]["capacity_param"]에
    명시된 단위(kW 또는 m2 또는 m3)와 반드시 일치해야 한다.
    용량범위를 벗어나면 경고 취지의 플래그를 함께 반환한다 (외삽 사용 시 주의).
    """
    spec = TURTON_EQUIPMENT_TABLE[equipment_key]
    a = np.asarray(capacity_value, dtype=float)
    a_safe = np.clip(a, 1e-6, None)  # log10(0) 방지

    log_a = np.log10(a_safe)
    log_cp = spec["K1"] + spec["K2"] * log_a + spec["K3"] * log_a**2
    cp0 = 10.0 ** log_cp

    out_of_range = _check_capacity_range(spec, a)
    return cp0, out_of_range


def _check_capacity_range(spec, a):
    for key in ("capacity_range_kw", "capacity_range_m2", "capacity_range_m3"):
        if key in spec:
            lo, hi = spec[key]
            return (a < lo) | (a > hi)
    return np.zeros_like(a, dtype=bool)


def _capacity_range(spec):
    for key in ("capacity_range_kw", "capacity_range_m2", "capacity_range_m3"):
        if key in spec:
            return spec[key]
    return None


def purchased_cost_ranged(equipment_key, capacity_value):
    """
    구매비 [2001 US$]를 계산하되, 용량이 상관식 유효구간을 벗어나면 Turton 관행대로
    "외삽" 대신 다음처럼 처리한다:
      - 상한 초과 : 동일 장비 N기 병렬 (N = ceil(cap / hi)), 각 기 용량 = cap/N 로
                    구간 안에 들어오게 한 뒤 1기 비용 * N.
                    (열교환기=다중 셸, 압축기=다중 케이싱, 베셀=병렬 드럼 등 물리적으로도 타당)
      - 하한 미만 : 용량을 하한값으로 클램프 (소형 장비의 최소 구매비. 10 m2 미만
                    열교환기라도 10 m2 가격은 든다).
    스칼라 capacity_value 기준. 배열이면 각 원소에 동일 규칙(단, N은 원소별로 다를 수 있음).

    Returns
    -------
    (cp0_per_unit_2001usd, n_units, per_unit_capacity, status)
        cp0_per_unit : 병렬 1기당 구매비. 총액은 호출측에서 bare module 계수를 곱한 뒤
                       n_units 배 한다 (베셀 F_P는 per_unit_capacity로 재계산할 것).
        status in {"in_range", "split", "clamped_low"}
    """
    spec = TURTON_EQUIPMENT_TABLE[equipment_key]
    a = np.atleast_1d(np.asarray(capacity_value, dtype=float))
    rng = _capacity_range(spec)

    if rng is None:
        cp0, _ = purchased_cost_2001usd(equipment_key, a)
        n = np.ones_like(a)
        return _squeeze(cp0), _squeeze(n), _squeeze(a), _squeeze(np.full(a.shape, "in_range"))

    lo, hi = rng
    n_units = np.ceil(np.maximum(a / hi, 1.0))
    per_unit_cap = np.where(a > hi, a / n_units, a)
    per_unit_cap = np.maximum(per_unit_cap, lo)
    cp0_each, _ = purchased_cost_2001usd(equipment_key, per_unit_cap)
    status = np.where(a > hi, "split",
             np.where(a < lo, "clamped_low", "in_range"))
    return _squeeze(cp0_each), _squeeze(n_units), _squeeze(per_unit_cap), _squeeze(status)


def _squeeze(x):
    x = np.asarray(x)
    return x.reshape(())[()] if x.size == 1 else x


# =====================================================================
# 2. 베어모듈 계수 (F_BM)
# =====================================================================
#
# VERIFIED=True  : 복수 공개 출처로 교차 확인한 값
# VERIFIED=False : 일반적으로 통용되는 값으로 기재했으나 원서 재대조 권장
#                  (논문에 인용하기 전 Turton Table A.3/A.4 원문 확인 필수)

# (a) 압축기·펌프류: 재질/압력 보정 없이 장비군별 고정 F_BM 하나를 곱하는 방식
TURTON_FBM_DIRECT = {
    "compressor_centrifugal": {"FBM": 2.8, "VERIFIED": True,
                                "source": "복수 공개 발췌 자료 교차확인"},
    "pump_centrifugal": {"FBM": 3.30, "VERIFIED": False,
                          "source": "통용값, 원서 Table A.4 대조 권장"},
    "turbine_expander": {"FBM": 2.8, "VERIFIED": False,
                          "source": "Turton Table A.4의 터빈/팽창기 전용 F_BM을 확보하지 "
                                    "못해, 같은 회전동력 장비군인 compressor_centrifugal의 "
                                    "검증된 F_BM=2.8을 유사장비 유추로 잠정 적용함. "
                                    "논문/코드에 쓰기 전 원서 Table A.4 원문 대조 필수."},
}

# (b) 열교환기·베셀·반응기류: F_BM = B1 + B2*F_M*F_P (재질/압력에 따라 변함)
TURTON_BM_FACTORS = {
    "heat_exchanger_floating_head": {"B1": 1.63, "B2": 1.66, "VERIFIED": False},
    "heat_exchanger_fixed_tube": {"B1": 1.63, "B2": 1.66, "VERIFIED": False},
    "heat_exchanger_u_tube": {"B1": 1.63, "B2": 1.66, "VERIFIED": False},
    "process_vessel_horizontal": {"B1": 2.25, "B2": 1.82, "VERIFIED": True,
                                   "source": "복수 공개 발췌 자료 교차확인"},
    "process_vessel_vertical": {"B1": 2.25, "B2": 1.82, "VERIFIED": True,
                                 "source": "복수 공개 발췌 자료 교차확인"},
    "reactor_jacketed_agitated": {"B1": 2.25, "B2": 1.82, "VERIFIED": False,
                                   "source": "베셀류와 동일 가정, 원서 대조 권장"},
    "fired_heater_nonreactive": {"FBM_DIRECT": 2.19, "VERIFIED": False,
                                  "source": "화염가열로는 통상 직접 F_BM 사용, 원서 대조 권장"},
}

MATERIAL_FACTOR_CARBON_STEEL = 1.0  # 탄소강 기준 F_M = 1.0 (Turton 표준 기준재질)


def pressure_factor_vessel(pressure_barg, diameter_m):
    """
    베셀류 압력계수 F_P (Turton 표준식).

        F_P = [ (P+1)*D + 0.00315 ] / [ 2*(850 - 0.6*(P+1)) * 0.0063 ]

    F_P가 1보다 작게 나오면(저압, 소형 베셀) 1.0으로 하한을 둔다 (Turton 관행).
    P는 barg(게이지압), D는 m.
    """
    p = np.asarray(pressure_barg, dtype=float)
    d = np.asarray(diameter_m, dtype=float)
    fp = ((p + 1) * d + 0.00315) / (2 * (850 - 0.6 * (p + 1)) * 0.0063)
    return np.maximum(fp, 1.0)


def bare_module_cost(equipment_key, cp0_2001usd, pressure_barg=None, diameter_m=None,
                      material_factor=MATERIAL_FACTOR_CARBON_STEEL):
    """
    베어모듈비용 C_BM [2001년 기준 US$]을 계산한다.

    압축기/펌프(TURTON_FBM_DIRECT에 등록)는 고정 F_BM을 곱한다.
    열교환기/베셀/반응기(TURTON_BM_FACTORS에 등록, B1/B2 형태)는
        F_BM = B1 + B2 * F_M * F_P
    를 계산해서 곱한다. 이때 F_P가 필요한 장비(베셀류)는 pressure_barg, diameter_m을
    함께 넘겨야 한다. 화염가열로처럼 FBM_DIRECT만 있는 항목은 직접 곱한다.
    """
    cp0 = np.asarray(cp0_2001usd, dtype=float)

    if equipment_key in TURTON_FBM_DIRECT:
        fbm = TURTON_FBM_DIRECT[equipment_key]["FBM"]
        return cp0 * fbm

    spec = TURTON_BM_FACTORS[equipment_key]
    if "FBM_DIRECT" in spec:
        return cp0 * spec["FBM_DIRECT"]

    b1, b2 = spec["B1"], spec["B2"]
    if pressure_barg is None or diameter_m is None:
        # 압력계수를 구할 수 없는 경우 F_P=1(상압 근사)로 보수적으로 처리
        fp = 1.0
    else:
        fp = pressure_factor_vessel(pressure_barg, diameter_m)

    fbm = b1 + b2 * material_factor * fp
    return cp0 * fbm


def total_module_cost(bare_module_costs_2001usd):
    """C_TM = 1.18 * sum(C_BM), 2001년 기준 US$"""
    return TOTAL_MODULE_FACTOR * np.sum(bare_module_costs_2001usd)


def grassroots_cost(bare_module_costs_2001usd):
    """C_GR = C_TM + 0.5 * sum(C_BM), 2001년 기준 US$"""
    total_bm = np.sum(bare_module_costs_2001usd)
    return TOTAL_MODULE_FACTOR * total_bm + GRASSROOTS_EXTRA_FACTOR * total_bm
