"""
economics.py
turton_costing.py(CAPEX) + Turton "Cost of Manufacturing" 식(OPEX)을 조립해
NPV, LCOH 등 최종 경제성 지표를 계산하는 모듈. GA/RL objective.py가 직접 호출하는
최상위 인터페이스다.

확정된 경제성 가정 (2025-09 논의에서 확정, 함부로 바꾸지 말 것)
    - 대표 방법론              : Turton module costing technique
    - 연간 가동시간             : 8,000 시간/년 (가동률 약 91.3%, PSE 분야 통상 관행값)
    - 할인율                   : 5.0%
    - CEPCI                    : 2025년 기준 가정값 800 (turton_costing.CEPCI_CURRENT)
    - 운전자본                 : FCI의 15% (Turton/Peters&Timmerhaus 공통 관행값)

OPEX는 Turton의 표준 "Cost of Manufacturing without depreciation" 식을 그대로 쓴다.

    COM_d = 0.180*FCI + 2.73*C_OL + 1.23*(C_UT + C_WT + C_RM)

    FCI  : 고정자본투자비 (turton_costing.grassroots_cost 결과를 그대로 사용)
    C_OL : 연간 운전인건비
    C_UT : 연간 유틸리티비 (압축기/펌프 동력, 열교환 열원 등)
    C_RM : 연간 원료비 (천연가스, 용수 등)
    C_WT : 연간 폐수/배출 처리비 (CO2 포집/배출권 비용 등, 해당 없으면 0)

    출처 Turton, Bailie, Whiting, Shaeiwitz, Bhattacharyya,
         "Analysis, Synthesis, and Design of Chemical Processes", Chapter 8.

이 식의 0.180*FCI 항은 감가상각을 제외한 세금/보험/유지보수 등을 포괄적으로
반영하는 항이므로, Peters&Timmerhaus의 개별 항목(세금+보험 2%, 유지보수 6% 등)을
따로 더하지 않는다. 두 방법을 동시에 쓰면 이중계상이 된다.

가격 파라미터(전력단가, 천연가스단가 등)는 이 모듈에 기본값을 심어두지 않았다.
잘못된 가정을 코드 속에 감추지 않기 위해서다. 반드시 호출 측에서 명시적으로
넘겨야 하며, 참고용 문헌값은 이 파일 하단 REFERENCE_PRICE_EXAMPLES에 출처와 함께
정리해 두었다 (그대로 쓰지 말고 최신 시세로 교체할 것).
"""

from __future__ import annotations

import numpy as np

# =====================================================================
# 확정 경제성 가정 (상수로 고정, 변경 시 이 파일과 보고서를 함께 갱신할 것)
# =====================================================================

DISCOUNT_RATE = 0.05                    # 할인율 5.0%
OPERATING_HOURS_PER_YEAR = 8000.0       # 연간 가동시간
WORKING_CAPITAL_FRACTION_OF_FCI = 0.15  # 운전자본 = FCI의 15%

# --- OPEX 방법 A: Turton Cost of Manufacturing (COM_d) 계수 ---
COM_FCI_COEFF = 0.180
COM_LABOR_COEFF = 2.73
COM_UTILITY_RM_WT_COEFF = 1.23

# --- OPEX 방법 B: 항목별 (Peters & Timmerhaus / Moon et al. 2025 스타일, 기본) ---
#   Fixed  = 유지보수 + 세금·보험 + 인건비 + 간접비(인건비+유지보수 기준)
#   Variable = 원료 + 유틸리티 + 폐수 + 촉매재생   (호출 측에서 합산해 넘김)
#   이 방식은 LCOH 정의 출처(이연호 2022, Moon et al. 2025)의 OPEX 구조와 일치하며
#   Turton COM_d 와 달리 판관비/R&D 마크업이 없어 값이 더 투명·보수적이지 않다.
MAINTENANCE_FRACTION_OF_FCI = 0.04     # 유지보수·수리. Peters&Timmerhaus 통상 2~6%, H2 TEA ~4%
TAX_INSURANCE_FRACTION_OF_FCI = 0.02   # 지방세 + 보험. Peters&Timmerhaus ~1~2%
PLANT_OVERHEAD_FRACTION = 0.60         # 플랜트+관리 간접비 = 0.60*(인건비 + 유지보수). P&T 통상 50~70%


# =====================================================================
# 1. 총자본투자비
# =====================================================================

def total_capital_investment(fci_usd, working_capital_fraction=WORKING_CAPITAL_FRACTION_OF_FCI):
    """
    TCI = FCI + Working Capital
    fci_usd는 turton_costing.grassroots_cost()의 결과(CEPCI 보정 완료된 값)를 넣는다.
    """
    fci = np.asarray(fci_usd, dtype=float)
    working_capital = fci * working_capital_fraction
    return fci + working_capital


# =====================================================================
# 2. 운영비 (Turton Cost of Manufacturing)
# =====================================================================

def operating_labor_headcount(n_particulate_steps, n_other_steps):
    """
    운전당 필요 운전원 수(1교대 기준) N_OL, Turton 표준 상관식.

        N_OL = sqrt(6.29 + 31.7*P^2 + 0.23*N_np)

    P : 고체 입자를 다루는 공정단계 수 (본 SMR 공정은 기본 0)
    N_np : 그 외 공정단계 수 (반응기, 압축기, 열교환기 등 유닛 개수로 근사)

    실제 필요 인원(3교대 + 연차 등 반영)은 이 값에 약 4.5를 곱한다 (Turton 관행).
    """
    p = np.asarray(n_particulate_steps, dtype=float)
    n_np = np.asarray(n_other_steps, dtype=float)
    n_ol_per_shift = np.sqrt(6.29 + 31.7 * p**2 + 0.23 * n_np)
    return n_ol_per_shift * 4.5  # 총 필요 인원(3교대 등 반영)


def cost_of_manufacturing(fci_usd, c_ol_usd_per_year, c_ut_usd_per_year,
                           c_rm_usd_per_year, c_wt_usd_per_year=0.0):
    """
    Turton Cost of Manufacturing (감가상각 제외), 연간 OPEX [US$/year].

        COM_d = 0.180*FCI + 2.73*C_OL + 1.23*(C_UT + C_WT + C_RM)
    """
    fci = np.asarray(fci_usd, dtype=float)
    c_ol = np.asarray(c_ol_usd_per_year, dtype=float)
    c_ut = np.asarray(c_ut_usd_per_year, dtype=float)
    c_rm = np.asarray(c_rm_usd_per_year, dtype=float)
    c_wt = np.asarray(c_wt_usd_per_year, dtype=float)

    return (COM_FCI_COEFF * fci + COM_LABOR_COEFF * c_ol
            + COM_UTILITY_RM_WT_COEFF * (c_ut + c_wt + c_rm))


def fixed_operating_cost(fci_usd, c_ol_usd_per_year,
                          maintenance_fraction=MAINTENANCE_FRACTION_OF_FCI,
                          tax_insurance_fraction=TAX_INSURANCE_FRACTION_OF_FCI,
                          plant_overhead_fraction=PLANT_OVERHEAD_FRACTION):
    """
    항목별 고정 운영비 [US$/year] (Peters & Timmerhaus 스타일).

        C_maint  = f_maint * FCI
        C_taxins = f_taxins * FCI
        C_OL     = (입력) 운전 인건비
        C_OH     = f_oh * (C_OL + C_maint)          플랜트+관리 간접비

    Returns dict(maintenance, tax_insurance, labor, overhead, total)
    """
    fci = np.asarray(fci_usd, dtype=float)
    c_ol = np.asarray(c_ol_usd_per_year, dtype=float)
    c_maint = maintenance_fraction * fci
    c_taxins = tax_insurance_fraction * fci
    c_oh = plant_overhead_fraction * (c_ol + c_maint)
    return {
        "maintenance": c_maint, "tax_insurance": c_taxins,
        "labor": c_ol, "overhead": c_oh,
        "total": c_maint + c_taxins + c_ol + c_oh,
    }


def itemized_opex(fci_usd, c_ol_usd_per_year, c_variable_usd_per_year,
                   **fixed_kwargs):
    """
    항목별 총 OPEX [US$/year] = 고정 + 변동.
    c_variable = C_RM + C_UT + C_WT + 촉매재생  (호출 측에서 합산해 넘김).
    """
    fixed = fixed_operating_cost(fci_usd, c_ol_usd_per_year, **fixed_kwargs)
    var = np.asarray(c_variable_usd_per_year, dtype=float)
    return fixed["total"] + var, fixed


def utility_cost_from_power(power_kw_list, electricity_price_usd_per_kwh,
                             operating_hours=OPERATING_HOURS_PER_YEAR):
    """
    압축기/펌프 동력 합계로부터 연간 전력비 [US$/year]를 계산한다.
    power_kw_list는 equipment_sizing.compressor_isentropic_power_kw /
    pump_power_kw로 얻은 개별 장비 동력 [kW]의 배열 또는 리스트.
    """
    total_power_kw = np.sum(np.asarray(power_kw_list, dtype=float))
    return total_power_kw * operating_hours * electricity_price_usd_per_kwh


def raw_material_cost(mass_flow_kgh, price_usd_per_kg, operating_hours=OPERATING_HOURS_PER_YEAR):
    """
    원료(천연가스, 용수 등) 연간 비용 [US$/year] = 시간당 유량 * 운전시간 * 단가.
    mass_flow_kgh는 properties.py에서 재구성한 공급 스트림 질량유량(설계점 기준).
    """
    return np.asarray(mass_flow_kgh, dtype=float) * operating_hours * price_usd_per_kg


# =====================================================================
# 3. NPV / LCOH
# =====================================================================

def npv(cash_flows, discount_rate=DISCOUNT_RATE):
    """
    NPV = sum_t [ CF_t / (1+r)^t ],  t = 0, 1, ..., len(cash_flows)-1
    cash_flows[0]은 사업 개시 시점(t=0) 현금흐름.
    """
    cf = np.asarray(cash_flows, dtype=float)
    t = np.arange(len(cf))
    return np.sum(cf / (1.0 + discount_rate) ** t)


def build_flat_schedule(value_per_year, n_years):
    """연간 동일값 스케줄 배열 생성 (t=0 제외, 1~n_years). 간단 시나리오용 헬퍼."""
    return np.full(n_years, float(value_per_year))


def lcoh(capex_schedule, opex_schedule, h2_production_schedule_kg,
         discount_rate=DISCOUNT_RATE):
    """
    균등화 수소생산단가 LCOH [US$/kg].

        LCOH = NPV(CAPEX + OPEX) / NPV(H2 생산량)

    세 스케줄(capex_schedule, opex_schedule, h2_production_schedule_kg)은 같은
    길이의 연도별 배열이어야 하며 t=0부터 시작한다 (건설기간 연도는 h2_production=0,
    CAPEX>0 으로 넣는다).

    출처 정의 BEIS(2021)를 인용한 이연호 외(2022), 한국자원공학회지 59(2), 식(8),
         Moon et al. (2025), Chemical Engineering Journal 520, Eq.(28)(29)와 동일한 정의.
    """
    capex = np.asarray(capex_schedule, dtype=float)
    opex = np.asarray(opex_schedule, dtype=float)
    h2 = np.asarray(h2_production_schedule_kg, dtype=float)

    total_cost_npv = npv(capex + opex, discount_rate)
    h2_npv = npv(h2, discount_rate)
    return total_cost_npv / h2_npv


def net_present_value_of_project(capex_schedule, opex_schedule, revenue_schedule,
                                  discount_rate=DISCOUNT_RATE):
    """
    프로젝트 NPV [US$] = NPV(매출 - CAPEX - OPEX)
    """
    capex = np.asarray(capex_schedule, dtype=float)
    opex = np.asarray(opex_schedule, dtype=float)
    revenue = np.asarray(revenue_schedule, dtype=float)
    free_cash_flow = revenue - capex - opex
    return npv(free_cash_flow, discount_rate)


# =====================================================================
# 참고용 가격 예시 (그대로 쓰지 말 것, 반드시 최신 값으로 교체)
# =====================================================================

REFERENCE_PRICE_EXAMPLES = {
    "natural_gas_usd_per_kg": {
        "value": 0.218,
        "source": "Moon et al. (2025), CEJ 520, EIA 2015-2024 10년 평균 기준",
    },
    "process_water_usd_per_kg": {
        "value": 2.38e-3,
        "source": "Moon et al. (2025), CEJ 520, Table 7",
    },
    "note": "전력단가는 확인된 문헌값이 없어 예시를 넣지 않았다. 국내 산업용 "
            "전기요금(한국전력 산업용 요금표)을 직접 조사해서 넣을 것.",
}
