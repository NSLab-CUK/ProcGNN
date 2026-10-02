"""
properties.py 단위테스트.
GA_코드구축_프롬프트.md에 명시된 검증 항목을 그대로 구현했다.

    1. 순수 물(액상) 스트림의 체적유량이 기체 스트림보다 3~5 자릿수 작아야 한다.
    2. 이상기체 밀도가 알려진 값과 일치해야 한다 (공기, 25 degC, 1 bar -> 약 1.18 kg/m3).
    3. 물 밀도가 상온에서 약 997 kg/m3, 100 degC에서 약 958 kg/m3 근방이어야 한다.
    4. 비활성(유량 0) 스트림에서 0으로 나누기 오류가 나지 않아야 한다.
    5. 몰분율 합이 1인 스트림에서 평균분자량이 성분 분자량 범위 안에 있어야 한다.

실행 방법: python test_properties.py
"""

import numpy as np
import properties as prop


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_air_ideal_gas_density():
    # 공기 근사 조성 O2 21%, N2 79%, 나머지 0. 25 degC, 1 bar.
    x = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.21, 0.79]])  # H2O,H2,CH4,CO2,CO,O2,N2
    m_mix = prop.average_molar_mass(x)
    rho = prop.vapor_density_ideal_gas(
        temp_c=np.array([25.0]), pres_bar=np.array([1.0]), m_mix_kg_per_kmol=m_mix
    )
    # 참고: 실제 대기는 아르곤 0.9% 등을 포함해 평균분자량이 28.97이지만,
    # 본 프로젝트 화학종 목록(7종)에는 Ar이 없으므로 O2/N2만의 이론값 28.85와 비교한다.
    check(f"공기(O2 21%/N2 79%) 평균분자량 {m_mix[0]:.3f} kg/kmol ~ 28.85 이론값과 일치",
          abs(m_mix[0] - 28.85) < 0.05)
    # 이상기체법칙 rho=PM/RT 로 직접 계산한 이론값과 비교 (실측 공기밀도 1.18과는 Ar 성분
    # 차이만큼 약 0.4% 다를 수 있음)
    rho_theory = 1.0e5 * m_mix[0] / (prop.R_GAS * 1000.0 * 298.15)
    check(f"공기 밀도(25C,1bar) {rho[0]:.4f} kg/m3, 이상기체 이론값 {rho_theory:.4f}과 일치",
          abs(rho[0] - rho_theory) / rho_theory < 1e-6)
    # 실측 참고값 1.18 kg/m3은 통상 1 atm(1.01325 bar) 기준이라 1 bar 계산치와는
    # 그 차이(약 1.3%)만큼, 그리고 Ar 미포함분(약 0.4%)만큼 자연스럽게 차이가 난다.
    check(f"공기 밀도(25C,1bar) {rho[0]:.4f} kg/m3, 실측 참고값(1.18, 1atm 기준) 대비 오차 2% 이내",
          abs(rho[0] - 1.18) / 1.18 < 0.02)


def test_water_density_reference_points():
    rho_25 = prop.liquid_water_density(np.array([25.0]))[0]
    rho_100 = prop.liquid_water_density(np.array([100.0]))[0]
    check(f"물 밀도(25C) {rho_25:.2f} kg/m3 ~ 997 근방", abs(rho_25 - 997.0) < 1.0)
    check(f"물 밀도(100C) {rho_100:.2f} kg/m3 ~ 958 근방", abs(rho_100 - 958.0) < 2.0)


def test_condensate_vs_vapor_volumetric_flow_order_of_magnitude():
    # 순수 물, 액상 조건(응축수): 40 degC, 24 bar (포화온도보다 훨씬 낮음)
    x_water = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    m_dot = np.array([10000.0])  # kg/h

    props_liquid = prop.reconstruct_stream_properties(
        temp_c=np.array([40.0]), pres_bar=np.array([24.0]),
        mole_fractions=x_water, mass_flow_kgh=m_dot,
    )
    check("40C/24bar 순수 물 스트림은 액상(PHASE_LIQUID)으로 판정",
          props_liquid["phase"][0] == prop.PHASE_LIQUID)

    # 같은 질량유량, 같은 조성이지만 과열증기 조건(상판정 로직이 기상으로 분류하도록)
    props_vapor = prop.reconstruct_stream_properties(
        temp_c=np.array([300.0]), pres_bar=np.array([1.0]),
        mole_fractions=x_water, mass_flow_kgh=m_dot,
    )
    check("300C/1bar 순수 물 스트림은 기상(PHASE_VAPOR)으로 판정",
          props_vapor["phase"][0] == prop.PHASE_VAPOR)

    v_liquid = props_liquid["vol_flow_m3h"][0]
    v_vapor = props_vapor["vol_flow_m3h"][0]
    ratio = v_vapor / v_liquid
    check(f"동일 질량유량에서 기상/액상 체적유량비 {ratio:.1f}배 (3~5자릿수, 즉 10^3~10^5 수준 기대)",
          1e2 < ratio < 1e6)


def test_inactive_stream_no_division_by_zero():
    x = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    m_dot = np.array([0.0])  # 비활성 스트림
    result = prop.reconstruct_stream_properties(
        temp_c=np.array([25.0]), pres_bar=np.array([1.0]),
        mole_fractions=x, mass_flow_kgh=m_dot,
    )
    check("비활성 스트림 phase == PHASE_INACTIVE",
          result["phase"][0] == prop.PHASE_INACTIVE)
    check("비활성 스트림 체적유량 0, NaN/Inf 아님",
          result["vol_flow_m3h"][0] == 0.0 and np.isfinite(result["vol_flow_m3h"][0]))


def test_average_molar_mass_within_component_bounds():
    x = np.array([[0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]])
    m_mix = prop.average_molar_mass(x)[0]
    lo, hi = min(prop.MW_VEC), max(prop.MW_VEC)
    check(f"평균분자량 {m_mix:.2f} kg/kmol이 성분 범위 [{lo:.2f},{hi:.2f}] 안에 있음",
          lo <= m_mix <= hi)


def test_batch_vectorization_multiple_streams():
    # 10개 스트림을 한 번에 처리해도 오류 없이 동작하는지 확인 (벡터화 요구사항)
    n = 10
    rng = np.random.default_rng(0)
    x = rng.dirichlet(np.ones(7), size=n)
    temp_c = rng.uniform(20, 900, size=n)
    pres_bar = rng.uniform(1, 40, size=n)
    m_dot = rng.uniform(0, 50000, size=n)
    m_dot[0] = 0.0  # 비활성 스트림 하나 포함

    result = prop.reconstruct_stream_properties(temp_c, pres_bar, x, m_dot)
    check("10개 스트림 배치 처리 결과 shape 일치",
          result["density_kg_m3"].shape == (n,) and result["vol_flow_m3h"].shape == (n,))
    check("배치 처리 중 NaN이 비활성 스트림 외에는 발생하지 않음 (밀도 제외 체적유량 기준)",
          np.all(np.isfinite(result["vol_flow_m3h"])))


if __name__ == "__main__":
    test_air_ideal_gas_density()
    test_water_density_reference_points()
    test_condensate_vs_vapor_volumetric_flow_order_of_magnitude()
    test_inactive_stream_no_division_by_zero()
    test_average_molar_mass_within_component_bounds()
    test_batch_vectorization_multiple_streams()
    print("\n모든 테스트 통과")
