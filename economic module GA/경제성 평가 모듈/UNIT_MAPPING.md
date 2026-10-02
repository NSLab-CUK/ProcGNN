# 10개 SMR 공정 유닛 → 장비 타입 매핑표

작성 2026-09-02. `process_unit_map.py`의 사람이 읽는 버전.
근거: 인접행렬(`DataSet/Steady state/인접행렬/`), `steady state_june/*.Process_Main*.csv`의
T_R*/Q_R* 중앙값으로 반응기 역할 판정.

## 카테고리 정의

| 카테고리 | equipment_sizing 함수 | 용량변수 | Turton 키 |
|---|---|---|---|
| compressor | `compressor_isentropic_power_kw` (η=0.75) | 동력 kW | `compressor_centrifugal` |
| pump | `pump_power_kw` (η: P1블록별 0.57/0.465/0.375, 그 외 0.75) | 동력 kW | `pump_centrifugal` |
| turbine | `turbine_isentropic_power_kw` (η=0.75) | 동력 kW | `turbine_expander` |
| process_hx | `heat_exchanger_area_m2` (U=850 W/m²K) | 면적 m² | `heat_exchanger_floating_head` |
| utility_cooler | `heat_exchanger_area_m2` (U=850, 냉각수 25→40°C) | 면적 m² | `heat_exchanger_floating_head` |
| fired_heater | `burner_fired_heater_capacity_kw` (η_furnace=0.88, >100MW 병렬) | 흡수듀티 kW | `fired_heater_nonreactive` |
| flash_vessel | `vertical_vessel_sizing` (Souders–Brown + 홀드업 5min) | 부피 m³ | `process_vessel_vertical` |
| psa | `psa_bed_sizing` (u=0.35 m/s, 4베드/트레인) × n_beds | 부피 m³ | `process_vessel_vertical` × n_beds + 흡착제 별도 |
| wgs_reactor | `wgs_reactor_sizing` (GHSV, Lee 2021) | 부피 m³ | `process_vessel_vertical` + 촉매 별도 |
| reformer_folded | — | — | 선행 fired_heater/burner 비용에 노로 통합 |
| none | — | — | MIX/SP/공급·제품 노드, 비용 없음 |

## 반응기 역할 (T_R, Q_R 중앙값 기준)

| 공정 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| P01 | 개질 631°C (folded) | 개질 898°C (folded) | HT-WGS 370°C | LT-WGS 245°C |
| P02 | 개질 900°C (folded) | LT-WGS 230°C | 연소기 (Q≈−23MW) | — |
| P03 | 개질 873°C (folded) | HT-WGS 401°C | — | — |
| P04 | 개질 906°C (folded) | HT-WGS 397°C | LT-WGS 240°C | — |
| P05 | 개질 907°C (folded→HEAT1) | HT-WGS 366°C | — | — |
| P06 | 개질 875°C (folded) | HT-WGS 314°C | R3_BURNER (Q≈−254MW) | — |
| P07 | 개질 840°C (folded→HEAT1) | HT-WGS 376°C | LT-WGS 229°C | — |
| P08 | 개질 851°C (folded→HEAT2) | HT-WGS 375°C | LT-WGS 230°C | — |
| P09 | 개질 878°C (folded) | 단일 WGS 281°C (HT_WGS) | — | — |
| P10 | 개질 871°C (folded) | HT-WGS 385°C | — | — |

## 분리 유닛

- **SEP1** (P03,04,05,06,08,09,10): 인접행렬에서 항상 OUT_PROD(H2)로 나감 → **PSA**.
  P05/06/08은 SEP1이 H2/CO2 2-way (테일가스가 CO2-rich) — 여전히 PSA로 코스팅, CO2 분리
  전용 장비(아민 등)는 계상하지 않음(한계로 명시).
- **F1** (P03,05,06,07,08,10): 항상 SEP1 또는 OUT_H2O로 나감 → **응축수 녹아웃 드럼**.
  P07은 F1이 OUT_PROD 직결 (PSA 없는 단순 상분리 공정).

## 개질 열원 (노) 처리

| 공정 | 노 비용 계상 대상 | 듀티 |
|---|---|---|
| P01,03,04,09,10 | BURNER | η·\|Q_BURNER\| |
| P02 | R3 (연소기) | η·\|Q_R3\| |
| P06 | R3_BURNER + HEAT1 | η·\|Q_R3_BURNER\| + \|Q_HEAT1\| |
| P05 | HEAT1 | \|Q_HEAT1\| + \|Q_R1\| (개질흡열 합산, 선행 연소노드 없음) |
| P07 | HEAT1 + HEAT2 | \|Q_HEAT1\|+\|Q_R1\|, \|Q_HEAT2\| |
| P08 | HEAT1 + HEAT2 | \|Q_HEAT1\|, \|Q_HEAT2\|+\|Q_R1\| |

## 아직 안 된 것 (파이프라인 배선의 다음 단계)

각 노드의 **입·출구 스트림 번호** 매핑. P01만 `validate_rotating_equipment.py`에 있음.
이게 있어야 GNN 예측 스트림에서 각 노드의 Q / W / V_dot / 면적을 실제로 계산 가능.
