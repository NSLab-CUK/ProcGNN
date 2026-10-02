"""
process_stream_map.py
10개 SMR 공정 각 블록의 포트->스트림 매핑. Aspen COM(각 .bkp/.apw headless open,
block.Elements('Ports'))으로 2026-09-02 직접 추출한 authoritative 값이다.
이름 문자열 추론이 아니라 Aspen 플로우시트 연결 그대로임.

스키마 (kind별)
  hx         : hot_in, hot_out, cold_in, cold_out  (열교환기/냉각기/가열기 중 2-sided)
  inout      : feed, prod                          (압축기/펌프/터빈/1-sided heater·cooler/RGibbs 반응기)
  reactor_vl : feed, vapor_out, liquid_out         (REQUIL 등 기액출구 반응기; liquid_out가 Z1/Z2 더미면 None)
  vessel     : feed, vapor_out, liquid_out         (FLASH2 응축수 녹아웃)
  sep        : feed, outlets[]                     (SEP 블록 = PSA/성분분리; outlets[0]이 통상 제품)
  mixer      : feed[], prod                        (비용 없음)
  splitter   : feed, prod[]                        (비용 없음)

스트림 이름은 해당 공정 Process_Streams CSV의 Stream_Name과 동일.
Z1/Z2는 더미(SKIP_STREAMS), 이미 None 처리됨.
"""

from __future__ import annotations

PROCESS_STREAM_MAP = {
    "P01": {
        "MIX3": {'kind': 'mixer', 'feed': ['26', '28'], 'prod': '29'},
        "BURNER": {'kind': 'burner', 'feed': ['11', '12'], 'prod': '13'},
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '01'},
        "C2": {'kind': 'inout', 'feed': 'FUEL', 'prod': '11'},
        "C3": {'kind': 'inout', 'feed': 'AIR', 'prod': '12'},
        "HX1": {'kind': 'hx', 'hot_in': '05', 'hot_out': '06', 'cold_in': '03', 'cold_out': '04'},
        "HX2": {'kind': 'hx', 'hot_in': '13', 'hot_out': 'FUELGAS', 'cold_in': '17', 'cold_out': '18'},
        "HX3": {'kind': 'hx', 'hot_in': '06', 'hot_out': '07', 'cold_in': '21', 'cold_out': '22'},
        "HX4": {'kind': 'hx', 'hot_in': '08', 'hot_out': '09', 'cold_in': '23', 'cold_out': '24'},
        "HX5": {'kind': 'hx', 'hot_in': '10', 'hot_out': 'PROD', 'cold_in': '25', 'cold_out': '26'},
        "MIX1": {'kind': 'mixer', 'feed': ['01', 'STEAM'], 'prod': '02'},
        "MIX2": {'kind': 'mixer', 'feed': ['24', '22', '19'], 'prod': 'M2'},
        "P1": {'kind': 'inout', 'feed': '14', 'prod': '17'},
        "P2": {'kind': 'inout', 'feed': '15', 'prod': '20'},
        "P3": {'kind': 'inout', 'feed': '16', 'prod': '25'},
        "R1": {'kind': 'inout', 'feed': '02', 'prod': '03'},
        "R2": {'kind': 'inout', 'feed': '04', 'prod': '05'},
        "R3": {'kind': 'reactor_vl', 'feed': '07', 'vapor_out': '08', 'liquid_out': None},
        "R4": {'kind': 'reactor_vl', 'feed': '09', 'vapor_out': '10', 'liquid_out': None},
        "SP1": {'kind': 'splitter', 'feed': 'WATER', 'prod': ['14', '15', '16']},
        "SP2": {'kind': 'splitter', 'feed': '20', 'prod': ['21', '23']},
        "SP3": {'kind': 'splitter', 'feed': 'M2', 'prod': ['27', 'STEAM']},
        "T1": {'kind': 'inout', 'feed': '18', 'prod': '19'},
        "T2": {'kind': 'inout', 'feed': '27', 'prod': '28'},
        "T3": {'kind': 'inout', 'feed': '29', 'prod': 'RESTEAM'},
    },
    "P02": {
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '01'},
        "C2": {'kind': 'inout', 'feed': 'AIR', 'prod': '09'},
        "C3": {'kind': 'inout', 'feed': '04', 'prod': '05'},
        "C4": {'kind': 'inout', 'feed': '06', 'prod': '07'},
        "COOL1": {'kind': 'inout', 'feed': '03', 'prod': '04'},
        "COOL2": {'kind': 'inout', 'feed': '05', 'prod': '06'},
        "COOL4": {'kind': 'inout', 'feed': 'REF2', 'prod': 'REF3'},
        "HX1": {'kind': 'hx', 'hot_in': '12', 'hot_out': '13', 'cold_in': 'WATER1', 'cold_out': 'WATER2'},
        "HX2": {'kind': 'hx', 'hot_in': '11', 'hot_out': '12', 'cold_in': 'WATER2', 'cold_out': 'STEAM'},
        "HX4": {'kind': 'hx', 'hot_in': 'REF1', 'hot_out': 'REF2', 'cold_in': '08', 'cold_out': 'R1-FEED'},
        "MIX1": {'kind': 'mixer', 'feed': ['02', '09'], 'prod': '10'},
        "MIX2": {'kind': 'mixer', 'feed': ['07', 'STEAM'], 'prod': '08'},
        "P1": {'kind': 'inout', 'feed': 'WATER', 'prod': 'WATER1'},
        "R1": {'kind': 'inout', 'feed': 'R1-FEED', 'prod': 'REF1'},
        "R2": {'kind': 'reactor_vl', 'feed': 'REF3', 'vapor_out': 'PROD', 'liquid_out': 'VENT'},
        "R3": {'kind': 'burner', 'feed': ['10'], 'prod': '11'},
        "SP1": {'kind': 'splitter', 'feed': '01', 'prod': ['02', '03']},
    },
    "P03": {
        "BURNER": {'kind': 'burner', 'feed': ['16'], 'prod': '17'},
        "C1": {'kind': 'inout', 'feed': '13', 'prod': '14'},
        "C2": {'kind': 'inout', 'feed': 'CH4', 'prod': '1'},
        "F1": {'kind': 'vessel', 'feed': '8', 'vapor_out': '12', 'liquid_out': 'RE'},
        "HX1": {'kind': 'hx', 'hot_in': '18', 'hot_out': '19', 'cold_in': '1', 'cold_out': '2'},
        "HX2": {'kind': 'hx', 'hot_in': '17', 'hot_out': '18', 'cold_in': '3', 'cold_out': '4'},
        "HX3": {'kind': 'hx', 'hot_in': '19', 'hot_out': 'OUT', 'cold_in': '15', 'cold_out': '16'},
        "HX4": {'kind': 'hx', 'hot_in': '5', 'hot_out': '6', 'cold_in': '10', 'cold_out': '11'},
        "HX5": {'kind': 'hx', 'hot_in': '7', 'hot_out': '8', 'cold_in': '9', 'cold_out': '10'},
        "MIX1": {'kind': 'mixer', 'feed': ['11', '2'], 'prod': '3'},
        "MIX2": {'kind': 'mixer', 'feed': ['14', 'AIR'], 'prod': '15'},
        "P1": {'kind': 'inout', 'feed': 'WATER', 'prod': '9'},
        "R1": {'kind': 'inout', 'feed': '4', 'prod': '5'},
        "R2": {'kind': 'reactor_vl', 'feed': '6', 'vapor_out': '7', 'liquid_out': None},
        "SEP1": {'kind': 'sep', 'feed': '12', 'outlets': ['PROD', '13']},
    },
    "P04": {
        "BURNER": {'kind': 'burner', 'feed': ['FUEL', 'AIR'], 'prod': '14'},
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '01'},
        "COOL1": {'kind': 'inout', 'feed': '08', 'prod': '09'},
        "HX1": {'kind': 'hx', 'hot_in': '14', 'hot_out': 'EX', 'cold_in': '02', 'cold_out': '03'},
        "HX2": {'kind': 'hx', 'hot_in': '04', 'hot_out': '05', 'cold_in': '12', 'cold_out': '13'},
        "HX3": {'kind': 'hx', 'hot_in': '06', 'hot_out': '07', 'cold_in': '11', 'cold_out': '12'},
        "MIX1": {'kind': 'mixer', 'feed': ['13', '01'], 'prod': '02'},
        "P1": {'kind': 'inout', 'feed': 'WATER', 'prod': '11'},
        "R1": {'kind': 'inout', 'feed': '03', 'prod': '04'},
        "R2": {'kind': 'reactor_vl', 'feed': '05', 'vapor_out': '06', 'liquid_out': None},
        "R3": {'kind': 'reactor_vl', 'feed': '07', 'vapor_out': '08', 'liquid_out': None},
        "SEP1": {'kind': 'sep', 'feed': '09', 'outlets': ['PROD', '10']},
    },
    "P05": {
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '04'},
        "COOL1": {'kind': 'inout', 'feed': '10', 'prod': '11'},
        "F1": {'kind': 'vessel', 'feed': '11', 'vapor_out': '12', 'liquid_out': 'RE'},
        "HEAT1": {'kind': 'inout', 'feed': '04', 'prod': '05'},
        "HX1": {'kind': 'hx', 'hot_in': '09', 'hot_out': '10', 'cold_in': '01', 'cold_out': '02'},
        "HX2": {'kind': 'hx', 'hot_in': '07', 'hot_out': '08', 'cold_in': '02', 'cold_out': '03'},
        "MIX1": {'kind': 'mixer', 'feed': ['03', '05'], 'prod': '06'},
        "P1": {'kind': 'inout', 'feed': 'WATER', 'prod': '01'},
        "R1": {'kind': 'inout', 'feed': '06', 'prod': '07'},
        "R2": {'kind': 'reactor_vl', 'feed': '08', 'vapor_out': '09', 'liquid_out': None},
        "SEP1": {'kind': 'sep', 'feed': '12', 'outlets': ['13', 'PROD']},
    },
    "P06": {
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '01'},
        "C2": {'kind': 'inout', 'feed': 'FUEL', 'prod': '13'},
        "C3": {'kind': 'inout', 'feed': 'AIR', 'prod': '14'},
        "COOL1": {'kind': 'inout', 'feed': '09', 'prod': '10'},
        "F1": {'kind': 'vessel', 'feed': '10', 'vapor_out': '11', 'liquid_out': 'RE'},
        "HEAT1": {'kind': 'inout', 'feed': '02', 'prod': '03'},
        "HX1": {'kind': 'hx', 'hot_in': '16', 'hot_out': '17', 'cold_in': '03', 'cold_out': '04'},
        "HX2": {'kind': 'hx', 'hot_in': '07', 'hot_out': '08', 'cold_in': '05', 'cold_out': '06'},
        "MIX1": {'kind': 'mixer', 'feed': ['04', '01'], 'prod': '05'},
        "MIX2": {'kind': 'mixer', 'feed': ['14', '13'], 'prod': '15'},
        "P1": {'kind': 'inout', 'feed': 'WATER', 'prod': '02'},
        "R1": {'kind': 'inout', 'feed': '06', 'prod': '07'},
        "R2": {'kind': 'reactor_vl', 'feed': '08', 'vapor_out': '09', 'liquid_out': None},
        "R3_BURNER": {'kind': 'burner', 'feed': ['15'], 'prod': '16'},
        "SEP1": {'kind': 'sep', 'feed': '11', 'outlets': ['PROD', '12']},
    },
    "P07": {
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '01'},
        "COOL1": {'kind': 'inout', 'feed': '12', 'prod': '13'},
        "F1": {'kind': 'vessel', 'feed': '13', 'vapor_out': 'PROD', 'liquid_out': 'RE'},
        "HEAT1": {'kind': 'inout', 'feed': '02', 'prod': '03'},
        "HEAT2": {'kind': 'inout', 'feed': '09', 'prod': '10'},
        "HX1": {'kind': 'hx', 'hot_in': '04', 'hot_out': '05', 'cold_in': '08', 'cold_out': '09'},
        "HX2": {'kind': 'hx', 'hot_in': '06', 'hot_out': '11', 'cold_in': '07', 'cold_out': '08'},
        "MIX1": {'kind': 'mixer', 'feed': ['01', '10'], 'prod': '02'},
        "P1": {'kind': 'inout', 'feed': 'WATER', 'prod': '07'},
        "R1": {'kind': 'inout', 'feed': '03', 'prod': '04'},
        "R2": {'kind': 'reactor_vl', 'feed': '05', 'vapor_out': '06', 'liquid_out': None},
        "R3": {'kind': 'reactor_vl', 'feed': '11', 'vapor_out': '12', 'liquid_out': None},
    },
    "P08": {
        "C1": {'kind': 'inout', 'feed': '2', 'prod': '3'},
        "COOL1": {'kind': 'inout', 'feed': '5', 'prod': '6'},
        "COOL2": {'kind': 'inout', 'feed': '7', 'prod': '8'},
        "COOL3": {'kind': 'inout', 'feed': '9', 'prod': '10'},
        "F1": {'kind': 'vessel', 'feed': '10', 'vapor_out': '11', 'liquid_out': 'RE'},
        "HEAT1": {'kind': 'inout', 'feed': 'WATER', 'prod': '1'},
        "HEAT2": {'kind': 'inout', 'feed': '3', 'prod': '4'},
        "MIX1": {'kind': 'mixer', 'feed': ['1', 'CH4'], 'prod': '2'},
        "R1": {'kind': 'inout', 'feed': '4', 'prod': '5'},
        "R2": {'kind': 'reactor_vl', 'feed': '6', 'vapor_out': '7', 'liquid_out': None},
        "R3": {'kind': 'reactor_vl', 'feed': '8', 'vapor_out': '9', 'liquid_out': None},
        "SEP1": {'kind': 'sep', 'feed': '11', 'outlets': ['PROD', '12']},
    },
    "P09": {
        "MIX1": {'kind': 'mixer', 'feed': ['01', '18'], 'prod': '02'},
        "BURNER": {'kind': 'burner', 'feed': ['11'], 'prod': '12'},
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '01'},
        "C2": {'kind': 'inout', 'feed': 'AIR', 'prod': '19'},
        "C3": {'kind': 'inout', 'feed': 'FUEL', 'prod': '21'},
        "HX1": {'kind': 'hx', 'hot_in': '04', 'hot_out': '05', 'cold_in': '02', 'cold_out': '03'},
        "HX2": {'kind': 'hx', 'hot_in': '05', 'hot_out': '06', 'cold_in': '15', 'cold_out': '16'},
        "HX3": {'kind': 'hx', 'hot_in': '13', 'hot_out': '14', 'cold_in': '07', 'cold_out': '08'},
        "HX4": {'kind': 'hx', 'hot_in': '12', 'hot_out': '13', 'cold_in': '17', 'cold_out': '18'},
        "HX5": {'kind': 'hx', 'hot_in': '09', 'hot_out': 'PROD', 'cold_in': '16', 'cold_out': '17'},
        "HX6": {'kind': 'hx', 'hot_in': '14', 'hot_out': 'EXHAUS', 'cold_in': '19', 'cold_out': '20'},
        "MIX2": {'kind': 'mixer', 'feed': ['10', '20', '21'], 'prod': '11'},
        "P1": {'kind': 'inout', 'feed': 'H2O', 'prod': '15'},
        "R1": {'kind': 'inout', 'feed': '03', 'prod': '04'},
        "R2": {'kind': 'reactor_vl', 'feed': '06', 'vapor_out': '07', 'liquid_out': None},
        "SEP1": {'kind': 'sep', 'feed': '08', 'outlets': ['09', '10']},
    },
    "P10": {
        "BURNER": {'kind': 'burner', 'feed': ['23', '17'], 'prod': '18'},
        "C1": {'kind': 'inout', 'feed': 'CH4', 'prod': '01'},
        "C2": {'kind': 'inout', 'feed': 'AIR', 'prod': '22'},
        "COOL1": {'kind': 'inout', 'feed': '13', 'prod': '14'},
        "F1": {'kind': 'vessel', 'feed': '14', 'vapor_out': '15', 'liquid_out': 'RE'},
        "HX1": {'kind': 'hx', 'hot_in': '08', 'hot_out': '09', 'cold_in': '01', 'cold_out': '02'},
        "HX2": {'kind': 'hx', 'hot_in': '07', 'hot_out': '08', 'cold_in': '03', 'cold_out': '04'},
        "HX3": {'kind': 'hx', 'hot_in': '06', 'hot_out': '07', 'cold_in': '04', 'cold_out': '05'},
        "HX4": {'kind': 'hx', 'hot_in': '18', 'hot_out': 'OUT', 'cold_in': '20', 'cold_out': '21'},
        "HX5": {'kind': 'hx', 'hot_in': '09', 'hot_out': '10', 'cold_in': '22', 'cold_out': '23'},
        "HX7": {'kind': 'hx', 'hot_in': '11', 'hot_out': '12', 'cold_in': '16', 'cold_out': '17'},
        "HX8": {'kind': 'hx', 'hot_in': '12', 'hot_out': '13', 'cold_in': '19', 'cold_out': '20'},
        "MIX1": {'kind': 'mixer', 'feed': ['21', '02'], 'prod': '03'},
        "P1": {'kind': 'inout', 'feed': 'WATER', 'prod': '19'},
        "R1": {'kind': 'inout', 'feed': '05', 'prod': '06'},
        "R2": {'kind': 'reactor_vl', 'feed': '10', 'vapor_out': '11', 'liquid_out': None},
        "SEP1": {'kind': 'sep', 'feed': '15', 'outlets': ['PROD', '16']},
    },
}


def block_streams(process_id, block):
    """해당 공정/블록의 포트 매핑 dict 반환."""
    return PROCESS_STREAM_MAP[process_id][block]


# 인접행렬의 Output 노드(OUT_*)에 연결된 말단 스트림.
# 어느 블록의 출력이면서 어느 블록의 입력도 아닌 스트림 = 계 경계를 넘는 출구.
# 이 중 무엇이 대기방출(flue_gas/vent) / 포집CO2 / 제품 / 응축수인지는
# 외부 스트림 역할매핑표에서 판정한다 (여기서는 목록만 제공).
OUTPUT_STREAMS = {
    "P01": ['FUELGAS', 'PROD', 'RESTEAM'],
    "P02": ['13', 'PROD', 'VENT'],
    "P03": ['OUT', 'PROD', 'RE'],
    "P04": ['10', 'EX', 'PROD'],
    "P05": ['13', 'PROD', 'RE'],
    "P06": ['12', '17', 'PROD', 'RE'],
    "P07": ['PROD', 'RE'],
    "P08": ['12', 'PROD', 'RE'],
    "P09": ['EXHAUS', 'PROD'],
    "P10": ['OUT', 'PROD', 'RE'],
}
