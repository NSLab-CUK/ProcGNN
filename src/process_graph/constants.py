from __future__ import annotations

from typing import Literal, Optional

NodeRole = Literal["source", "unit", "sink", "input_virtual", "output_virtual"]
CellKind = Literal["reference", "constant", "missing", "passthrough"]
HXRole = Optional[Literal["hot_outlet", "cold_outlet", "hot_inlet_cold_outlet_diff"]]
PassthroughPolicy = Literal["zero", "inherit_upstream"]

OPER_FEATURE_SLOTS = [
    "ch4",
    "air",
    "h2o",
    "h2",
    "co2",
    "split_ratio",
    "split_ratio2",
    "split_flow",
    "press",
    "temp",
    "cold_temp",
    "cold_hot_dt",
    "hot_temp",
]

DECODER_CATEGORY_MAP = {
    "소비 에너지": "power",
    "열 부하": "heat_duty",
    "유량": "flow",
    "열교환면적": "hx_area",
}

ROLE_TO_IDX = {
    "source": 0,
    "unit": 1,
    "sink": 2,
    "input_virtual": 3,
    "output_virtual": 4,
}

UNIT_TYPE_TO_CANONICAL = {
    "V_INPUT": "input_virtual",
    "V_OUTPUT": "output_virtual",
    "input_virtual": "input_virtual",
    "output_virtual": "output_virtual",
    "Stream": "stream",
    "stream": "stream",
    "mixer": "mixer",
    "SMR reactor": "smr_reactor",
    "WGS reactor": "wgs_reactor",
    "burner": "burner",
    "HX_dt": "hx_dt",
    "HX_hot": "hx_hot",
    "HX_cold": "hx_cold",
    "compressor": "compressor",
    "pump": "pump",
    "turbine": "turbine",
    "splitter": "splitter",
    "cooler": "cooler",
    "heater": "heater",
    "flash": "flash",
    "psa": "psa",
}

UNIT_VOCAB = [
    "input_virtual",
    "output_virtual",
    "stream",
    "mixer",
    "smr_reactor",
    "wgs_reactor",
    "burner",
    "hx_dt",
    "hx_hot",
    "hx_cold",
    "compressor",
    "pump",
    "turbine",
    "splitter",
    "cooler",
    "heater",
    "flash",
    "psa",
]
UNIT_TO_IDX = {u: i for i, u in enumerate(UNIT_VOCAB)}

STREAM_EDGE_FEATURE_SLOTS = [
    "Temp",
    "Pres",
    "Vol_Flow",
    "Mole_Flow",
    "Mass_Flow",
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
]

STREAM_ROLE_VOCAB = [
    "unknown",
    "feed",
    "internal",
    "recycle_internal",
    "product",
    "exhaust",
    "output",
    "zero_or_side_output",
    "recycle_or_output",
    "utility_or_internal",
]
STREAM_ROLE_TO_IDX = {role: i for i, role in enumerate(STREAM_ROLE_VOCAB)}

# Property prediction needs finer output semantics than the generic GNN role
# vocabulary, where every terminal stream is simply "output".
PROPERTY_STREAM_ROLE_VOCAB = [
    *STREAM_ROLE_VOCAB,
    "output_product",
    "output_h2o",
    "output_co2",
    "output_exhaust",
    "output_resteam",
    "output_vent",
    "output_other",
]
PROPERTY_STREAM_ROLE_TO_IDX = {
    role: i for i, role in enumerate(PROPERTY_STREAM_ROLE_VOCAB)
}


def property_stream_role_name(stream_role: object, dst_node_raw: object = "") -> str:
    """Return a process-agnostic role derived only from flowsheet metadata."""
    generic_role = str(stream_role or "unknown").strip().lower()
    output_port = str(dst_node_raw or "").strip().lower()
    output_roles = {
        "out_prod": "output_product",
        "out_h2o": "output_h2o",
        "out_co2": "output_co2",
        "out_exhaust": "output_exhaust",
        "out_resteam": "output_resteam",
        "out_vent": "output_vent",
        "out_stream": "output_other",
    }
    if output_port in output_roles:
        return output_roles[output_port]
    if output_port in {"out_z1", "out_z2"}:
        return "zero_or_side_output"
    if generic_role in PROPERTY_STREAM_ROLE_TO_IDX:
        return generic_role
    return "unknown"


def property_stream_role_id(stream_role: object, dst_node_raw: object = "") -> int:
    return PROPERTY_STREAM_ROLE_TO_IDX[
        property_stream_role_name(stream_role, dst_node_raw)
    ]

HX_ROLE_TO_IDX = {
    None: 0,
    "hot_outlet": 1,
    "cold_outlet": 2,
    "hot_inlet_cold_outlet_diff": 3,
}
