"""Static vocabularies for bounded TPC-H string domains."""

from __future__ import annotations

STATIC_STRING_DICTIONARIES: dict[str, tuple[str, ...]] = {
    "c_mktsegment": ("AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"),
    "l_linestatus": ("F", "O"),
    "l_returnflag": ("A", "N", "R"),
    "l_shipinstruct": ("COLLECT COD", "DELIVER IN PERSON", "NONE", "TAKE BACK RETURN"),
    "l_shipmode": ("AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"),
    "n_name": (
        "ALGERIA",
        "ARGENTINA",
        "BRAZIL",
        "CANADA",
        "CHINA",
        "EGYPT",
        "ETHIOPIA",
        "FRANCE",
        "GERMANY",
        "INDIA",
        "INDONESIA",
        "IRAN",
        "IRAQ",
        "JAPAN",
        "JORDAN",
        "KENYA",
        "MOROCCO",
        "MOZAMBIQUE",
        "PERU",
        "ROMANIA",
        "RUSSIA",
        "SAUDI ARABIA",
        "UNITED KINGDOM",
        "UNITED STATES",
        "VIETNAM",
    ),
    "o_orderpriority": ("1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"),
    "o_orderstatus": ("F", "O", "P"),
    "p_mfgr": tuple(f"Manufacturer#{index}" for index in range(1, 6)),
    "r_name": ("AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"),
}

STATIC_STRING_DICTIONARY_TABLES: dict[str, frozenset[str]] = {
    "customer": frozenset({"c_mktsegment"}),
    "lineitem": frozenset({"l_linestatus", "l_returnflag", "l_shipinstruct", "l_shipmode"}),
    "nation": frozenset({"n_name"}),
    "orders": frozenset({"o_orderpriority", "o_orderstatus"}),
    "part": frozenset({"p_mfgr"}),
    "region": frozenset({"r_name"}),
}


def static_string_dictionary(table_name: str | None, column_name: str | None) -> tuple[str, ...] | None:
    """Return a static TPC-H vocabulary only for known table/column pairs."""

    if table_name is None or column_name is None:
        return None
    normalized_table = table_name.strip('"').rsplit(".", 1)[-1]
    allowed_columns = STATIC_STRING_DICTIONARY_TABLES.get(normalized_table)
    if allowed_columns is None or column_name not in allowed_columns:
        return None
    return STATIC_STRING_DICTIONARIES[column_name]
