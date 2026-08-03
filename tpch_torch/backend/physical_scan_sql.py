"""SQL fragments used by DuckDB-backed physical scans."""

from __future__ import annotations

from tpch_torch.backend.static_dictionaries import static_string_dictionary
from tpch_torch.backend.type_mapping import column_meta_from_duckdb_type
from tpch_torch.record_batch import LogicalDType
from tpch_torch.relational import DATE_COLUMNS_EXTENDED


def scan_select_list(
    table_name: str,
    fetched_columns: tuple[str, ...],
    column_types: dict[str, str],
) -> str:
    return ", ".join(
        scan_select_expression(table_name, column, column_types.get(column, ""))
        for column in fetched_columns
    )


def scan_select_expression(table_name: str, column: str, duckdb_type: str) -> str:
    if column in DATE_COLUMNS_EXTENDED:
        return f"strftime({column}, '%Y%m%d')::integer as {column}"
    decimal_meta = column_meta_from_duckdb_type(duckdb_type)
    if decimal_meta.logical_dtype == LogicalDType.DECIMAL:
        return f"(({column}) * {10 ** int(decimal_meta.scale or 0)})::bigint as {column}"
    dictionary = static_string_dictionary(table_name, column)
    if dictionary is not None:
        return f"{_static_dictionary_case(column, dictionary)} as {column}"
    return column


def _static_dictionary_case(column: str, dictionary: tuple[str, ...]) -> str:
    arms = " ".join(
        f"when {_sql_string_literal(value)} then {index}"
        for index, value in enumerate(dictionary)
    )
    return f"(case {column} {arms} else -1 end)::bigint"


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
