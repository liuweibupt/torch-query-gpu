"""TPC-H query identification and executor support metadata."""

from __future__ import annotations

import re

from tpch_torch.errors import UnsupportedPlanError

QUERY_MARKERS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, ("l_returnflag", "sum_qty", "count_order")),
    (2, ("s_acctbal", "p_type LIKE '%BRASS'", "min(ps_supplycost)")),
    (3, ("c_mktsegment", "BUILDING", "o_shippriority")),
    (4, ("o_orderpriority", "l_commitdate < l_receiptdate", "order_count")),
    (5, ("r_name = 'ASIA'", "n_name", "revenue")),
    (6, ("l_discount BETWEEN 0.05", "l_quantity < 24")),
    (7, ("supp_nation", "cust_nation", "FRANCE", "GERMANY")),
    (8, ("mkt_share", "BRAZIL", "ECONOMY ANODIZED STEEL")),
    (9, ("sum_profit", "%green%", "ps_supplycost")),
    (10, ("l_returnflag = 'R'", "c_acctbal", "LIMIT 20")),
    (11, ("ps_supplycost * ps_availqty", "GERMANY", "0.0001000000")),
    (12, ("l_shipmode IN ('MAIL', 'SHIP')", "high_line_count")),
    (13, ("special%requests", "custdist")),
    (14, ("promo_revenue", "PROMO%")),
    (15, ("WITH revenue AS", "total_revenue", "max(total_revenue)")),
    (16, ("count(DISTINCT ps_suppkey)", "Brand#45", "Customer%Complaints")),
    (17, ("avg_yearly", "Brand#23", "MED BOX")),
    (18, ("sum(l_quantity) > 300", "o_totalprice")),
    (19, ("Brand#12", "Brand#23", "Brand#34")),
    (20, ("forest%", "0.5 * sum(l_quantity)", "CANADA")),
    (21, ("numwait", "SAUDI ARABIA", "l1.l_receiptdate > l1.l_commitdate")),
    (22, ("cntrycode", "substring(c_phone FROM 1 FOR 2)", "numcust")),
)
SUPPORTED_EXECUTOR_QUERIES: frozenset[int] = frozenset(query_id for query_id, _ in QUERY_MARKERS)


def identify_tpch_query(sql: str) -> int:
    normalized = _normalize_sql(sql)
    for query_id, markers in QUERY_MARKERS:
        if all(_normalize_sql(marker) in normalized for marker in markers):
            return query_id
    raise UnsupportedPlanError("SQL text does not match a supported TPC-H query shape")


def is_query_executor_supported(query_id: int) -> bool:
    return query_id in SUPPORTED_EXECUTOR_QUERIES


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).upper()
