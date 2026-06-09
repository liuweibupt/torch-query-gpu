import pytest

from tpch_torch.sql import TPC_H_Q1_SQL
from tpch_torch.substrait import UnsupportedPlanError, compile_q1_substrait_plan


Q1_PLAN_JSON = {
    "extensions": [
        {"extensionFunction": {"functionAnchor": 1, "name": "lte"}},
        {"extensionFunction": {"functionAnchor": 2, "name": "sum"}},
        {"extensionFunction": {"functionAnchor": 3, "name": "count"}},
    ],
    "relations": [
        {
            "root": {
                "input": {
                    "sort": {
                        "input": {
                            "aggregate": {
                                "input": {
                                    "project": {
                                        "input": {
                                            "read": {
                                                "baseSchema": {
                                                    "names": [
                                                        "l_returnflag",
                                                        "l_linestatus",
                                                        "l_quantity",
                                                        "l_extendedprice",
                                                        "l_discount",
                                                        "l_tax",
                                                        "l_shipdate",
                                                    ]
                                                },
                                                "filter": {
                                                    "scalarFunction": {
                                                        "functionReference": 1,
                                                        "arguments": [
                                                            {
                                                                "value": {
                                                                    "selection": {
                                                                        "directReference": {
                                                                            "structField": {"field": 6}
                                                                        }
                                                                    }
                                                                }
                                                            },
                                                            {"value": {"literal": {"date": 10471}}},
                                                        ],
                                                    }
                                                },
                                                "namedTable": {"names": ["lineitem"]},
                                            }
                                        }
                                    }
                                },
                                "groupings": [
                                    {
                                        "groupingExpressions": [
                                            {"selection": {"directReference": {"structField": {"field": 0}}}},
                                            {"selection": {"directReference": {"structField": {"field": 1}}}},
                                        ]
                                    }
                                ],
                                "measures": [
                                    {"measure": {"functionReference": 2}},
                                    {"measure": {"functionReference": 3}},
                                ],
                            }
                        },
                        "sorts": [
                            {"expr": {"selection": {"directReference": {"structField": {"field": 0}}}}},
                            {"expr": {"selection": {"directReference": {"structField": {"field": 1}}}}},
                        ],
                    }
                },
                "names": ["l_returnflag", "l_linestatus"],
            }
        }
    ],
}


DUCKDB_PROJECTED_Q1_PLAN_JSON = {
    "relations": [
        {
            "root": {
                "input": {
                    "sort": {
                        "input": {
                            "project": {
                                "input": {
                                    "aggregate": {
                                        "input": {
                                            "project": {
                                                "expressions": [
                                                    {
                                                        "selection": {
                                                            "directReference": {
                                                                "structField": {}
                                                            }
                                                        }
                                                    },
                                                    {
                                                        "selection": {
                                                            "directReference": {
                                                                "structField": {"field": 1}
                                                            }
                                                        }
                                                    },
                                                    {
                                                        "selection": {
                                                            "directReference": {
                                                                "structField": {"field": 2}
                                                            }
                                                        }
                                                    },
                                                ],
                                                "input": {
                                                    "read": {
                                                        "baseSchema": {
                                                            "names": [
                                                                "l_orderkey",
                                                                "l_partkey",
                                                                "l_suppkey",
                                                                "l_linenumber",
                                                                "l_quantity",
                                                                "l_extendedprice",
                                                                "l_discount",
                                                                "l_tax",
                                                                "l_returnflag",
                                                                "l_linestatus",
                                                                "l_shipdate",
                                                            ]
                                                        },
                                                        "filter": {
                                                            "scalarFunction": {
                                                                "arguments": [
                                                                    {
                                                                        "value": {
                                                                            "selection": {
                                                                                "directReference": {
                                                                                    "structField": {
                                                                                        "field": 10
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    },
                                                                    {
                                                                        "value": {
                                                                            "literal": {"date": 10471}
                                                                        }
                                                                    },
                                                                ]
                                                            }
                                                        },
                                                        "namedTable": {"names": ["lineitem"]},
                                                        "projection": {
                                                            "select": {
                                                                "structItems": [
                                                                    {"field": 8},
                                                                    {"field": 9},
                                                                    {"field": 4},
                                                                    {"field": 5},
                                                                    {"field": 6},
                                                                    {"field": 7},
                                                                    {"field": 10},
                                                                ]
                                                            }
                                                        },
                                                    }
                                                },
                                            }
                                        },
                                        "groupings": [
                                            {
                                                "groupingExpressions": [
                                                    {
                                                        "selection": {
                                                            "directReference": {
                                                                "structField": {}
                                                            }
                                                        }
                                                    },
                                                    {
                                                        "selection": {
                                                            "directReference": {
                                                                "structField": {"field": 1}
                                                            }
                                                        }
                                                    },
                                                ]
                                            }
                                        ],
                                    }
                                },
                                "expressions": [
                                    {
                                        "selection": {
                                            "directReference": {"structField": {}}
                                        }
                                    },
                                    {
                                        "selection": {
                                            "directReference": {
                                                "structField": {"field": 1}
                                            }
                                        }
                                    },
                                ],
                            }
                        },
                        "sorts": [
                            {
                                "expr": {
                                    "selection": {
                                        "directReference": {"structField": {}}
                                    }
                                }
                            },
                            {
                                "expr": {
                                    "selection": {
                                        "directReference": {
                                            "structField": {"field": 1}
                                        }
                                    }
                                }
                            },
                        ],
                    }
                },
                "names": ["l_returnflag", "l_linestatus"],
            }
        }
    ],
}


def test_tpch_q1_sql_has_required_semantics():
    normalized = " ".join(TPC_H_Q1_SQL.lower().split())

    assert "from lineitem" in normalized
    assert "l_shipdate <= date '1998-09-02'" in normalized
    assert "group by l_returnflag, l_linestatus" in normalized
    assert "order by l_returnflag, l_linestatus" in normalized


def test_compile_q1_substrait_plan_validates_required_nodes():
    plan = compile_q1_substrait_plan(Q1_PLAN_JSON)

    assert plan.table_name == "lineitem"
    assert plan.shipdate_cutoff_yyyymmdd == 19980902
    assert plan.group_keys == ("l_returnflag", "l_linestatus")
    assert plan.order_keys == ("l_returnflag", "l_linestatus")
    assert plan.required_columns == (
        "l_returnflag",
        "l_linestatus",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_shipdate",
    )


def test_compile_q1_substrait_plan_resolves_duckdb_read_projection_fields():
    plan = compile_q1_substrait_plan(DUCKDB_PROJECTED_Q1_PLAN_JSON)

    assert plan.table_name == "lineitem"
    assert plan.shipdate_cutoff_yyyymmdd == 19980902
    assert plan.group_keys == ("l_returnflag", "l_linestatus")
    assert plan.order_keys == ("l_returnflag", "l_linestatus")


def test_compile_q1_substrait_plan_rejects_missing_aggregate():
    broken_plan = {
        "relations": [
            {
                "root": {
                    "input": {
                        "read": {
                            "namedTable": {"names": ["lineitem"]},
                            "baseSchema": {"names": ["l_shipdate"]},
                        }
                    }
                }
            }
        ]
    }

    with pytest.raises(UnsupportedPlanError, match="aggregate"):
        compile_q1_substrait_plan(broken_plan)
