import duckdb

from tpch_torch.queries.q20 import execute_q20


def test_q20_excludes_partsupp_pairs_without_1994_shipments():
    con = duckdb.connect()
    con.execute("create table part(p_partkey bigint, p_name varchar)")
    con.execute("create table partsupp(ps_partkey bigint, ps_suppkey bigint, ps_availqty bigint)")
    con.execute("create table lineitem(l_partkey bigint, l_suppkey bigint, l_quantity double, l_shipdate date)")
    con.execute("create table supplier(s_suppkey bigint, s_name varchar, s_address varchar, s_nationkey bigint)")
    con.execute("create table nation(n_nationkey bigint, n_name varchar)")
    con.execute("insert into part values (1, 'forest green'), (2, 'forest other')")
    con.execute("insert into partsupp values (1, 1, 10)")
    con.execute("insert into lineitem values (2, 2, 5.0, date '1994-01-01')")
    con.execute("insert into supplier values (1, 'Supplier#1', 'No shipment address', 3)")
    con.execute("insert into nation values (3, 'CANADA')")

    assert execute_q20(con, device="cpu") == []
