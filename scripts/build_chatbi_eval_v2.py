# ruff: noqa: RUF001

"""Build the repository-safe ChatBI v2 frozen dataset and audit artifact.

The v2 fixture is the source of truth for the benchmark.  SQLite materializes
a deterministic oracle; the committed PostgreSQL-backed integration test is
the authoritative validation of every positive query against the actual
fixture.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from knowledge_scope.evaluation.chatbi_evaluation import (
    ExpectedQueryResult,
    _canonical_json,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
    DEFAULT_DATASET_V2,
    DEFAULT_FIXTURE_PATH_V2,
    ChatBIEvaluationCaseV2,
    ChatBIEvaluationDatasetV2,
    ChatBIEvaluationV2AnswerFact,
    ChatBIEvaluationV2Behavior,
    ChatBIEvaluationV2Category,
    ChatBIEvaluationV2Coverage,
    ChatBIEvaluationV2Difficulty,
    write_v2_reviewer_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "docs/benchmarks/a5-7-chatbi-eval-v2-review.md"


def _spec(
    case_id: str,
    split: str,
    category: str,
    difficulty: str,
    question: str,
    sql: str,
    columns: tuple[str, ...],
    *,
    numeric_columns: tuple[int, ...] = (),
    order_sensitive: bool = False,
    repair_applicable: bool = False,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "split": split,
        "category": category,
        "difficulty": difficulty,
        "question": question,
        "sql": sql,
        "columns": columns,
        "numeric_columns": numeric_columns,
        "order_sensitive": order_sensitive,
        "repair_applicable": repair_applicable,
    }


def _negative(
    case_id: str,
    split: str,
    category: str,
    question: str,
    *,
    behavior: str = "refuse",
    failure: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "split": split,
        "category": category,
        "difficulty": "hard",
        "question": question,
        "behavior": behavior,
        "failure": failure,
    }


def _case_specs() -> list[dict[str, object]]:
    """Keep IDs and splits stable while revising the business questions."""
    return [
        _spec(
            "simple-01",
            "dev",
            "simple_filter_projection",
            "easy",
            "列出客户名称，并按客户编号顺序显示。",
            "SELECT customer_name FROM chatbi_demo.customers ORDER BY customer_id",
            ("customer_name",),
            order_sensitive=True,
        ),
        _spec(
            "simple-02",
            "dev",
            "simple_filter_projection",
            "easy",
            "查看每笔销售的编号、地区和金额，并按编号排列。",
            "SELECT s.sale_id, r.region_name AS region, s.amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code ORDER BY s.sale_id",
            ("sale_id", "region", "amount"),
            numeric_columns=(0, 2),
            order_sensitive=True,
        ),
        _spec(
            "simple-03",
            "dev",
            "simple_filter_projection",
            "easy",
            "华南地区的销售发生在什么时候，金额是多少？",
            "SELECT s.sold_on, s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华南' ORDER BY s.sale_id",
            ("sold_on", "amount"),
            numeric_columns=(1,),
        ),
        _spec(
            "simple-04",
            "dev",
            "simple_filter_projection",
            "easy",
            "客户编号为 1 的客户名称是什么？",
            "SELECT customer_name FROM chatbi_demo.customers WHERE customer_id = 1",
            ("customer_name",),
        ),
        _spec(
            "simple-05",
            "test",
            "simple_filter_projection",
            "easy",
            "金额不少于 1200 的销售有哪些编号和日期？",
            "SELECT sale_id, sold_on FROM chatbi_demo.sales WHERE amount >= 1200 ORDER BY sale_id",
            ("sale_id", "sold_on"),
            numeric_columns=(0,),
        ),
        _spec(
            "simple-06",
            "test",
            "simple_filter_projection",
            "medium",
            "华东地区在 2026 年 1 月以来有哪些销售编号和客户编号？",
            "SELECT s.sale_id, s.customer_id FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华东' AND s.sold_on >= DATE '2026-01-01' "
            "ORDER BY s.sale_id",
            ("sale_id", "customer_id"),
            numeric_columns=(0, 1),
        ),
        _spec(
            "simple-07",
            "test",
            "simple_filter_projection",
            "easy",
            "西部市场包含哪些地区？",
            "SELECT region_name FROM chatbi_demo.regions WHERE market = '西部市场' "
            "ORDER BY region_code",
            ("region_name",),
        ),
        _spec(
            "aggregation-01",
            "dev",
            "aggregation",
            "easy",
            "全部销售合计金额是多少？",
            "SELECT SUM(amount) AS total_amount FROM chatbi_demo.sales",
            ("total_amount",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-02",
            "dev",
            "aggregation",
            "easy",
            "当前共有多少笔销售记录？",
            "SELECT COUNT(*) AS sale_count FROM chatbi_demo.sales",
            ("sale_count",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-03",
            "dev",
            "aggregation",
            "easy",
            "平均每笔销售金额是多少？",
            "SELECT AVG(amount) AS average_amount FROM chatbi_demo.sales",
            ("average_amount",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-04",
            "dev",
            "aggregation",
            "easy",
            "所有销售中金额最大的单笔是多少？",
            "SELECT MAX(amount) AS maximum_amount FROM chatbi_demo.sales",
            ("maximum_amount",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-05",
            "dev",
            "aggregation",
            "easy",
            "当前最低的一笔销售金额是多少？",
            "SELECT MIN(amount) AS minimum_amount FROM chatbi_demo.sales",
            ("minimum_amount",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-06",
            "dev",
            "aggregation",
            "easy",
            "客户表中有多少位客户？",
            "SELECT COUNT(*) AS customer_count FROM chatbi_demo.customers",
            ("customer_count",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-07",
            "test",
            "aggregation",
            "medium",
            "华东地区合计销售金额是多少？",
            "SELECT SUM(s.amount) AS east_total FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华东'",
            ("east_total",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-08",
            "test",
            "aggregation",
            "medium",
            "华东地区发生了几笔销售？",
            "SELECT COUNT(*) AS east_sale_count FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华东'",
            ("east_sale_count",),
            numeric_columns=(0,),
        ),
        _spec(
            "aggregation-09",
            "test",
            "aggregation",
            "medium",
            "2026 年 2 月 1 日及以后，华南地区的销售总额是多少？",
            "SELECT SUM(s.amount) AS south_total FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华南' AND s.sold_on >= DATE '2026-02-01'",
            ("south_total",),
            numeric_columns=(0,),
        ),
        _spec(
            "group-by-01",
            "dev",
            "group_by",
            "medium",
            "按地区汇总销售笔数和金额。",
            "SELECT r.region_name AS region, COUNT(s.sale_id) AS sale_count, "
            "SUM(s.amount) AS total_amount FROM chatbi_demo.regions AS r "
            "LEFT JOIN chatbi_demo.sales AS s ON s.region_code = r.region_code "
            "GROUP BY r.region_code, r.region_name ORDER BY r.region_code",
            ("region", "sale_count", "total_amount"),
            numeric_columns=(1, 2),
        ),
        _spec(
            "group-by-02",
            "dev",
            "group_by",
            "medium",
            "按客户编号统计销售笔数，并按客户编号排列。",
            "SELECT customer_id, COUNT(*) AS sale_count FROM chatbi_demo.sales "
            "GROUP BY customer_id ORDER BY customer_id",
            ("customer_id", "sale_count"),
            numeric_columns=(0, 1),
            order_sensitive=True,
        ),
        _spec(
            "group-by-03",
            "dev",
            "group_by",
            "medium",
            "按客户编号汇总销售金额。",
            "SELECT customer_id, SUM(amount) AS total_amount FROM chatbi_demo.sales "
            "GROUP BY customer_id ORDER BY customer_id",
            ("customer_id", "total_amount"),
            numeric_columns=(0, 1),
        ),
        _spec(
            "group-by-04",
            "dev",
            "group_by",
            "medium",
            "比较各地区的最高销售金额。",
            "SELECT r.region_name AS region, MAX(s.amount) AS maximum_amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name "
            "ORDER BY r.region_code",
            ("region", "maximum_amount"),
            numeric_columns=(1,),
        ),
        _spec(
            "group-by-05",
            "dev",
            "group_by",
            "medium",
            "各地区平均每笔销售金额是多少？",
            "SELECT r.region_name AS region, AVG(s.amount) AS average_amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name "
            "ORDER BY r.region_code",
            ("region", "average_amount"),
            numeric_columns=(1,),
        ),
        _spec(
            "group-by-06",
            "test",
            "group_by",
            "medium",
            "按客户编号和名称统计每位客户的销售笔数，包含没有销售记录的客户。",
            "SELECT c.customer_id, c.customer_name, COUNT(s.sale_id) AS sale_count "
            "FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name "
            "ORDER BY c.customer_id",
            ("customer_id", "customer_name", "sale_count"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "group-by-07",
            "test",
            "group_by",
            "medium",
            "各地区最小的一笔销售金额是多少？",
            "SELECT r.region_name AS region, MIN(s.amount) AS minimum_amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name "
            "ORDER BY r.region_code",
            ("region", "minimum_amount"),
            numeric_columns=(1,),
        ),
        _spec(
            "group-by-08",
            "test",
            "group_by",
            "medium",
            "各地区分别有多少笔销售？",
            "SELECT r.region_name AS region, COUNT(*) AS sale_count "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name "
            "ORDER BY sale_count DESC, r.region_code",
            ("region", "sale_count"),
            numeric_columns=(1,),
        ),
        _spec(
            "top-k-01",
            "dev",
            "ordering_top_k",
            "medium",
            "按金额从高到低取前两笔销售；金额相同时销售编号较小的优先。",
            "SELECT sale_id, customer_id, amount FROM chatbi_demo.sales "
            "ORDER BY amount DESC, sale_id ASC LIMIT 2",
            ("sale_id", "customer_id", "amount"),
            numeric_columns=(0, 1, 2),
            order_sensitive=True,
        ),
        _spec(
            "top-k-02",
            "dev",
            "ordering_top_k",
            "medium",
            "最早发生的销售是哪一笔？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales "
            "ORDER BY sold_on ASC, sale_id LIMIT 1",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
            order_sensitive=True,
        ),
        _spec(
            "top-k-03",
            "dev",
            "ordering_top_k",
            "medium",
            "最近发生的销售是哪一笔？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales "
            "ORDER BY sold_on DESC, sale_id DESC LIMIT 1",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
            order_sensitive=True,
        ),
        _spec(
            "top-k-04",
            "dev",
            "ordering_top_k",
            "medium",
            "金额最低的销售是哪一笔？",
            "SELECT sale_id, amount FROM chatbi_demo.sales ORDER BY amount ASC, sale_id LIMIT 1",
            ("sale_id", "amount"),
            numeric_columns=(0, 1),
            order_sensitive=True,
        ),
        _spec(
            "top-k-05",
            "test",
            "ordering_top_k",
            "medium",
            "请按客户编号从大到小列出客户名称。",
            "SELECT customer_id, customer_name FROM chatbi_demo.customers "
            "ORDER BY customer_id DESC",
            ("customer_id", "customer_name"),
            numeric_columns=(0,),
            order_sensitive=True,
        ),
        _spec(
            "top-k-06",
            "test",
            "ordering_top_k",
            "medium",
            "请把所有销售按金额从低到高列出；金额相同时按销售编号从小到大排列。",
            "SELECT sale_id, amount FROM chatbi_demo.sales ORDER BY amount ASC, sale_id ASC",
            ("sale_id", "amount"),
            numeric_columns=(0, 1),
            order_sensitive=True,
        ),
        _spec(
            "top-k-07",
            "test",
            "ordering_top_k",
            "medium",
            "华东地区金额最高的销售是哪一笔？",
            "SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华东' ORDER BY s.amount DESC, s.sale_id LIMIT 1",
            ("sale_id", "amount"),
            numeric_columns=(0, 1),
            order_sensitive=True,
        ),
        _spec(
            "predicate-01",
            "dev",
            "multi_predicate",
            "medium",
            "华东地区中金额超过 1300 的销售有哪些？",
            "SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华东' AND s.amount > 1300 ORDER BY s.sale_id",
            ("sale_id", "amount"),
            numeric_columns=(0, 1),
        ),
        _spec(
            "predicate-02",
            "dev",
            "multi_predicate",
            "medium",
            "2026 年 1 月 12 日起且金额低于 1000 的销售有哪些？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales "
            "WHERE sold_on >= DATE '2026-01-12' AND amount < 1000 "
            "ORDER BY sold_on, sale_id",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "predicate-03",
            "dev",
            "multi_predicate",
            "medium",
            "华东地区中编号小于 3 的销售有哪些？",
            "SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华东' AND s.sale_id < 3 ORDER BY s.sale_id",
            ("sale_id", "amount"),
            numeric_columns=(0, 1),
        ),
        _spec(
            "predicate-04",
            "dev",
            "multi_predicate",
            "medium",
            "华东地区金额在 1000 到 1300 之间的销售有哪些？",
            "SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE r.region_name = '华东' AND s.amount BETWEEN 1000 AND 1300 "
            "ORDER BY s.sale_id",
            ("sale_id", "amount"),
            numeric_columns=(0, 1),
        ),
        _spec(
            "predicate-05",
            "test",
            "multi_predicate",
            "medium",
            "客户编号为 1 且早于 2026 年 3 月 1 日的销售有哪些？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales "
            "WHERE customer_id = 1 AND sold_on < DATE '2026-03-01' "
            "ORDER BY sold_on, sale_id",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "predicate-06",
            "test",
            "multi_predicate",
            "hard",
            "华东或华南地区中金额至少为 860.50 的销售有哪些？",
            "SELECT s.sale_id, r.region_name AS region, s.amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code WHERE r.region_name IN ('华东', '华南') "
            "AND s.amount >= 860.50 ORDER BY s.sale_id",
            ("sale_id", "region", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "predicate-07",
            "test",
            "multi_predicate",
            "hard",
            "非华东地区且金额超过 800 的销售有哪些？",
            "SELECT s.sale_id, r.region_name AS region, s.amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code WHERE r.region_name <> '华东' "
            "AND s.amount > 800 ORDER BY s.sale_id",
            ("sale_id", "region", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "join-01",
            "dev",
            "join",
            "medium",
            "列出每笔销售对应的客户名称和金额。",
            "SELECT s.sale_id, c.customer_name, s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
            "ORDER BY s.sale_id",
            ("sale_id", "customer_name", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "join-02",
            "dev",
            "join",
            "medium",
            "按客户编号和名称查看他们对应的销售编号。",
            "SELECT c.customer_id, c.customer_name, s.sale_id "
            "FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id ORDER BY c.customer_id, s.sale_id",
            ("customer_id", "customer_name", "sale_id"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "join-03",
            "dev",
            "join",
            "medium",
            "只看华东销售时，对应的客户名称和金额是什么？",
            "SELECT s.sale_id, c.customer_name, s.amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c "
            "ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code WHERE r.region_name = '华东' "
            "ORDER BY s.sale_id",
            ("sale_id", "customer_name", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "join-04",
            "dev",
            "join",
            "medium",
            "每笔销售对应客户的名称和日期是什么？",
            "SELECT s.sale_id, c.customer_name, s.sold_on "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c "
            "ON c.customer_id = s.customer_id ORDER BY s.sale_id",
            ("sale_id", "customer_name", "sold_on"),
            numeric_columns=(0,),
        ),
        _spec(
            "join-05",
            "dev",
            "join",
            "hard",
            "每位客户（含没有销售的客户）最早的一笔销售日期是什么？",
            "SELECT c.customer_id, c.customer_name, MIN(s.sold_on) AS first_sale_on "
            "FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name "
            "ORDER BY c.customer_id",
            ("customer_id", "customer_name", "first_sale_on"),
            numeric_columns=(0,),
        ),
        _spec(
            "join-06",
            "dev",
            "join",
            "medium",
            "金额超过 1000 的销售分别属于哪位客户？",
            "SELECT s.sale_id, c.customer_name, s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
            "WHERE s.amount > 1000 ORDER BY s.sale_id",
            ("sale_id", "customer_name", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "join-07",
            "test",
            "join",
            "medium",
            "哪些客户至少有一笔销售？",
            "SELECT DISTINCT c.customer_id, c.customer_name "
            "FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id ORDER BY c.customer_id",
            ("customer_id", "customer_name"),
            numeric_columns=(0,),
        ),
        _spec(
            "join-08",
            "test",
            "join",
            "medium",
            "按客户编号查看每笔销售所在地区和金额。",
            "SELECT c.customer_id, c.customer_name, r.region_name AS region, s.amount "
            "FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code ORDER BY c.customer_id, s.sale_id",
            ("customer_id", "customer_name", "region", "amount"),
            numeric_columns=(0, 3),
        ),
        _spec(
            "join-09",
            "test",
            "join",
            "medium",
            "2026 年 2 月发生的销售对应哪位客户？",
            "SELECT s.sale_id, c.customer_name, s.sold_on "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c "
            "ON c.customer_id = s.customer_id WHERE s.sold_on >= DATE '2026-02-01' "
            "AND s.sold_on < DATE '2026-03-01' ORDER BY s.sale_id",
            ("sale_id", "customer_name", "sold_on"),
            numeric_columns=(0,),
        ),
        _spec(
            "join-10",
            "test",
            "join",
            "hard",
            "每位客户涉及过多少个不同销售地区？包含没有销售记录的客户。",
            "SELECT c.customer_id, c.customer_name, COUNT(DISTINCT s.region_code) AS region_count "
            "FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name "
            "ORDER BY c.customer_id",
            ("customer_id", "customer_name", "region_count"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "multi-join-01",
            "dev",
            "multi_table_join",
            "hard",
            "按客户和地区查看销售笔数及总额。",
            "SELECT c.customer_id, c.customer_name, r.region_name AS region, "
            "COUNT(s.sale_id) AS sale_count, SUM(s.amount) AS total_amount "
            "FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code GROUP BY c.customer_id, c.customer_name, "
            "r.region_code, r.region_name ORDER BY c.customer_id, r.region_code",
            ("customer_id", "customer_name", "region", "sale_count", "total_amount"),
            numeric_columns=(0, 3, 4),
        ),
        _spec(
            "multi-join-02",
            "dev",
            "multi_table_join",
            "hard",
            "按销售市场和地区汇总销售金额，并显示涉及的客户数。",
            "SELECT r.market, r.region_name AS region, "
            "COUNT(DISTINCT c.customer_id) AS customer_count, "
            "SUM(s.amount) AS total_amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "GROUP BY r.market, r.region_code, r.region_name ORDER BY r.market, r.region_code",
            ("market", "region", "customer_count", "total_amount"),
            numeric_columns=(2, 3),
        ),
        _spec(
            "multi-join-03",
            "dev",
            "multi_table_join",
            "hard",
            "列出每个地区有销售记录的客户及其销售笔数。",
            "SELECT r.region_name AS region, c.customer_id, c.customer_name, "
            "COUNT(*) AS sale_count FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "GROUP BY r.region_code, r.region_name, c.customer_id, c.customer_name "
            "ORDER BY r.region_code, c.customer_id",
            ("region", "customer_id", "customer_name", "sale_count"),
            numeric_columns=(1, 3),
        ),
        _spec(
            "multi-join-04",
            "dev",
            "multi_table_join",
            "hard",
            "找出金额最高的销售及其客户和地区。",
            "SELECT s.sale_id, c.customer_name, r.region_name AS region, s.amount "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c "
            "ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code ORDER BY s.amount DESC, s.sale_id LIMIT 1",
            ("sale_id", "customer_name", "region", "amount"),
            numeric_columns=(0, 3),
            order_sensitive=True,
        ),
        _spec(
            "multi-join-05",
            "test",
            "multi_table_join",
            "hard",
            "按客户编号和地区统计销售笔数、金额，并显示所属市场。",
            "SELECT c.customer_id, r.region_name AS region, r.market, COUNT(*) AS sale_count, "
            "SUM(s.amount) AS total_amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "GROUP BY c.customer_id, r.region_code, r.region_name, r.market "
            "ORDER BY c.customer_id, r.region_code",
            ("customer_id", "region", "market", "sale_count", "total_amount"),
            numeric_columns=(0, 3, 4),
        ),
        _spec(
            "multi-join-06",
            "test",
            "multi_table_join",
            "hard",
            "哪个销售市场的累计销售额最高，以及该市场覆盖了多少位客户？",
            "SELECT r.market, SUM(s.amount) AS total_amount, "
            "COUNT(DISTINCT c.customer_id) AS customer_count "
            "FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c "
            "ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code GROUP BY r.market "
            "ORDER BY total_amount DESC, r.market LIMIT 1",
            ("market", "total_amount", "customer_count"),
            numeric_columns=(1, 2),
            order_sensitive=True,
        ),
        _spec(
            "date-01",
            "dev",
            "date_filter",
            "easy",
            "2026 年 1 月有多少笔销售？",
            "SELECT COUNT(*) AS sale_count FROM chatbi_demo.sales "
            "WHERE sold_on >= DATE '2026-01-01' AND sold_on < DATE '2026-02-01'",
            ("sale_count",),
            numeric_columns=(0,),
        ),
        _spec(
            "date-02",
            "dev",
            "date_filter",
            "easy",
            "2026 年 2 月发生了哪些销售？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales "
            "WHERE sold_on >= DATE '2026-02-01' AND sold_on < DATE '2026-03-01' "
            "ORDER BY sale_id",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "date-03",
            "dev",
            "date_filter",
            "easy",
            "2026 年第一个月有哪些销售编号和日期？",
            "SELECT sale_id, sold_on FROM chatbi_demo.sales "
            "WHERE sold_on >= DATE '2026-01-01' AND sold_on < DATE '2026-02-01' "
            "ORDER BY sale_id",
            ("sale_id", "sold_on"),
            numeric_columns=(0,),
        ),
        _spec(
            "date-04",
            "test",
            "date_filter",
            "medium",
            "早于 2026 年 2 月 1 日（不含当日）的销售总金额是多少？",
            "SELECT SUM(amount) AS total_amount FROM chatbi_demo.sales "
            "WHERE sold_on < DATE '2026-02-01'",
            ("total_amount",),
            numeric_columns=(0,),
        ),
        _spec(
            "date-05",
            "test",
            "date_filter",
            "medium",
            "2026 年 4 月 1 日至 2026 年 4 月 30 日（含首尾）有哪些销售？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales "
            "WHERE sold_on >= DATE '2026-04-01' AND sold_on <= DATE '2026-04-30' "
            "ORDER BY sold_on, sale_id",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "null-01",
            "dev",
            "null_boundary",
            "medium",
            "哪些客户从未产生销售记录？",
            "SELECT c.customer_id, c.customer_name FROM chatbi_demo.customers AS c "
            "LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id "
            "WHERE s.sale_id IS NULL ORDER BY c.customer_id",
            ("customer_id", "customer_name"),
            numeric_columns=(0,),
        ),
        _spec(
            "null-02",
            "dev",
            "null_boundary",
            "medium",
            "每位客户的销售笔数和总额是多少？没有销售的客户总额按 0 计。",
            "SELECT c.customer_id, c.customer_name, COUNT(s.sale_id) AS sale_count, "
            "COALESCE(SUM(s.amount), 0) AS total_amount FROM chatbi_demo.customers AS c "
            "LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id "
            "GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id",
            ("customer_id", "customer_name", "sale_count", "total_amount"),
            numeric_columns=(0, 2, 3),
        ),
        _spec(
            "null-03",
            "dev",
            "null_boundary",
            "medium",
            "列出客户名称及最近销售日期，并保留无销售客户的空日期。",
            "SELECT c.customer_id, c.customer_name, MAX(s.sold_on) AS latest_sale_on "
            "FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name "
            "ORDER BY c.customer_id",
            ("customer_id", "customer_name", "latest_sale_on"),
            numeric_columns=(0,),
        ),
        _spec(
            "null-04",
            "test",
            "null_boundary",
            "medium",
            "2026 年 6 月哪些客户没有销售记录？",
            "SELECT c.customer_id, c.customer_name FROM chatbi_demo.customers AS c "
            "LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id "
            "AND s.sold_on >= DATE '2026-06-01' AND s.sold_on < DATE '2026-07-01' "
            "WHERE s.sale_id IS NULL ORDER BY c.customer_id",
            ("customer_id", "customer_name"),
            numeric_columns=(0,),
        ),
        _spec(
            "empty-01",
            "dev",
            "empty",
            "easy",
            "销售编号为 999 的记录是什么？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales WHERE sale_id = 999",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "empty-02",
            "dev",
            "empty",
            "easy",
            "客户名称以“不存在”开头的客户有哪些？",
            "SELECT customer_id, customer_name FROM chatbi_demo.customers "
            "WHERE customer_name LIKE '不存在%'",
            ("customer_id", "customer_name"),
            numeric_columns=(0,),
        ),
        _spec(
            "empty-03",
            "test",
            "empty",
            "easy",
            "2027 年 1 月 1 日及以后还有销售记录吗？",
            "SELECT sale_id, sold_on, amount FROM chatbi_demo.sales "
            "WHERE sold_on >= DATE '2027-01-01'",
            ("sale_id", "sold_on", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "derived-01",
            "dev",
            "alias_derived",
            "medium",
            "每笔销售金额比全部销售的平均金额高或低多少？",
            "SELECT sale_id, amount, amount - (SELECT AVG(amount) FROM chatbi_demo.sales) AS delta "
            "FROM chatbi_demo.sales ORDER BY sale_id",
            ("sale_id", "amount", "delta"),
            numeric_columns=(0, 1, 2),
        ),
        _spec(
            "derived-02",
            "dev",
            "alias_derived",
            "medium",
            "各地区销售笔数占全部销售笔数的比例是多少？",
            "SELECT r.region_name, "
            "AVG(CAST(s.region_code = r.region_code AS INTEGER)) "
            "AS sales_share FROM chatbi_demo.regions AS r "
            "CROSS JOIN chatbi_demo.sales AS s "
            "GROUP BY r.region_code, r.region_name ORDER BY r.region_code",
            ("region_name", "sales_share"),
            numeric_columns=(1,),
        ),
        _spec(
            "derived-03",
            "dev",
            "alias_derived",
            "medium",
            "只统计有销售记录的客户，每位客户的平均销售金额是多少？",
            "SELECT c.customer_id, c.customer_name, AVG(s.amount) AS average_amount "
            "FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name "
            "ORDER BY c.customer_id",
            ("customer_id", "customer_name", "average_amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "derived-04",
            "test",
            "alias_derived",
            "hard",
            "每位有销售记录的客户，其销售总额比这些客户的平均销售总额高或低多少？",
            "WITH customer_totals AS ("
            "SELECT c.customer_id, c.customer_name, SUM(s.amount) AS total_amount "
            "FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s "
            "ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name) "
            "SELECT customer_id, customer_name, total_amount, "
            "total_amount - (SELECT AVG(total_amount) FROM customer_totals) AS delta "
            "FROM customer_totals ORDER BY customer_id",
            ("customer_id", "customer_name", "total_amount", "delta"),
            numeric_columns=(0, 2, 3),
        ),
        _spec(
            "cte-set-01",
            "dev",
            "cte_set_operation",
            "hard",
            "哪些客户的累计销售额高于有销售客户的平均累计销售额？",
            "WITH customer_totals AS ("
            "SELECT customer_id, SUM(amount) AS total_amount FROM chatbi_demo.sales "
            "GROUP BY customer_id) SELECT customer_id, total_amount FROM customer_totals "
            "WHERE total_amount > (SELECT AVG(total_amount) FROM customer_totals) "
            "ORDER BY customer_id",
            ("customer_id", "total_amount"),
            numeric_columns=(0, 1),
        ),
        _spec(
            "cte-set-02",
            "dev",
            "cte_set_operation",
            "hard",
            "哪些销售金额高于其所在地区的平均销售金额？",
            "WITH region_average AS ("
            "SELECT region_code, AVG(amount) AS average_amount FROM chatbi_demo.sales "
            "GROUP BY region_code) SELECT s.sale_id, r.region_name, s.amount "
            "FROM chatbi_demo.sales AS s JOIN region_average AS a "
            "ON a.region_code = s.region_code JOIN chatbi_demo.regions AS r "
            "ON r.region_code = s.region_code WHERE s.amount > a.average_amount "
            "ORDER BY s.sale_id",
            ("sale_id", "region_name", "amount"),
            numeric_columns=(0, 2),
        ),
        _spec(
            "cte-set-03",
            "dev",
            "cte_set_operation",
            "hard",
            "哪些地区在 2026 年 2 月和 2026 年 3 月都出现过销售？",
            "SELECT r.region_name FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE s.sold_on >= DATE '2026-02-01' AND s.sold_on < DATE '2026-03-01' "
            "INTERSECT SELECT r2.region_name FROM chatbi_demo.sales AS s2 "
            "JOIN chatbi_demo.regions AS r2 ON r2.region_code = s2.region_code "
            "WHERE s2.sold_on >= DATE '2026-03-01' AND s2.sold_on < DATE '2026-04-01' "
            "ORDER BY region_name",
            ("region_name",),
        ),
        _spec(
            "cte-set-04",
            "test",
            "cte_set_operation",
            "hard",
            "哪些地区在 2026 年 2 月出现过、但在 2026 年 3 月没有出现？",
            "SELECT r.region_name FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
            "WHERE s.sold_on >= DATE '2026-02-01' AND s.sold_on < DATE '2026-03-01' "
            "EXCEPT SELECT r2.region_name FROM chatbi_demo.sales AS s2 "
            "JOIN chatbi_demo.regions AS r2 ON r2.region_code = s2.region_code "
            "WHERE s2.sold_on >= DATE '2026-03-01' AND s2.sold_on < DATE '2026-04-01' "
            "ORDER BY region_name",
            ("region_name",),
        ),
        _negative(
            "ambiguous-01",
            "dev",
            "ambiguous_question",
            "销售情况怎么样？",
            behavior="clarify_or_refuse_safely",
            failure="ambiguous_question",
        ),
        _negative(
            "ambiguous-02",
            "test",
            "ambiguous_question",
            "客户表现如何？",
            behavior="clarify_or_refuse_safely",
            failure="ambiguous_question",
        ),
        _negative(
            "ambiguous-03",
            "test",
            "ambiguous_question",
            "哪些数据更重要？",
            behavior="clarify_or_refuse_safely",
            failure="ambiguous_question",
        ),
        _negative(
            "unsupported-01",
            "dev",
            "unsupported_request",
            "删除金额小于 1000 的销售记录。",
            failure="unsupported_request",
        ),
        _negative(
            "unsupported-02",
            "dev",
            "unsupported_request",
            "把华东销售额改成 0。",
            failure="unsupported_request",
        ),
        _negative(
            "unsupported-03",
            "test",
            "unsupported_request",
            "创建一张新的销售汇总表。",
            failure="unsupported_request",
        ),
    ]


def _sqlite_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("ATTACH DATABASE ':memory:' AS chatbi_demo")
    fixture = (ROOT / DEFAULT_FIXTURE_PATH_V2).read_text(encoding="utf-8")
    fixture = re.sub(r"(?m)^\s*--.*$", "", fixture)
    fixture = re.sub(r"(?is)DROP\s+SCHEMA.*?;", "", fixture)
    fixture = re.sub(r"(?is)CREATE\s+SCHEMA.*?;", "", fixture)
    fixture = re.sub(r"(?m)^\s*COMMENT\s+ON.*$", "", fixture)
    fixture = fixture.replace("REFERENCES chatbi_demo.", "REFERENCES ")
    fixture = re.sub(r"\bDATE\s*'([^']*)'", r"'\1'", fixture, flags=re.IGNORECASE)
    for statement in fixture.split(";"):
        if statement.strip():
            connection.execute(statement)
    return connection


def _sqlite_sql(sql: str) -> str:
    return re.sub(r"\bDATE\s*'([^']*)'", r"'\1'", sql, flags=re.IGNORECASE)


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _fact_values(columns: tuple[str, ...], row: tuple[object, ...]) -> dict[str, object]:
    return {column: _json_value(value) for column, value in zip(columns, row, strict=True)}


def _build_case(
    connection: sqlite3.Connection,
    spec: dict[str, object],
) -> ChatBIEvaluationCaseV2:
    if "behavior" in spec:
        return ChatBIEvaluationCaseV2(
            case_id=spec["case_id"],
            split=spec["split"],
            question=spec["question"],
            category=ChatBIEvaluationV2Category(spec["category"]),
            difficulty=ChatBIEvaluationV2Difficulty(spec["difficulty"]),
            positive=False,
            order_sensitive=False,
            expected_behavior=ChatBIEvaluationV2Behavior(spec["behavior"]),
            expected_failure_category=spec["failure"],
        )

    columns = spec["columns"]
    assert isinstance(columns, tuple)
    sql = spec["sql"]
    assert isinstance(sql, str)
    rows = [
        tuple(_json_value(value) for value in row) for row in connection.execute(_sqlite_sql(sql))
    ]
    numeric_columns = list(spec["numeric_columns"])
    structured_facts = [
        ChatBIEvaluationV2AnswerFact(
            kind=(
                "null"
                if any(value is None for value in row)
                else ("scalar" if len(columns) == 1 else "row")
            ),
            values=_fact_values(columns, row),
        )
        for row in rows
    ]
    if not structured_facts:
        structured_facts = [ChatBIEvaluationV2AnswerFact(kind="empty")]
    text_facts: list[str] = []
    for row in rows:
        for value in row:
            if value is not None:
                text = str(value)
                if text not in text_facts:
                    text_facts.append(text)
    return ChatBIEvaluationCaseV2(
        case_id=spec["case_id"],
        split=spec["split"],
        question=spec["question"],
        category=ChatBIEvaluationV2Category(spec["category"]),
        difficulty=ChatBIEvaluationV2Difficulty(spec["difficulty"]),
        positive=True,
        order_sensitive=spec["order_sensitive"],
        expected_behavior=ChatBIEvaluationV2Behavior.ANSWERABLE,
        reference_sql=sql,
        expected_result=ExpectedQueryResult(
            columns=list(columns),
            rows=[list(row) for row in rows],
            numeric_columns=numeric_columns,
            row_order_sensitive=spec["order_sensitive"],
        ),
        expected_answer_facts=text_facts[:24],
        structured_answer_facts=structured_facts,
        repair_applicable=spec["repair_applicable"],
    )


def build_dataset() -> ChatBIEvaluationDatasetV2:
    connection = _sqlite_connection()
    try:
        cases = [_build_case(connection, spec) for spec in _case_specs()]
    finally:
        connection.close()
    category_counts = {category.value: 0 for category in ChatBIEvaluationV2Category}
    difficulty_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category.value] += 1
        difficulty_counts[case.difficulty.value] = (
            difficulty_counts.get(case.difficulty.value, 0) + 1
        )
    fixture_path = ROOT / DEFAULT_FIXTURE_PATH_V2
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    coverage = ChatBIEvaluationV2Coverage(
        case_count=len(cases),
        dev_count=sum(case.split == "dev" for case in cases),
        test_count=sum(case.split == "test" for case in cases),
        positive_count=sum(case.positive for case in cases),
        negative_count=sum(not case.positive for case in cases),
        category_counts=category_counts,
        difficulty_counts=dict(sorted(difficulty_counts.items())),
    )
    dataset = ChatBIEvaluationDatasetV2(
        datasource_fixture=DEFAULT_FIXTURE_PATH_V2.as_posix(),
        datasource_fixture_sha256=fixture_sha256,
        cases=cases,
        coverage=coverage,
    )
    return ChatBIEvaluationDatasetV2.model_validate(
        {**dataset.model_dump(mode="json"), "dataset_fingerprint": dataset.fingerprint}
    )


def main() -> None:
    dataset = build_dataset()
    output = ROOT / DEFAULT_DATASET_V2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dataset.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_v2_reviewer_artifact(dataset, ROOT / REVIEW_PATH)
    print(
        _canonical_json(
            {
                "fixture_sha256": dataset.datasource_fixture_sha256,
                "dataset_fingerprint": dataset.fingerprint,
                "coverage": dataset.coverage.model_dump(mode="json"),
            }
        )
    )


if __name__ == "__main__":
    main()
