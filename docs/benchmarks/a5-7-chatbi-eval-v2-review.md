# ChatBI 评测数据集 v2 审阅清单

状态: `human_reviewed_frozen` (已完成最终人工审核与技术冻结审计)
数据集指纹: `60c75c597da8fc71a0fa5b25d335b63410b44a4ab3a403da40ca72c5ae375ab3`
fixture 指纹: `fd972106c39c7a8b31b57975118708e213a32e4e008ee13fafc15b1ea1b5182d`

每个条目用于记录问题自然性、SQL oracle、结果语义和负例行为的冻结审计。
reference SQL 与期望结果不会进入模型输入；条目级人工审核结论已经应用，本文件不是待填写的审核表单。
`repair-applicable` 仅是审核元数据，不作为 v2 修复率的分母；修复尝试以运行时记录为准。

## simple-01

- split: `dev`
- category: `simple_filter_projection`
- difficulty: `easy`
- question: 列出客户名称，并按客户编号顺序显示。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT customer_name FROM chatbi_demo.customers ORDER BY customer_id`
- expected result: `{"columns":["customer_name"],"omitted_rows":2,"row_order_sensitive":true,"rows":[["北辰制造"],["星河贸易"],["远山科技"],["江南医药"],["海岳能源"],["北辰制造"],["晨光物流"],["云杉教育"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"scalar","values":{"customer_name":"北辰制造"}},{"kind":"scalar","values":{"customer_name":"星河贸易"}},{"kind":"scalar","values":{"customer_name":"远山科技"}},{"kind":"scalar","values":{"customer_name":"江南医药"}},{"kind":"scalar","values":{"customer_name":"海岳能源"}},{"kind":"scalar","values":{"customer_name":"北辰制造"}},{"kind":"scalar","values":{"customer_name":"晨光物流"}},{"kind":"scalar","values":{"customer_name":"云杉教育"}},{"kind":"scalar","values":{"customer_name":"南岭食品"}},{"kind":"scalar","values":{"customer_name":"未成交客户"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## simple-02

- split: `dev`
- category: `simple_filter_projection`
- difficulty: `easy`
- question: 查看每笔销售的编号、地区和金额，并按编号排列。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, r.region_name AS region, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","region","amount"],"omitted_rows":22,"row_order_sensitive":true,"rows":[[1,"华东",1200],[2,"华南",860.5],[3,"华北",1540.25],[4,"华东",999.99],[5,"西南",2100],[6,"华南",750],[7,"华东",1200],[8,"西北",430]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"region":"华东","sale_id":1}},{"kind":"row","values":{"amount":860.5,"region":"华南","sale_id":2}},{"kind":"row","values":{"amount":1540.25,"region":"华北","sale_id":3}},{"kind":"row","values":{"amount":999.99,"region":"华东","sale_id":4}},{"kind":"row","values":{"amount":2100,"region":"西南","sale_id":5}},{"kind":"row","values":{"amount":750,"region":"华南","sale_id":6}},{"kind":"row","values":{"amount":1200,"region":"华东","sale_id":7}},{"kind":"row","values":{"amount":430,"region":"西北","sale_id":8}},{"kind":"row","values":{"amount":1750,"region":"华北","sale_id":9}},{"kind":"row","values":{"amount":860.5,"region":"华南","sale_id":10}},{"kind":"row","values":{"amount":1120,"region":"西南","sale_id":11}},{"kind":"row","values":{"amount":3100,"region":"华东","sale_id":12}},{"kind":"row","values":{"amount":640,"region":"华南","sale_id":13}},{"kind":"row","values":{"amount":980,"region":"西北","sale_id":14}},{"kind":"row","values":{"amount":1200,"region":"华北","sale_id":15...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## simple-03

- split: `dev`
- category: `simple_filter_projection`
- difficulty: `easy`
- question: 华南地区的销售发生在什么时候，金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sold_on, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华南' ORDER BY s.sale_id`
- expected result: `{"columns":["sold_on","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["2026-01-05",860.5],["2026-02-02",750],["2026-03-01",860.5],["2026-03-31",640],["2026-05-01",1320],["2026-06-01",1450],["2026-01-20",2500],["2026-06-10",1200]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":860.5,"sold_on":"2026-01-05"}},{"kind":"row","values":{"amount":750,"sold_on":"2026-02-02"}},{"kind":"row","values":{"amount":860.5,"sold_on":"2026-03-01"}},{"kind":"row","values":{"amount":640,"sold_on":"2026-03-31"}},{"kind":"row","values":{"amount":1320,"sold_on":"2026-05-01"}},{"kind":"row","values":{"amount":1450,"sold_on":"2026-06-01"}},{"kind":"row","values":{"amount":2500,"sold_on":"2026-01-20"}},{"kind":"row","values":{"amount":1200,"sold_on":"2026-06-10"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## simple-04

- split: `dev`
- category: `simple_filter_projection`
- difficulty: `easy`
- question: 客户编号为 1 的客户名称是什么？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT customer_name FROM chatbi_demo.customers WHERE customer_id = 1`
- expected result: `{"columns":["customer_name"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["北辰制造"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"customer_name":"北辰制造"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## simple-05

- split: `test`
- category: `simple_filter_projection`
- difficulty: `easy`
- question: 金额不少于 1200 的销售有哪些编号和日期？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on FROM chatbi_demo.sales WHERE amount >= 1200 ORDER BY sale_id`
- expected result: `{"columns":["sale_id","sold_on"],"omitted_rows":7,"row_order_sensitive":false,"rows":[[1,"2026-01-01"],[3,"2026-01-12"],[5,"2026-02-01"],[7,"2026-02-10"],[9,"2026-02-28"],[12,"2026-03-15"],[15,"2026-04-10"],[16,"2026-04-20"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"sale_id":1,"sold_on":"2026-01-01"}},{"kind":"row","values":{"sale_id":3,"sold_on":"2026-01-12"}},{"kind":"row","values":{"sale_id":5,"sold_on":"2026-02-01"}},{"kind":"row","values":{"sale_id":7,"sold_on":"2026-02-10"}},{"kind":"row","values":{"sale_id":9,"sold_on":"2026-02-28"}},{"kind":"row","values":{"sale_id":12,"sold_on":"2026-03-15"}},{"kind":"row","values":{"sale_id":15,"sold_on":"2026-04-10"}},{"kind":"row","values":{"sale_id":16,"sold_on":"2026-04-20"}},{"kind":"row","values":{"sale_id":18,"sold_on":"2026-05-01"}},{"kind":"row","values":{"sale_id":20,"sold_on":"2026-05-31"}},{"kind":"row","values":{"sale_id":21,"sold_on":"2026-06-01"}},{"kind":"row","values":{"sale_id":23,"sold_on":"2026-06-15"}},{"kind":"row","values":{"sale_id":25,"sold_on":"2026-01-20"}},{"kind":"row","values":{"sale_id":27,"sold_on":"2026-03-20"}},{"kind":"row","values":{"sale_id":30,"sold_on":"2026-06-10"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## simple-06

- split: `test`
- category: `simple_filter_projection`
- difficulty: `medium`
- question: 华东地区在 2026 年 1 月以来有哪些销售编号和客户编号？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, s.customer_id FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东' AND s.sold_on >= DATE '2026-01-01' ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","customer_id"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1,1],[4,3],[7,6],[12,1],[17,8],[23,5],[29,3]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"sale_id":1}},{"kind":"row","values":{"customer_id":3,"sale_id":4}},{"kind":"row","values":{"customer_id":6,"sale_id":7}},{"kind":"row","values":{"customer_id":1,"sale_id":12}},{"kind":"row","values":{"customer_id":8,"sale_id":17}},{"kind":"row","values":{"customer_id":5,"sale_id":23}},{"kind":"row","values":{"customer_id":3,"sale_id":29}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## simple-07

- split: `test`
- category: `simple_filter_projection`
- difficulty: `easy`
- question: 西部市场包含哪些地区？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT region_name FROM chatbi_demo.regions WHERE market = '西部市场' ORDER BY region_code`
- expected result: `{"columns":["region_name"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["西南"],["西北"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"region_name":"西南"}},{"kind":"scalar","values":{"region_name":"西北"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-01

- split: `dev`
- category: `aggregation`
- difficulty: `easy`
- question: 全部销售合计金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT SUM(amount) AS total_amount FROM chatbi_demo.sales`
- expected result: `{"columns":["total_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[37626.24]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"total_amount":37626.24}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-02

- split: `dev`
- category: `aggregation`
- difficulty: `easy`
- question: 当前共有多少笔销售记录？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT COUNT(*) AS sale_count FROM chatbi_demo.sales`
- expected result: `{"columns":["sale_count"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[30]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"sale_count":30}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-03

- split: `dev`
- category: `aggregation`
- difficulty: `easy`
- question: 平均每笔销售金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT AVG(amount) AS average_amount FROM chatbi_demo.sales`
- expected result: `{"columns":["average_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1254.2079999999999]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"average_amount":1254.2079999999999}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-04

- split: `dev`
- category: `aggregation`
- difficulty: `easy`
- question: 所有销售中金额最大的单笔是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT MAX(amount) AS maximum_amount FROM chatbi_demo.sales`
- expected result: `{"columns":["maximum_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[3100]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"maximum_amount":3100}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-05

- split: `dev`
- category: `aggregation`
- difficulty: `easy`
- question: 当前最低的一笔销售金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT MIN(amount) AS minimum_amount FROM chatbi_demo.sales`
- expected result: `{"columns":["minimum_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[430]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"minimum_amount":430}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-06

- split: `dev`
- category: `aggregation`
- difficulty: `easy`
- question: 客户表中有多少位客户？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT COUNT(*) AS customer_count FROM chatbi_demo.customers`
- expected result: `{"columns":["customer_count"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[10]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"customer_count":10}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-07

- split: `test`
- category: `aggregation`
- difficulty: `medium`
- question: 华东地区合计销售金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT SUM(s.amount) AS east_total FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东'`
- expected result: `{"columns":["east_total"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[9459.99]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"east_total":9459.99}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-08

- split: `test`
- category: `aggregation`
- difficulty: `medium`
- question: 华东地区发生了几笔销售？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT COUNT(*) AS east_sale_count FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东'`
- expected result: `{"columns":["east_sale_count"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[7]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"east_sale_count":7}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## aggregation-09

- split: `test`
- category: `aggregation`
- difficulty: `medium`
- question: 2026 年 2 月 1 日及以后，华南地区的销售总额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT SUM(s.amount) AS south_total FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华南' AND s.sold_on >= DATE '2026-02-01'`
- expected result: `{"columns":["south_total"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[6220.5]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"south_total":6220.5}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-01

- split: `dev`
- category: `group_by`
- difficulty: `medium`
- question: 按地区汇总销售笔数和金额。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name AS region, COUNT(s.sale_id) AS sale_count, SUM(s.amount) AS total_amount FROM chatbi_demo.regions AS r LEFT JOIN chatbi_demo.sales AS s ON s.region_code = r.region_code GROUP BY r.region_code, r.region_name ORDER BY r.region_code`
- expected result: `{"columns":["region","sale_count","total_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["华东",7,9459.99],["华南",8,9581.0],["华北",5,7560.25],["西南",5,7300],["西北",5,3725]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"region":"华东","sale_count":7,"total_amount":9459.99}},{"kind":"row","values":{"region":"华南","sale_count":8,"total_amount":9581.0}},{"kind":"row","values":{"region":"华北","sale_count":5,"total_amount":7560.25}},{"kind":"row","values":{"region":"西南","sale_count":5,"total_amount":7300}},{"kind":"row","values":{"region":"西北","sale_count":5,"total_amount":3725}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-02

- split: `dev`
- category: `group_by`
- difficulty: `medium`
- question: 按客户编号统计销售笔数，并按客户编号排列。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT customer_id, COUNT(*) AS sale_count FROM chatbi_demo.sales GROUP BY customer_id ORDER BY customer_id`
- expected result: `{"columns":["customer_id","sale_count"],"omitted_rows":1,"row_order_sensitive":true,"rows":[[1,5],[2,3],[3,4],[4,3],[5,3],[6,4],[7,3],[8,3]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"sale_count":5}},{"kind":"row","values":{"customer_id":2,"sale_count":3}},{"kind":"row","values":{"customer_id":3,"sale_count":4}},{"kind":"row","values":{"customer_id":4,"sale_count":3}},{"kind":"row","values":{"customer_id":5,"sale_count":3}},{"kind":"row","values":{"customer_id":6,"sale_count":4}},{"kind":"row","values":{"customer_id":7,"sale_count":3}},{"kind":"row","values":{"customer_id":8,"sale_count":3}},{"kind":"row","values":{"customer_id":9,"sale_count":2}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-03

- split: `dev`
- category: `group_by`
- difficulty: `medium`
- question: 按客户编号汇总销售金额。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT customer_id, SUM(amount) AS total_amount FROM chatbi_demo.sales GROUP BY customer_id ORDER BY customer_id`
- expected result: `{"columns":["customer_id","total_amount"],"omitted_rows":1,"row_order_sensitive":false,"rows":[[1,8165.25],[2,2401.0],[3,4809.99],[4,3660],[5,3210],[6,4360],[7,5430],[8,2950]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"total_amount":8165.25}},{"kind":"row","values":{"customer_id":2,"total_amount":2401.0}},{"kind":"row","values":{"customer_id":3,"total_amount":4809.99}},{"kind":"row","values":{"customer_id":4,"total_amount":3660}},{"kind":"row","values":{"customer_id":5,"total_amount":3210}},{"kind":"row","values":{"customer_id":6,"total_amount":4360}},{"kind":"row","values":{"customer_id":7,"total_amount":5430}},{"kind":"row","values":{"customer_id":8,"total_amount":2950}},{"kind":"row","values":{"customer_id":9,"total_amount":2640}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-04

- split: `dev`
- category: `group_by`
- difficulty: `medium`
- question: 比较各地区的最高销售金额。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name AS region, MAX(s.amount) AS maximum_amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name ORDER BY r.region_code`
- expected result: `{"columns":["region","maximum_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["华东",3100],["华南",2500],["华北",1750],["西南",2500],["西北",980]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"maximum_amount":3100,"region":"华东"}},{"kind":"row","values":{"maximum_amount":2500,"region":"华南"}},{"kind":"row","values":{"maximum_amount":1750,"region":"华北"}},{"kind":"row","values":{"maximum_amount":2500,"region":"西南"}},{"kind":"row","values":{"maximum_amount":980,"region":"西北"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-05

- split: `dev`
- category: `group_by`
- difficulty: `medium`
- question: 各地区平均每笔销售金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name AS region, AVG(s.amount) AS average_amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name ORDER BY r.region_code`
- expected result: `{"columns":["region","average_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["华东",1351.4271428571428],["华南",1197.625],["华北",1512.05],["西南",1460.0],["西北",745.0]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"average_amount":1351.4271428571428,"region":"华东"}},{"kind":"row","values":{"average_amount":1197.625,"region":"华南"}},{"kind":"row","values":{"average_amount":1512.05,"region":"华北"}},{"kind":"row","values":{"average_amount":1460.0,"region":"西南"}},{"kind":"row","values":{"average_amount":745.0,"region":"西北"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-06

- split: `test`
- category: `group_by`
- difficulty: `medium`
- question: 按客户编号和名称统计每位客户的销售笔数，包含没有销售记录的客户。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, COUNT(s.sale_id) AS sale_count FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name","sale_count"],"omitted_rows":2,"row_order_sensitive":false,"rows":[[1,"北辰制造",5],[2,"星河贸易",3],[3,"远山科技",4],[4,"江南医药",3],[5,"海岳能源",3],[6,"北辰制造",4],[7,"晨光物流",3],[8,"云杉教育",3]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","sale_count":5}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","sale_count":3}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","sale_count":4}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","sale_count":3}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","sale_count":3}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","sale_count":4}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流","sale_count":3}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育","sale_count":3}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品","sale_count":2}},{"kind":"row","values":{"customer_id":10,"customer_name":"未成交客户","sale_count":0}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-07

- split: `test`
- category: `group_by`
- difficulty: `medium`
- question: 各地区最小的一笔销售金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name AS region, MIN(s.amount) AS minimum_amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name ORDER BY r.region_code`
- expected result: `{"columns":["region","minimum_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["华东",540],["华南",640],["华北",1200],["西南",660],["西北",430]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"minimum_amount":540,"region":"华东"}},{"kind":"row","values":{"minimum_amount":640,"region":"华南"}},{"kind":"row","values":{"minimum_amount":1200,"region":"华北"}},{"kind":"row","values":{"minimum_amount":660,"region":"西南"}},{"kind":"row","values":{"minimum_amount":430,"region":"西北"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## group-by-08

- split: `test`
- category: `group_by`
- difficulty: `medium`
- question: 各地区分别有多少笔销售？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name AS region, COUNT(*) AS sale_count FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name ORDER BY sale_count DESC, r.region_code`
- expected result: `{"columns":["region","sale_count"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["华南",8],["华东",7],["华北",5],["西南",5],["西北",5]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"region":"华南","sale_count":8}},{"kind":"row","values":{"region":"华东","sale_count":7}},{"kind":"row","values":{"region":"华北","sale_count":5}},{"kind":"row","values":{"region":"西南","sale_count":5}},{"kind":"row","values":{"region":"西北","sale_count":5}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## top-k-01

- split: `dev`
- category: `ordering_top_k`
- difficulty: `medium`
- question: 按金额从高到低取前两笔销售；金额相同时销售编号较小的优先。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, customer_id, amount FROM chatbi_demo.sales ORDER BY amount DESC, sale_id ASC LIMIT 2`
- expected result: `{"columns":["sale_id","customer_id","amount"],"omitted_rows":0,"row_order_sensitive":true,"rows":[[12,1,3100],[16,7,2500]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":3100,"customer_id":1,"sale_id":12}},{"kind":"row","values":{"amount":2500,"customer_id":7,"sale_id":16}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## top-k-02

- split: `dev`
- category: `ordering_top_k`
- difficulty: `medium`
- question: 最早发生的销售是哪一笔？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales ORDER BY sold_on ASC, sale_id LIMIT 1`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":0,"row_order_sensitive":true,"rows":[[1,"2026-01-01",1200]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"sale_id":1,"sold_on":"2026-01-01"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## top-k-03

- split: `dev`
- category: `ordering_top_k`
- difficulty: `medium`
- question: 最近发生的销售是哪一笔？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales ORDER BY sold_on DESC, sale_id DESC LIMIT 1`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":0,"row_order_sensitive":true,"rows":[[24,"2026-06-30",760]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":760,"sale_id":24,"sold_on":"2026-06-30"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## top-k-04

- split: `dev`
- category: `ordering_top_k`
- difficulty: `medium`
- question: 金额最低的销售是哪一笔？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, amount FROM chatbi_demo.sales ORDER BY amount ASC, sale_id LIMIT 1`
- expected result: `{"columns":["sale_id","amount"],"omitted_rows":0,"row_order_sensitive":true,"rows":[[8,430]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":430,"sale_id":8}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## top-k-05

- split: `test`
- category: `ordering_top_k`
- difficulty: `medium`
- question: 请按客户编号从大到小列出客户名称。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT customer_id, customer_name FROM chatbi_demo.customers ORDER BY customer_id DESC`
- expected result: `{"columns":["customer_id","customer_name"],"omitted_rows":2,"row_order_sensitive":true,"rows":[[10,"未成交客户"],[9,"南岭食品"],[8,"云杉教育"],[7,"晨光物流"],[6,"北辰制造"],[5,"海岳能源"],[4,"江南医药"],[3,"远山科技"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"customer_id":10,"customer_name":"未成交客户"}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品"}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育"}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流"}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造"}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源"}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药"}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技"}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易"}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## top-k-06

- split: `test`
- category: `ordering_top_k`
- difficulty: `medium`
- question: 请把所有销售按金额从低到高列出；金额相同时按销售编号从小到大排列。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, amount FROM chatbi_demo.sales ORDER BY amount ASC, sale_id ASC`
- expected result: `{"columns":["sale_id","amount"],"omitted_rows":22,"row_order_sensitive":true,"rows":[[8,430],[17,540],[13,640],[26,660],[19,680],[6,750],[24,760],[2,860.5]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":430,"sale_id":8}},{"kind":"row","values":{"amount":540,"sale_id":17}},{"kind":"row","values":{"amount":640,"sale_id":13}},{"kind":"row","values":{"amount":660,"sale_id":26}},{"kind":"row","values":{"amount":680,"sale_id":19}},{"kind":"row","values":{"amount":750,"sale_id":6}},{"kind":"row","values":{"amount":760,"sale_id":24}},{"kind":"row","values":{"amount":860.5,"sale_id":2}},{"kind":"row","values":{"amount":860.5,"sale_id":10}},{"kind":"row","values":{"amount":875,"sale_id":28}},{"kind":"row","values":{"amount":920,"sale_id":22}},{"kind":"row","values":{"amount":940,"sale_id":29}},{"kind":"row","values":{"amount":980,"sale_id":14}},{"kind":"row","values":{"amount":999.99,"sale_id":4}},{"kind":"row","values":{"amount":1120,"sale_id":11}},{"kind":"row","values":{"amount":1200,"sale_id":1}},{"kind":"row","values":{"amount":1200,"sale_id":7}},{"kind":"row","values":{"amount":1200,"sale_id":15}},{"kind":"row","values":{"amount":1200,"sale_id":30}},{"...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## top-k-07

- split: `test`
- category: `ordering_top_k`
- difficulty: `medium`
- question: 华东地区金额最高的销售是哪一笔？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东' ORDER BY s.amount DESC, s.sale_id LIMIT 1`
- expected result: `{"columns":["sale_id","amount"],"omitted_rows":0,"row_order_sensitive":true,"rows":[[12,3100]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":3100,"sale_id":12}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## predicate-01

- split: `dev`
- category: `multi_predicate`
- difficulty: `medium`
- question: 华东地区中金额超过 1300 的销售有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东' AND s.amount > 1300 ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[12,3100],[23,1480]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":3100,"sale_id":12}},{"kind":"row","values":{"amount":1480,"sale_id":23}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## predicate-02

- split: `dev`
- category: `multi_predicate`
- difficulty: `medium`
- question: 2026 年 1 月 12 日起且金额低于 1000 的销售有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales WHERE sold_on >= DATE '2026-01-12' AND amount < 1000 ORDER BY sold_on, sale_id`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":5,"row_order_sensitive":false,"rows":[[4,"2026-01-31",999.99],[6,"2026-02-02",750],[8,"2026-02-15",430],[26,"2026-02-20",660],[10,"2026-03-01",860.5],[13,"2026-03-31",640],[14,"2026-04-01",980],[28,"2026-04-15",875]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":999.99,"sale_id":4,"sold_on":"2026-01-31"}},{"kind":"row","values":{"amount":750,"sale_id":6,"sold_on":"2026-02-02"}},{"kind":"row","values":{"amount":430,"sale_id":8,"sold_on":"2026-02-15"}},{"kind":"row","values":{"amount":660,"sale_id":26,"sold_on":"2026-02-20"}},{"kind":"row","values":{"amount":860.5,"sale_id":10,"sold_on":"2026-03-01"}},{"kind":"row","values":{"amount":640,"sale_id":13,"sold_on":"2026-03-31"}},{"kind":"row","values":{"amount":980,"sale_id":14,"sold_on":"2026-04-01"}},{"kind":"row","values":{"amount":875,"sale_id":28,"sold_on":"2026-04-15"}},{"kind":"row","values":{"amount":540,"sale_id":17,"sold_on":"2026-04-30"}},{"kind":"row","values":{"amount":680,"sale_id":19,"sold_on":"2026-05-12"}},{"kind":"row","values":{"amount":940,"sale_id":29,"sold_on":"2026-05-15"}},{"kind":"row","values":{"amount":920,"sale_id":22,"sold_on":"2026-06-05"}},{"kind":"row","values":{"amount":760,"sale_id":24,"sold_on":"2026-06-30"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## predicate-03

- split: `dev`
- category: `multi_predicate`
- difficulty: `medium`
- question: 华东地区中编号小于 3 的销售有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东' AND s.sale_id < 3 ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1,1200]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"sale_id":1}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## predicate-04

- split: `dev`
- category: `multi_predicate`
- difficulty: `medium`
- question: 华东地区金额在 1000 到 1300 之间的销售有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东' AND s.amount BETWEEN 1000 AND 1300 ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1,1200],[7,1200]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"sale_id":1}},{"kind":"row","values":{"amount":1200,"sale_id":7}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## predicate-05

- split: `test`
- category: `multi_predicate`
- difficulty: `medium`
- question: 客户编号为 1 且早于 2026 年 3 月 1 日的销售有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales WHERE customer_id = 1 AND sold_on < DATE '2026-03-01' ORDER BY sold_on, sale_id`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1,"2026-01-01",1200],[3,"2026-01-12",1540.25]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"sale_id":1,"sold_on":"2026-01-01"}},{"kind":"row","values":{"amount":1540.25,"sale_id":3,"sold_on":"2026-01-12"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## predicate-06

- split: `test`
- category: `multi_predicate`
- difficulty: `hard`
- question: 华东或华南地区中金额至少为 860.50 的销售有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, r.region_name AS region, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name IN ('华东', '华南') AND s.amount >= 860.50 ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","region","amount"],"omitted_rows":4,"row_order_sensitive":false,"rows":[[1,"华东",1200],[2,"华南",860.5],[4,"华东",999.99],[7,"华东",1200],[10,"华南",860.5],[12,"华东",3100],[18,"华南",1320],[21,"华南",1450]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"region":"华东","sale_id":1}},{"kind":"row","values":{"amount":860.5,"region":"华南","sale_id":2}},{"kind":"row","values":{"amount":999.99,"region":"华东","sale_id":4}},{"kind":"row","values":{"amount":1200,"region":"华东","sale_id":7}},{"kind":"row","values":{"amount":860.5,"region":"华南","sale_id":10}},{"kind":"row","values":{"amount":3100,"region":"华东","sale_id":12}},{"kind":"row","values":{"amount":1320,"region":"华南","sale_id":18}},{"kind":"row","values":{"amount":1450,"region":"华南","sale_id":21}},{"kind":"row","values":{"amount":1480,"region":"华东","sale_id":23}},{"kind":"row","values":{"amount":2500,"region":"华南","sale_id":25}},{"kind":"row","values":{"amount":940,"region":"华东","sale_id":29}},{"kind":"row","values":{"amount":1200,"region":"华南","sale_id":30}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## predicate-07

- split: `test`
- category: `multi_predicate`
- difficulty: `hard`
- question: 非华东地区且金额超过 800 的销售有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, r.region_name AS region, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name <> '华东' AND s.amount > 800 ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","region","amount"],"omitted_rows":9,"row_order_sensitive":false,"rows":[[2,"华南",860.5],[3,"华北",1540.25],[5,"西南",2100],[9,"华北",1750],[10,"华南",860.5],[11,"西南",1120],[14,"西北",980],[15,"华北",1200]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":860.5,"region":"华南","sale_id":2}},{"kind":"row","values":{"amount":1540.25,"region":"华北","sale_id":3}},{"kind":"row","values":{"amount":2100,"region":"西南","sale_id":5}},{"kind":"row","values":{"amount":1750,"region":"华北","sale_id":9}},{"kind":"row","values":{"amount":860.5,"region":"华南","sale_id":10}},{"kind":"row","values":{"amount":1120,"region":"西南","sale_id":11}},{"kind":"row","values":{"amount":980,"region":"西北","sale_id":14}},{"kind":"row","values":{"amount":1200,"region":"华北","sale_id":15}},{"kind":"row","values":{"amount":2500,"region":"西南","sale_id":16}},{"kind":"row","values":{"amount":1320,"region":"华南","sale_id":18}},{"kind":"row","values":{"amount":1750,"region":"华北","sale_id":20}},{"kind":"row","values":{"amount":1450,"region":"华南","sale_id":21}},{"kind":"row","values":{"amount":920,"region":"西南","sale_id":22}},{"kind":"row","values":{"amount":2500,"region":"华南","sale_id":25}},{"kind":"row","values":{"amount":1320,"region":"华北","sale_i...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-01

- split: `dev`
- category: `join`
- difficulty: `medium`
- question: 列出每笔销售对应的客户名称和金额。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, c.customer_name, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","customer_name","amount"],"omitted_rows":22,"row_order_sensitive":false,"rows":[[1,"北辰制造",1200],[2,"星河贸易",860.5],[3,"北辰制造",1540.25],[4,"远山科技",999.99],[5,"江南医药",2100],[6,"海岳能源",750],[7,"北辰制造",1200],[8,"晨光物流",430]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"customer_name":"北辰制造","sale_id":1}},{"kind":"row","values":{"amount":860.5,"customer_name":"星河贸易","sale_id":2}},{"kind":"row","values":{"amount":1540.25,"customer_name":"北辰制造","sale_id":3}},{"kind":"row","values":{"amount":999.99,"customer_name":"远山科技","sale_id":4}},{"kind":"row","values":{"amount":2100,"customer_name":"江南医药","sale_id":5}},{"kind":"row","values":{"amount":750,"customer_name":"海岳能源","sale_id":6}},{"kind":"row","values":{"amount":1200,"customer_name":"北辰制造","sale_id":7}},{"kind":"row","values":{"amount":430,"customer_name":"晨光物流","sale_id":8}},{"kind":"row","values":{"amount":1750,"customer_name":"云杉教育","sale_id":9}},{"kind":"row","values":{"amount":860.5,"customer_name":"星河贸易","sale_id":10}},{"kind":"row","values":{"amount":1120,"customer_name":"远山科技","sale_id":11}},{"kind":"row","values":{"amount":3100,"customer_name":"北辰制造","sale_id":12}},{"kind":"row","values":{"amount":640,"customer_name":"江南医药","sale_id":13}},{"kind":"row"...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-02

- split: `dev`
- category: `join`
- difficulty: `medium`
- question: 按客户编号和名称查看他们对应的销售编号。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, s.sale_id FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id ORDER BY c.customer_id, s.sale_id`
- expected result: `{"columns":["customer_id","customer_name","sale_id"],"omitted_rows":22,"row_order_sensitive":false,"rows":[[1,"北辰制造",1],[1,"北辰制造",3],[1,"北辰制造",12],[1,"北辰制造",21],[1,"北辰制造",28],[2,"星河贸易",2],[2,"星河贸易",10],[2,"星河贸易",19]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","sale_id":1}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","sale_id":3}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","sale_id":12}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","sale_id":21}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","sale_id":28}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","sale_id":2}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","sale_id":10}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","sale_id":19}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","sale_id":4}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","sale_id":11}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","sale_id":20}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","sale_id":29}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-03

- split: `dev`
- category: `join`
- difficulty: `medium`
- question: 只看华东销售时，对应的客户名称和金额是什么？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, c.customer_name, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE r.region_name = '华东' ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","customer_name","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1,"北辰制造",1200],[4,"远山科技",999.99],[7,"北辰制造",1200],[12,"北辰制造",3100],[17,"云杉教育",540],[23,"海岳能源",1480],[29,"远山科技",940]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"customer_name":"北辰制造","sale_id":1}},{"kind":"row","values":{"amount":999.99,"customer_name":"远山科技","sale_id":4}},{"kind":"row","values":{"amount":1200,"customer_name":"北辰制造","sale_id":7}},{"kind":"row","values":{"amount":3100,"customer_name":"北辰制造","sale_id":12}},{"kind":"row","values":{"amount":540,"customer_name":"云杉教育","sale_id":17}},{"kind":"row","values":{"amount":1480,"customer_name":"海岳能源","sale_id":23}},{"kind":"row","values":{"amount":940,"customer_name":"远山科技","sale_id":29}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-04

- split: `dev`
- category: `join`
- difficulty: `medium`
- question: 每笔销售对应客户的名称和日期是什么？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, c.customer_name, s.sold_on FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","customer_name","sold_on"],"omitted_rows":22,"row_order_sensitive":false,"rows":[[1,"北辰制造","2026-01-01"],[2,"星河贸易","2026-01-05"],[3,"北辰制造","2026-01-12"],[4,"远山科技","2026-01-31"],[5,"江南医药","2026-02-01"],[6,"海岳能源","2026-02-02"],[7,"北辰制造","2026-02-10"],[8,"晨光物流","2026-02-15"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_name":"北辰制造","sale_id":1,"sold_on":"2026-01-01"}},{"kind":"row","values":{"customer_name":"星河贸易","sale_id":2,"sold_on":"2026-01-05"}},{"kind":"row","values":{"customer_name":"北辰制造","sale_id":3,"sold_on":"2026-01-12"}},{"kind":"row","values":{"customer_name":"远山科技","sale_id":4,"sold_on":"2026-01-31"}},{"kind":"row","values":{"customer_name":"江南医药","sale_id":5,"sold_on":"2026-02-01"}},{"kind":"row","values":{"customer_name":"海岳能源","sale_id":6,"sold_on":"2026-02-02"}},{"kind":"row","values":{"customer_name":"北辰制造","sale_id":7,"sold_on":"2026-02-10"}},{"kind":"row","values":{"customer_name":"晨光物流","sale_id":8,"sold_on":"2026-02-15"}},{"kind":"row","values":{"customer_name":"云杉教育","sale_id":9,"sold_on":"2026-02-28"}},{"kind":"row","values":{"customer_name":"星河贸易","sale_id":10,"sold_on":"2026-03-01"}},{"kind":"row","values":{"customer_name":"远山科技","sale_id":11,"sold_on":"2026-03-03"}},{"kind":"row","values":{"customer_name":"北辰制造","sale_id":12,"sold_on":...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-05

- split: `dev`
- category: `join`
- difficulty: `hard`
- question: 每位客户（含没有销售的客户）最早的一笔销售日期是什么？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, MIN(s.sold_on) AS first_sale_on FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name","first_sale_on"],"omitted_rows":2,"row_order_sensitive":false,"rows":[[1,"北辰制造","2026-01-01"],[2,"星河贸易","2026-01-05"],[3,"远山科技","2026-01-31"],[4,"江南医药","2026-02-01"],[5,"海岳能源","2026-02-02"],[6,"北辰制造","2026-02-10"],[7,"晨光物流","2026-01-20"],[8,"云杉教育","2026-02-20"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","first_sale_on":"2026-01-01"}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","first_sale_on":"2026-01-05"}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","first_sale_on":"2026-01-31"}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","first_sale_on":"2026-02-01"}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","first_sale_on":"2026-02-02"}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","first_sale_on":"2026-02-10"}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流","first_sale_on":"2026-01-20"}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育","first_sale_on":"2026-02-20"}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品","first_sale_on":"2026-03-20"}},{"kind":"null","values":{"customer_id":10,"customer_name":"未成交客户","first_sale_on":null}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-06

- split: `dev`
- category: `join`
- difficulty: `medium`
- question: 金额超过 1000 的销售分别属于哪位客户？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, c.customer_name, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id WHERE s.amount > 1000 ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","customer_name","amount"],"omitted_rows":8,"row_order_sensitive":false,"rows":[[1,"北辰制造",1200],[3,"北辰制造",1540.25],[5,"江南医药",2100],[7,"北辰制造",1200],[9,"云杉教育",1750],[11,"远山科技",1120],[12,"北辰制造",3100],[15,"北辰制造",1200]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"customer_name":"北辰制造","sale_id":1}},{"kind":"row","values":{"amount":1540.25,"customer_name":"北辰制造","sale_id":3}},{"kind":"row","values":{"amount":2100,"customer_name":"江南医药","sale_id":5}},{"kind":"row","values":{"amount":1200,"customer_name":"北辰制造","sale_id":7}},{"kind":"row","values":{"amount":1750,"customer_name":"云杉教育","sale_id":9}},{"kind":"row","values":{"amount":1120,"customer_name":"远山科技","sale_id":11}},{"kind":"row","values":{"amount":3100,"customer_name":"北辰制造","sale_id":12}},{"kind":"row","values":{"amount":1200,"customer_name":"北辰制造","sale_id":15}},{"kind":"row","values":{"amount":2500,"customer_name":"晨光物流","sale_id":16}},{"kind":"row","values":{"amount":1320,"customer_name":"南岭食品","sale_id":18}},{"kind":"row","values":{"amount":1750,"customer_name":"远山科技","sale_id":20}},{"kind":"row","values":{"amount":1450,"customer_name":"北辰制造","sale_id":21}},{"kind":"row","values":{"amount":1480,"customer_name":"海岳能源","sale_id":23}},{"kind":"r...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-07

- split: `test`
- category: `join`
- difficulty: `medium`
- question: 哪些客户至少有一笔销售？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT DISTINCT c.customer_id, c.customer_name FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name"],"omitted_rows":1,"row_order_sensitive":false,"rows":[[1,"北辰制造"],[2,"星河贸易"],[3,"远山科技"],[4,"江南医药"],[5,"海岳能源"],[6,"北辰制造"],[7,"晨光物流"],[8,"云杉教育"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造"}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易"}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技"}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药"}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源"}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造"}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流"}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育"}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-08

- split: `test`
- category: `join`
- difficulty: `medium`
- question: 按客户编号查看每笔销售所在地区和金额。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, r.region_name AS region, s.amount FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code ORDER BY c.customer_id, s.sale_id`
- expected result: `{"columns":["customer_id","customer_name","region","amount"],"omitted_rows":22,"row_order_sensitive":false,"rows":[[1,"北辰制造","华东",1200],[1,"北辰制造","华北",1540.25],[1,"北辰制造","华东",3100],[1,"北辰制造","华南",1450],[1,"北辰制造","西北",875],[2,"星河贸易","华南",860.5],[2,"星河贸易","华南",860.5],[2,"星河贸易","西北",680]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"customer_id":1,"customer_name":"北辰制造","region":"华东"}},{"kind":"row","values":{"amount":1540.25,"customer_id":1,"customer_name":"北辰制造","region":"华北"}},{"kind":"row","values":{"amount":3100,"customer_id":1,"customer_name":"北辰制造","region":"华东"}},{"kind":"row","values":{"amount":1450,"customer_id":1,"customer_name":"北辰制造","region":"华南"}},{"kind":"row","values":{"amount":875,"customer_id":1,"customer_name":"北辰制造","region":"西北"}},{"kind":"row","values":{"amount":860.5,"customer_id":2,"customer_name":"星河贸易","region":"华南"}},{"kind":"row","values":{"amount":860.5,"customer_id":2,"customer_name":"星河贸易","region":"华南"}},{"kind":"row","values":{"amount":680,"customer_id":2,"customer_name":"星河贸易","region":"西北"}},{"kind":"row","values":{"amount":999.99,"customer_id":3,"customer_name":"远山科技","region":"华东"}},{"kind":"row","values":{"amount":1120,"customer_id":3,"customer_name":"远山科技","region":"西南"}},{"kind":"row","values":{"amount":1750,"customer_id":3,"custom...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-09

- split: `test`
- category: `join`
- difficulty: `medium`
- question: 2026 年 2 月发生的销售对应哪位客户？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, c.customer_name, s.sold_on FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id WHERE s.sold_on >= DATE '2026-02-01' AND s.sold_on < DATE '2026-03-01' ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","customer_name","sold_on"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[5,"江南医药","2026-02-01"],[6,"海岳能源","2026-02-02"],[7,"北辰制造","2026-02-10"],[8,"晨光物流","2026-02-15"],[9,"云杉教育","2026-02-28"],[26,"云杉教育","2026-02-20"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_name":"江南医药","sale_id":5,"sold_on":"2026-02-01"}},{"kind":"row","values":{"customer_name":"海岳能源","sale_id":6,"sold_on":"2026-02-02"}},{"kind":"row","values":{"customer_name":"北辰制造","sale_id":7,"sold_on":"2026-02-10"}},{"kind":"row","values":{"customer_name":"晨光物流","sale_id":8,"sold_on":"2026-02-15"}},{"kind":"row","values":{"customer_name":"云杉教育","sale_id":9,"sold_on":"2026-02-28"}},{"kind":"row","values":{"customer_name":"云杉教育","sale_id":26,"sold_on":"2026-02-20"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## join-10

- split: `test`
- category: `join`
- difficulty: `hard`
- question: 每位客户涉及过多少个不同销售地区？包含没有销售记录的客户。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, COUNT(DISTINCT s.region_code) AS region_count FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name","region_count"],"omitted_rows":2,"row_order_sensitive":false,"rows":[[1,"北辰制造",4],[2,"星河贸易",2],[3,"远山科技",3],[4,"江南医药",2],[5,"海岳能源",3],[6,"北辰制造",4],[7,"晨光物流",3],[8,"云杉教育",3]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","region_count":4}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","region_count":2}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","region_count":3}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","region_count":2}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","region_count":3}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","region_count":4}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流","region_count":3}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育","region_count":3}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品","region_count":2}},{"kind":"row","values":{"customer_id":10,"customer_name":"未成交客户","region_count":0}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## multi-join-01

- split: `dev`
- category: `multi_table_join`
- difficulty: `hard`
- question: 按客户和地区查看销售笔数及总额。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, r.region_name AS region, COUNT(s.sale_id) AS sale_count, SUM(s.amount) AS total_amount FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY c.customer_id, c.customer_name, r.region_code, r.region_name ORDER BY c.customer_id, r.region_code`
- expected result: `{"columns":["customer_id","customer_name","region","sale_count","total_amount"],"omitted_rows":18,"row_order_sensitive":false,"rows":[[1,"北辰制造","华东",2,4300],[1,"北辰制造","华南",1,1450],[1,"北辰制造","华北",1,1540.25],[1,"北辰制造","西北",1,875],[2,"星河贸易","华南",2,1721.0],[2,"星河贸易","西北",1,680],[3,"远山科技","华东",2,1939.99],[3,"远山科技","华北",1,1750]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","region":"华东","sale_count":2,"total_amount":4300}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","region":"华南","sale_count":1,"total_amount":1450}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","region":"华北","sale_count":1,"total_amount":1540.25}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","region":"西北","sale_count":1,"total_amount":875}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","region":"华南","sale_count":2,"total_amount":1721.0}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","region":"西北","sale_count":1,"total_amount":680}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","region":"华东","sale_count":2,"total_amount":1939.99}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","region":"华北","sale_count":1,"total_amount":1750}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","region":"西南","...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## multi-join-02

- split: `dev`
- category: `multi_table_join`
- difficulty: `hard`
- question: 按销售市场和地区汇总销售金额，并显示涉及的客户数。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.market, r.region_name AS region, COUNT(DISTINCT c.customer_id) AS customer_count, SUM(s.amount) AS total_amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY r.market, r.region_code, r.region_name ORDER BY r.market, r.region_code`
- expected result: `{"columns":["market","region","customer_count","total_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["东部市场","华东",5,9459.99],["北部市场","华北",5,7560.25],["南部市场","华南",7,9581.0],["西部市场","西南",4,7300],["西部市场","西北",5,3725]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_count":5,"market":"东部市场","region":"华东","total_amount":9459.99}},{"kind":"row","values":{"customer_count":5,"market":"北部市场","region":"华北","total_amount":7560.25}},{"kind":"row","values":{"customer_count":7,"market":"南部市场","region":"华南","total_amount":9581.0}},{"kind":"row","values":{"customer_count":4,"market":"西部市场","region":"西南","total_amount":7300}},{"kind":"row","values":{"customer_count":5,"market":"西部市场","region":"西北","total_amount":3725}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## multi-join-03

- split: `dev`
- category: `multi_table_join`
- difficulty: `hard`
- question: 列出每个地区有销售记录的客户及其销售笔数。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name AS region, c.customer_id, c.customer_name, COUNT(*) AS sale_count FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY r.region_code, r.region_name, c.customer_id, c.customer_name ORDER BY r.region_code, c.customer_id`
- expected result: `{"columns":["region","customer_id","customer_name","sale_count"],"omitted_rows":18,"row_order_sensitive":false,"rows":[["华东",1,"北辰制造",2],["华东",3,"远山科技",2],["华东",5,"海岳能源",1],["华东",6,"北辰制造",1],["华东",8,"云杉教育",1],["华南",1,"北辰制造",1],["华南",2,"星河贸易",2],["华南",4,"江南医药",1]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","region":"华东","sale_count":2}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","region":"华东","sale_count":2}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","region":"华东","sale_count":1}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","region":"华东","sale_count":1}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育","region":"华东","sale_count":1}},{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","region":"华南","sale_count":1}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","region":"华南","sale_count":2}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","region":"华南","sale_count":1}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","region":"华南","sale_count":1}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","region":"华南","sale_count":1}},{"kind":"row","values":{"customer_id":7,"customer_name":...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## multi-join-04

- split: `dev`
- category: `multi_table_join`
- difficulty: `hard`
- question: 找出金额最高的销售及其客户和地区。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT s.sale_id, c.customer_name, r.region_name AS region, s.amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code ORDER BY s.amount DESC, s.sale_id LIMIT 1`
- expected result: `{"columns":["sale_id","customer_name","region","amount"],"omitted_rows":0,"row_order_sensitive":true,"rows":[[12,"北辰制造","华东",3100]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"amount":3100,"customer_name":"北辰制造","region":"华东","sale_id":12}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## multi-join-05

- split: `test`
- category: `multi_table_join`
- difficulty: `hard`
- question: 按客户编号和地区统计销售笔数、金额，并显示所属市场。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, r.region_name AS region, r.market, COUNT(*) AS sale_count, SUM(s.amount) AS total_amount FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY c.customer_id, r.region_code, r.region_name, r.market ORDER BY c.customer_id, r.region_code`
- expected result: `{"columns":["customer_id","region","market","sale_count","total_amount"],"omitted_rows":18,"row_order_sensitive":false,"rows":[[1,"华东","东部市场",2,4300],[1,"华南","南部市场",1,1450],[1,"华北","北部市场",1,1540.25],[1,"西北","西部市场",1,875],[2,"华南","南部市场",2,1721.0],[2,"西北","西部市场",1,680],[3,"华东","东部市场",2,1939.99],[3,"华北","北部市场",1,1750]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"market":"东部市场","region":"华东","sale_count":2,"total_amount":4300}},{"kind":"row","values":{"customer_id":1,"market":"南部市场","region":"华南","sale_count":1,"total_amount":1450}},{"kind":"row","values":{"customer_id":1,"market":"北部市场","region":"华北","sale_count":1,"total_amount":1540.25}},{"kind":"row","values":{"customer_id":1,"market":"西部市场","region":"西北","sale_count":1,"total_amount":875}},{"kind":"row","values":{"customer_id":2,"market":"南部市场","region":"华南","sale_count":2,"total_amount":1721.0}},{"kind":"row","values":{"customer_id":2,"market":"西部市场","region":"西北","sale_count":1,"total_amount":680}},{"kind":"row","values":{"customer_id":3,"market":"东部市场","region":"华东","sale_count":2,"total_amount":1939.99}},{"kind":"row","values":{"customer_id":3,"market":"北部市场","region":"华北","sale_count":1,"total_amount":1750}},{"kind":"row","values":{"customer_id":3,"market":"西部市场","region":"西南","sale_count":1,"total_amount":1120}},{"kind":"row","values":{"cu...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## multi-join-06

- split: `test`
- category: `multi_table_join`
- difficulty: `hard`
- question: 哪个销售市场的累计销售额最高，以及该市场覆盖了多少位客户？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.market, SUM(s.amount) AS total_amount, COUNT(DISTINCT c.customer_id) AS customer_count FROM chatbi_demo.sales AS s JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code GROUP BY r.market ORDER BY total_amount DESC, r.market LIMIT 1`
- expected result: `{"columns":["market","total_amount","customer_count"],"omitted_rows":0,"row_order_sensitive":true,"rows":[["西部市场",11025,8]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `True`
- structured answer facts: `[{"kind":"row","values":{"customer_count":8,"market":"西部市场","total_amount":11025}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## date-01

- split: `dev`
- category: `date_filter`
- difficulty: `easy`
- question: 2026 年 1 月有多少笔销售？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT COUNT(*) AS sale_count FROM chatbi_demo.sales WHERE sold_on >= DATE '2026-01-01' AND sold_on < DATE '2026-02-01'`
- expected result: `{"columns":["sale_count"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[5]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"sale_count":5}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## date-02

- split: `dev`
- category: `date_filter`
- difficulty: `easy`
- question: 2026 年 2 月发生了哪些销售？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales WHERE sold_on >= DATE '2026-02-01' AND sold_on < DATE '2026-03-01' ORDER BY sale_id`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[5,"2026-02-01",2100],[6,"2026-02-02",750],[7,"2026-02-10",1200],[8,"2026-02-15",430],[9,"2026-02-28",1750],[26,"2026-02-20",660]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":2100,"sale_id":5,"sold_on":"2026-02-01"}},{"kind":"row","values":{"amount":750,"sale_id":6,"sold_on":"2026-02-02"}},{"kind":"row","values":{"amount":1200,"sale_id":7,"sold_on":"2026-02-10"}},{"kind":"row","values":{"amount":430,"sale_id":8,"sold_on":"2026-02-15"}},{"kind":"row","values":{"amount":1750,"sale_id":9,"sold_on":"2026-02-28"}},{"kind":"row","values":{"amount":660,"sale_id":26,"sold_on":"2026-02-20"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## date-03

- split: `dev`
- category: `date_filter`
- difficulty: `easy`
- question: 2026 年第一个月有哪些销售编号和日期？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on FROM chatbi_demo.sales WHERE sold_on >= DATE '2026-01-01' AND sold_on < DATE '2026-02-01' ORDER BY sale_id`
- expected result: `{"columns":["sale_id","sold_on"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1,"2026-01-01"],[2,"2026-01-05"],[3,"2026-01-12"],[4,"2026-01-31"],[25,"2026-01-20"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"sale_id":1,"sold_on":"2026-01-01"}},{"kind":"row","values":{"sale_id":2,"sold_on":"2026-01-05"}},{"kind":"row","values":{"sale_id":3,"sold_on":"2026-01-12"}},{"kind":"row","values":{"sale_id":4,"sold_on":"2026-01-31"}},{"kind":"row","values":{"sale_id":25,"sold_on":"2026-01-20"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## date-04

- split: `test`
- category: `date_filter`
- difficulty: `medium`
- question: 早于 2026 年 2 月 1 日（不含当日）的销售总金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT SUM(amount) AS total_amount FROM chatbi_demo.sales WHERE sold_on < DATE '2026-02-01'`
- expected result: `{"columns":["total_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[7100.74]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"total_amount":7100.74}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## date-05

- split: `test`
- category: `date_filter`
- difficulty: `medium`
- question: 2026 年 4 月 1 日至 2026 年 4 月 30 日（含首尾）有哪些销售？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales WHERE sold_on >= DATE '2026-04-01' AND sold_on <= DATE '2026-04-30' ORDER BY sold_on, sale_id`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[14,"2026-04-01",980],[15,"2026-04-10",1200],[28,"2026-04-15",875],[16,"2026-04-20",2500],[17,"2026-04-30",540]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":980,"sale_id":14,"sold_on":"2026-04-01"}},{"kind":"row","values":{"amount":1200,"sale_id":15,"sold_on":"2026-04-10"}},{"kind":"row","values":{"amount":875,"sale_id":28,"sold_on":"2026-04-15"}},{"kind":"row","values":{"amount":2500,"sale_id":16,"sold_on":"2026-04-20"}},{"kind":"row","values":{"amount":540,"sale_id":17,"sold_on":"2026-04-30"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## null-01

- split: `dev`
- category: `null_boundary`
- difficulty: `medium`
- question: 哪些客户从未产生销售记录？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id WHERE s.sale_id IS NULL ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[10,"未成交客户"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":10,"customer_name":"未成交客户"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## null-02

- split: `dev`
- category: `null_boundary`
- difficulty: `medium`
- question: 每位客户的销售笔数和总额是多少？没有销售的客户总额按 0 计。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, COUNT(s.sale_id) AS sale_count, COALESCE(SUM(s.amount), 0) AS total_amount FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name","sale_count","total_amount"],"omitted_rows":2,"row_order_sensitive":false,"rows":[[1,"北辰制造",5,8165.25],[2,"星河贸易",3,2401.0],[3,"远山科技",4,4809.99],[4,"江南医药",3,3660],[5,"海岳能源",3,3210],[6,"北辰制造",4,4360],[7,"晨光物流",3,5430],[8,"云杉教育",3,2950]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","sale_count":5,"total_amount":8165.25}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","sale_count":3,"total_amount":2401.0}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","sale_count":4,"total_amount":4809.99}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","sale_count":3,"total_amount":3660}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","sale_count":3,"total_amount":3210}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","sale_count":4,"total_amount":4360}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流","sale_count":3,"total_amount":5430}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育","sale_count":3,"total_amount":2950}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品","sale_count":2,"total_amount":2640}},{"kind":"row","values":{"customer_id":10,"customer_name":"未成交客户","sale_count":0,"total_a...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## null-03

- split: `dev`
- category: `null_boundary`
- difficulty: `medium`
- question: 列出客户名称及最近销售日期，并保留无销售客户的空日期。
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, MAX(s.sold_on) AS latest_sale_on FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name","latest_sale_on"],"omitted_rows":2,"row_order_sensitive":false,"rows":[[1,"北辰制造","2026-06-01"],[2,"星河贸易","2026-05-12"],[3,"远山科技","2026-05-31"],[4,"江南医药","2026-06-05"],[5,"海岳能源","2026-06-15"],[6,"北辰制造","2026-06-30"],[7,"晨光物流","2026-04-20"],[8,"云杉教育","2026-04-30"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","latest_sale_on":"2026-06-01"}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","latest_sale_on":"2026-05-12"}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","latest_sale_on":"2026-05-31"}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","latest_sale_on":"2026-06-05"}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","latest_sale_on":"2026-06-15"}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","latest_sale_on":"2026-06-30"}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流","latest_sale_on":"2026-04-20"}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育","latest_sale_on":"2026-04-30"}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品","latest_sale_on":"2026-05-01"}},{"kind":"null","values":{"customer_id":10,"customer_name":"未成交客户","latest_sale_on":null}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## null-04

- split: `test`
- category: `null_boundary`
- difficulty: `medium`
- question: 2026 年 6 月哪些客户没有销售记录？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name FROM chatbi_demo.customers AS c LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id AND s.sold_on >= DATE '2026-06-01' AND s.sold_on < DATE '2026-07-01' WHERE s.sale_id IS NULL ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[2,"星河贸易"],[3,"远山科技"],[7,"晨光物流"],[8,"云杉教育"],[9,"南岭食品"],[10,"未成交客户"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易"}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技"}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流"}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育"}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品"}},{"kind":"row","values":{"customer_id":10,"customer_name":"未成交客户"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## empty-01

- split: `dev`
- category: `empty`
- difficulty: `easy`
- question: 销售编号为 999 的记录是什么？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales WHERE sale_id = 999`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"empty","values":{}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## empty-02

- split: `dev`
- category: `empty`
- difficulty: `easy`
- question: 客户名称以“不存在”开头的客户有哪些？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT customer_id, customer_name FROM chatbi_demo.customers WHERE customer_name LIKE '不存在%'`
- expected result: `{"columns":["customer_id","customer_name"],"omitted_rows":0,"row_order_sensitive":false,"rows":[],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"empty","values":{}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## empty-03

- split: `test`
- category: `empty`
- difficulty: `easy`
- question: 2027 年 1 月 1 日及以后还有销售记录吗？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, sold_on, amount FROM chatbi_demo.sales WHERE sold_on >= DATE '2027-01-01'`
- expected result: `{"columns":["sale_id","sold_on","amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"empty","values":{}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## derived-01

- split: `dev`
- category: `alias_derived`
- difficulty: `medium`
- question: 每笔销售金额比全部销售的平均金额高或低多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT sale_id, amount, amount - (SELECT AVG(amount) FROM chatbi_demo.sales) AS delta FROM chatbi_demo.sales ORDER BY sale_id`
- expected result: `{"columns":["sale_id","amount","delta"],"omitted_rows":22,"row_order_sensitive":false,"rows":[[1,1200,-54.207999999999856],[2,860.5,-393.70799999999986],[3,1540.25,286.04200000000014],[4,999.99,-254.21799999999985],[5,2100,845.7920000000001],[6,750,-504.20799999999986],[7,1200,-54.207999999999856],[8,430,-824.2079999999999]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1200,"delta":-54.207999999999856,"sale_id":1}},{"kind":"row","values":{"amount":860.5,"delta":-393.70799999999986,"sale_id":2}},{"kind":"row","values":{"amount":1540.25,"delta":286.04200000000014,"sale_id":3}},{"kind":"row","values":{"amount":999.99,"delta":-254.21799999999985,"sale_id":4}},{"kind":"row","values":{"amount":2100,"delta":845.7920000000001,"sale_id":5}},{"kind":"row","values":{"amount":750,"delta":-504.20799999999986,"sale_id":6}},{"kind":"row","values":{"amount":1200,"delta":-54.207999999999856,"sale_id":7}},{"kind":"row","values":{"amount":430,"delta":-824.2079999999999,"sale_id":8}},{"kind":"row","values":{"amount":1750,"delta":495.79200000000014,"sale_id":9}},{"kind":"row","values":{"amount":860.5,"delta":-393.70799999999986,"sale_id":10}},{"kind":"row","values":{"amount":1120,"delta":-134.20799999999986,"sale_id":11}},{"kind":"row","values":{"amount":3100,"delta":1845.7920000000001,"sale_id":12}},{"kind":"row","values":{"amount":6...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## derived-02

- split: `dev`
- category: `alias_derived`
- difficulty: `medium`
- question: 各地区销售笔数占全部销售笔数的比例是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name, AVG(CAST(s.region_code = r.region_code AS INTEGER)) AS sales_share FROM chatbi_demo.regions AS r CROSS JOIN chatbi_demo.sales AS s GROUP BY r.region_code, r.region_name ORDER BY r.region_code`
- expected result: `{"columns":["region_name","sales_share"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["华东",0.23333333333333334],["华南",0.26666666666666666],["华北",0.16666666666666666],["西南",0.16666666666666666],["西北",0.16666666666666666]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"region_name":"华东","sales_share":0.23333333333333334}},{"kind":"row","values":{"region_name":"华南","sales_share":0.26666666666666666}},{"kind":"row","values":{"region_name":"华北","sales_share":0.16666666666666666}},{"kind":"row","values":{"region_name":"西南","sales_share":0.16666666666666666}},{"kind":"row","values":{"region_name":"西北","sales_share":0.16666666666666666}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## derived-03

- split: `dev`
- category: `alias_derived`
- difficulty: `medium`
- question: 只统计有销售记录的客户，每位客户的平均销售金额是多少？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT c.customer_id, c.customer_name, AVG(s.amount) AS average_amount FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id`
- expected result: `{"columns":["customer_id","customer_name","average_amount"],"omitted_rows":1,"row_order_sensitive":false,"rows":[[1,"北辰制造",1633.05],[2,"星河贸易",800.3333333333334],[3,"远山科技",1202.4975],[4,"江南医药",1220.0],[5,"海岳能源",1070.0],[6,"北辰制造",1090.0],[7,"晨光物流",1810.0],[8,"云杉教育",983.3333333333334]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"average_amount":1633.05,"customer_id":1,"customer_name":"北辰制造"}},{"kind":"row","values":{"average_amount":800.3333333333334,"customer_id":2,"customer_name":"星河贸易"}},{"kind":"row","values":{"average_amount":1202.4975,"customer_id":3,"customer_name":"远山科技"}},{"kind":"row","values":{"average_amount":1220.0,"customer_id":4,"customer_name":"江南医药"}},{"kind":"row","values":{"average_amount":1070.0,"customer_id":5,"customer_name":"海岳能源"}},{"kind":"row","values":{"average_amount":1090.0,"customer_id":6,"customer_name":"北辰制造"}},{"kind":"row","values":{"average_amount":1810.0,"customer_id":7,"customer_name":"晨光物流"}},{"kind":"row","values":{"average_amount":983.3333333333334,"customer_id":8,"customer_name":"云杉教育"}},{"kind":"row","values":{"average_amount":1320.0,"customer_id":9,"customer_name":"南岭食品"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## derived-04

- split: `test`
- category: `alias_derived`
- difficulty: `hard`
- question: 每位有销售记录的客户，其销售总额比这些客户的平均销售总额高或低多少？
- semantic: `positive` / `answerable`
- reference SQL: `WITH customer_totals AS (SELECT c.customer_id, c.customer_name, SUM(s.amount) AS total_amount FROM chatbi_demo.customers AS c JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name) SELECT customer_id, customer_name, total_amount, total_amount - (SELECT AVG(total_amount) FROM customer_totals) AS delta FROM customer_totals ORDER BY customer_id`
- expected result: `{"columns":["customer_id","customer_name","total_amount","delta"],"omitted_rows":1,"row_order_sensitive":false,"rows":[[1,"北辰制造",8165.25,3984.5566666666673],[2,"星河贸易",2401.0,-1779.6933333333327],[3,"远山科技",4809.99,629.2966666666671],[4,"江南医药",3660,-520.6933333333327],[5,"海岳能源",3210,-970.6933333333327],[6,"北辰制造",4360,179.3066666666673],[7,"晨光物流",5430,1249.3066666666673],[8,"云杉教育",2950,-1230.6933333333327]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"customer_name":"北辰制造","delta":3984.5566666666673,"total_amount":8165.25}},{"kind":"row","values":{"customer_id":2,"customer_name":"星河贸易","delta":-1779.6933333333327,"total_amount":2401.0}},{"kind":"row","values":{"customer_id":3,"customer_name":"远山科技","delta":629.2966666666671,"total_amount":4809.99}},{"kind":"row","values":{"customer_id":4,"customer_name":"江南医药","delta":-520.6933333333327,"total_amount":3660}},{"kind":"row","values":{"customer_id":5,"customer_name":"海岳能源","delta":-970.6933333333327,"total_amount":3210}},{"kind":"row","values":{"customer_id":6,"customer_name":"北辰制造","delta":179.3066666666673,"total_amount":4360}},{"kind":"row","values":{"customer_id":7,"customer_name":"晨光物流","delta":1249.3066666666673,"total_amount":5430}},{"kind":"row","values":{"customer_id":8,"customer_name":"云杉教育","delta":-1230.6933333333327,"total_amount":2950}},{"kind":"row","values":{"customer_id":9,"customer_name":"南岭食品","delta":-1540.6933333333327,"...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## cte-set-01

- split: `dev`
- category: `cte_set_operation`
- difficulty: `hard`
- question: 哪些客户的累计销售额高于有销售客户的平均累计销售额？
- semantic: `positive` / `answerable`
- reference SQL: `WITH customer_totals AS (SELECT customer_id, SUM(amount) AS total_amount FROM chatbi_demo.sales GROUP BY customer_id) SELECT customer_id, total_amount FROM customer_totals WHERE total_amount > (SELECT AVG(total_amount) FROM customer_totals) ORDER BY customer_id`
- expected result: `{"columns":["customer_id","total_amount"],"omitted_rows":0,"row_order_sensitive":false,"rows":[[1,8165.25],[3,4809.99],[6,4360],[7,5430]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"customer_id":1,"total_amount":8165.25}},{"kind":"row","values":{"customer_id":3,"total_amount":4809.99}},{"kind":"row","values":{"customer_id":6,"total_amount":4360}},{"kind":"row","values":{"customer_id":7,"total_amount":5430}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## cte-set-02

- split: `dev`
- category: `cte_set_operation`
- difficulty: `hard`
- question: 哪些销售金额高于其所在地区的平均销售金额？
- semantic: `positive` / `answerable`
- reference SQL: `WITH region_average AS (SELECT region_code, AVG(amount) AS average_amount FROM chatbi_demo.sales GROUP BY region_code) SELECT s.sale_id, r.region_name, s.amount FROM chatbi_demo.sales AS s JOIN region_average AS a ON a.region_code = s.region_code JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE s.amount > a.average_amount ORDER BY s.sale_id`
- expected result: `{"columns":["sale_id","region_name","amount"],"omitted_rows":6,"row_order_sensitive":false,"rows":[[3,"华北",1540.25],[5,"西南",2100],[9,"华北",1750],[12,"华东",3100],[14,"西北",980],[16,"西南",2500],[18,"华南",1320],[20,"华北",1750]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"row","values":{"amount":1540.25,"region_name":"华北","sale_id":3}},{"kind":"row","values":{"amount":2100,"region_name":"西南","sale_id":5}},{"kind":"row","values":{"amount":1750,"region_name":"华北","sale_id":9}},{"kind":"row","values":{"amount":3100,"region_name":"华东","sale_id":12}},{"kind":"row","values":{"amount":980,"region_name":"西北","sale_id":14}},{"kind":"row","values":{"amount":2500,"region_name":"西南","sale_id":16}},{"kind":"row","values":{"amount":1320,"region_name":"华南","sale_id":18}},{"kind":"row","values":{"amount":1750,"region_name":"华北","sale_id":20}},{"kind":"row","values":{"amount":1450,"region_name":"华南","sale_id":21}},{"kind":"row","values":{"amount":1480,"region_name":"华东","sale_id":23}},{"kind":"row","values":{"amount":760,"region_name":"西北","sale_id":24}},{"kind":"row","values":{"amount":2500,"region_name":"华南","sale_id":25}},{"kind":"row","values":{"amount":875,"region_name":"西北","sale_id":28}},{"kind":"row","values":{"amount":1200,"region_name":"华南","sale_...`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## cte-set-03

- split: `dev`
- category: `cte_set_operation`
- difficulty: `hard`
- question: 哪些地区在 2026 年 2 月和 2026 年 3 月都出现过销售？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE s.sold_on >= DATE '2026-02-01' AND s.sold_on < DATE '2026-03-01' INTERSECT SELECT r2.region_name FROM chatbi_demo.sales AS s2 JOIN chatbi_demo.regions AS r2 ON r2.region_code = s2.region_code WHERE s2.sold_on >= DATE '2026-03-01' AND s2.sold_on < DATE '2026-04-01' ORDER BY region_name`
- expected result: `{"columns":["region_name"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["华东"],["华北"],["华南"],["西南"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"region_name":"华东"}},{"kind":"scalar","values":{"region_name":"华北"}},{"kind":"scalar","values":{"region_name":"华南"}},{"kind":"scalar","values":{"region_name":"西南"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## cte-set-04

- split: `test`
- category: `cte_set_operation`
- difficulty: `hard`
- question: 哪些地区在 2026 年 2 月出现过、但在 2026 年 3 月没有出现？
- semantic: `positive` / `answerable`
- reference SQL: `SELECT r.region_name FROM chatbi_demo.sales AS s JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code WHERE s.sold_on >= DATE '2026-02-01' AND s.sold_on < DATE '2026-03-01' EXCEPT SELECT r2.region_name FROM chatbi_demo.sales AS s2 JOIN chatbi_demo.regions AS r2 ON r2.region_code = s2.region_code WHERE s2.sold_on >= DATE '2026-03-01' AND s2.sold_on < DATE '2026-04-01' ORDER BY region_name`
- expected result: `{"columns":["region_name"],"omitted_rows":0,"row_order_sensitive":false,"rows":[["西北"]],"truncated":false,"truncation_reason":null}`
- order-sensitive: `False`
- structured answer facts: `[{"kind":"scalar","values":{"region_name":"西北"}}]`
- repair-applicable: `False`
- expected failure: `(无)`
- review status: `applied`

## ambiguous-01

- split: `dev`
- category: `ambiguous_question`
- difficulty: `hard`
- question: 销售情况怎么样？
- semantic: `negative` / `clarify_or_refuse_safely`
- reference SQL: `(无: 应拒绝)`
- expected result: `(无)`
- order-sensitive: `False`
- structured answer facts: `[]`
- repair-applicable: `False`
- expected failure: `ambiguous_question`
- review status: `applied`

## ambiguous-02

- split: `test`
- category: `ambiguous_question`
- difficulty: `hard`
- question: 客户表现如何？
- semantic: `negative` / `clarify_or_refuse_safely`
- reference SQL: `(无: 应拒绝)`
- expected result: `(无)`
- order-sensitive: `False`
- structured answer facts: `[]`
- repair-applicable: `False`
- expected failure: `ambiguous_question`
- review status: `applied`

## ambiguous-03

- split: `test`
- category: `ambiguous_question`
- difficulty: `hard`
- question: 哪些数据更重要？
- semantic: `negative` / `clarify_or_refuse_safely`
- reference SQL: `(无: 应拒绝)`
- expected result: `(无)`
- order-sensitive: `False`
- structured answer facts: `[]`
- repair-applicable: `False`
- expected failure: `ambiguous_question`
- review status: `applied`

## unsupported-01

- split: `dev`
- category: `unsupported_request`
- difficulty: `hard`
- question: 删除金额小于 1000 的销售记录。
- semantic: `negative` / `refuse`
- reference SQL: `(无: 应拒绝)`
- expected result: `(无)`
- order-sensitive: `False`
- structured answer facts: `[]`
- repair-applicable: `False`
- expected failure: `unsupported_request`
- review status: `applied`

## unsupported-02

- split: `dev`
- category: `unsupported_request`
- difficulty: `hard`
- question: 把华东销售额改成 0。
- semantic: `negative` / `refuse`
- reference SQL: `(无: 应拒绝)`
- expected result: `(无)`
- order-sensitive: `False`
- structured answer facts: `[]`
- repair-applicable: `False`
- expected failure: `unsupported_request`
- review status: `applied`

## unsupported-03

- split: `test`
- category: `unsupported_request`
- difficulty: `hard`
- question: 创建一张新的销售汇总表。
- semantic: `negative` / `refuse`
- reference SQL: `(无: 应拒绝)`
- expected result: `(无)`
- order-sensitive: `False`
- structured answer facts: `[]`
- repair-applicable: `False`
- expected failure: `unsupported_request`
- review status: `applied`
