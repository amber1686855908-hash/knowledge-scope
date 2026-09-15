-- A5.7 v2 formal benchmark fixture. Load into an isolated business database;
-- never load this fixture into the KnowledgeScope application database.
DROP SCHEMA IF EXISTS chatbi_demo CASCADE;
CREATE SCHEMA chatbi_demo;

CREATE TABLE chatbi_demo.customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name TEXT NOT NULL
);

CREATE TABLE chatbi_demo.regions (
    region_code TEXT PRIMARY KEY,
    region_name TEXT NOT NULL UNIQUE,
    market TEXT NOT NULL
);

CREATE TABLE chatbi_demo.sales (
    sale_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES chatbi_demo.customers(customer_id),
    region_code TEXT NOT NULL REFERENCES chatbi_demo.regions(region_code),
    sold_on DATE NOT NULL,
    amount NUMERIC(12, 2) NOT NULL CHECK (amount >= 0)
);

COMMENT ON TABLE chatbi_demo.customers IS 'Business customer dimension';
COMMENT ON COLUMN chatbi_demo.customers.customer_id IS 'Stable customer identifier';
COMMENT ON COLUMN chatbi_demo.customers.customer_name IS 'Display name, not a business key';
COMMENT ON TABLE chatbi_demo.regions IS 'Sales region dimension';
COMMENT ON COLUMN chatbi_demo.regions.region_code IS 'Stable region identifier';
COMMENT ON TABLE chatbi_demo.sales IS 'Recorded sales by customer, region and date';
COMMENT ON COLUMN chatbi_demo.sales.amount IS 'Non-negative sale amount';

CREATE VIEW chatbi_demo.region_sales AS
SELECT
    r.region_code,
    r.region_name,
    COUNT(s.sale_id) AS sale_count,
    COALESCE(SUM(s.amount), 0) AS total_amount
FROM chatbi_demo.regions AS r
LEFT JOIN chatbi_demo.sales AS s ON s.region_code = r.region_code
GROUP BY r.region_code, r.region_name;

COMMENT ON VIEW chatbi_demo.region_sales IS 'Derived regional sales summary, not an executable v2 oracle target';

INSERT INTO chatbi_demo.customers (customer_id, customer_name)
VALUES
    (1, '北辰制造'),
    (2, '星河贸易'),
    (3, '远山科技'),
    (4, '江南医药'),
    (5, '海岳能源'),
    (6, '北辰制造'),
    (7, '晨光物流'),
    (8, '云杉教育'),
    (9, '南岭食品'),
    (10, '未成交客户');

INSERT INTO chatbi_demo.regions (region_code, region_name, market)
VALUES
    ('R1', '华东', '东部市场'),
    ('R2', '华南', '南部市场'),
    ('R3', '华北', '北部市场'),
    ('R4', '西南', '西部市场'),
    ('R5', '西北', '西部市场');

INSERT INTO chatbi_demo.sales (sale_id, customer_id, region_code, sold_on, amount)
VALUES
    (1, 1, 'R1', DATE '2026-01-01', 1200.00),
    (2, 2, 'R2', DATE '2026-01-05', 860.50),
    (3, 1, 'R3', DATE '2026-01-12', 1540.25),
    (4, 3, 'R1', DATE '2026-01-31', 999.99),
    (5, 4, 'R4', DATE '2026-02-01', 2100.00),
    (6, 5, 'R2', DATE '2026-02-02', 750.00),
    (7, 6, 'R1', DATE '2026-02-10', 1200.00),
    (8, 7, 'R5', DATE '2026-02-15', 430.00),
    (9, 8, 'R3', DATE '2026-02-28', 1750.00),
    (10, 2, 'R2', DATE '2026-03-01', 860.50),
    (11, 3, 'R4', DATE '2026-03-03', 1120.00),
    (12, 1, 'R1', DATE '2026-03-15', 3100.00),
    (13, 4, 'R2', DATE '2026-03-31', 640.00),
    (14, 5, 'R5', DATE '2026-04-01', 980.00),
    (15, 6, 'R3', DATE '2026-04-10', 1200.00),
    (16, 7, 'R4', DATE '2026-04-20', 2500.00),
    (17, 8, 'R1', DATE '2026-04-30', 540.00),
    (18, 9, 'R2', DATE '2026-05-01', 1320.00),
    (19, 2, 'R5', DATE '2026-05-12', 680.00),
    (20, 3, 'R3', DATE '2026-05-31', 1750.00),
    (21, 1, 'R2', DATE '2026-06-01', 1450.00),
    (22, 4, 'R4', DATE '2026-06-05', 920.00),
    (23, 5, 'R1', DATE '2026-06-15', 1480.00),
    (24, 6, 'R5', DATE '2026-06-30', 760.00),
    (25, 7, 'R2', DATE '2026-01-20', 2500.00),
    (26, 8, 'R4', DATE '2026-02-20', 660.00),
    (27, 9, 'R3', DATE '2026-03-20', 1320.00),
    (28, 1, 'R5', DATE '2026-04-15', 875.00),
    (29, 3, 'R1', DATE '2026-05-15', 940.00),
    (30, 6, 'R2', DATE '2026-06-10', 1200.00);
