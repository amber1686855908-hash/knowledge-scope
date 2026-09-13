-- A5.2 schema-discovery fixture. Load into a separate business/demo database,
-- never into the KnowledgeScope application database.
DROP SCHEMA IF EXISTS chatbi_demo CASCADE;
CREATE SCHEMA IF NOT EXISTS chatbi_demo;

CREATE TABLE chatbi_demo.customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS chatbi_demo.sales (
    sale_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES chatbi_demo.customers(customer_id),
    sold_on DATE NOT NULL,
    region TEXT NOT NULL,
    amount NUMERIC(12, 2) NOT NULL CHECK (amount >= 0)
);

COMMENT ON TABLE chatbi_demo.customers IS 'Business customer dimension';
COMMENT ON COLUMN chatbi_demo.customers.customer_name IS 'Display name';
COMMENT ON TABLE chatbi_demo.sales IS 'Recorded sales by region and date';
COMMENT ON COLUMN chatbi_demo.sales.amount IS 'Non-negative sale amount';

CREATE VIEW chatbi_demo.region_sales AS
SELECT region, COUNT(*) AS sale_count, SUM(amount) AS total_amount
FROM chatbi_demo.sales
GROUP BY region;

COMMENT ON VIEW chatbi_demo.region_sales IS 'Aggregated sales by region';

INSERT INTO chatbi_demo.customers (customer_id, customer_name)
VALUES
    (1, '示例客户 A'),
    (2, '示例客户 B');

INSERT INTO chatbi_demo.sales (sale_id, customer_id, sold_on, region, amount)
VALUES
    (1, 1, DATE '2026-01-05', '华东', 1200.00),
    (2, 2, DATE '2026-01-12', '华南', 860.50),
    (3, 1, DATE '2026-02-03', '华东', 1540.25);
