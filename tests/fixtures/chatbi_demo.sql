-- A5.1 future-evaluation fixture. Load into a separate business/demo database,
-- never into the KnowledgeScope application database.
CREATE SCHEMA IF NOT EXISTS chatbi_demo;

CREATE TABLE IF NOT EXISTS chatbi_demo.sales (
    sale_id BIGINT PRIMARY KEY,
    sold_on DATE NOT NULL,
    region TEXT NOT NULL,
    amount NUMERIC(12, 2) NOT NULL CHECK (amount >= 0)
);

INSERT INTO chatbi_demo.sales (sale_id, sold_on, region, amount)
VALUES
    (1, DATE '2026-01-05', '华东', 1200.00),
    (2, DATE '2026-01-12', '华南', 860.50),
    (3, DATE '2026-02-03', '华东', 1540.25)
ON CONFLICT (sale_id) DO NOTHING;
