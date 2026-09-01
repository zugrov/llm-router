-- Схема request_log для llm-router.
-- Без сырого prompt/response — только метаданные вызова (см. app/log.py).

CREATE TABLE IF NOT EXISTS request_log (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL,
    project TEXT NOT NULL,
    task_type TEXT NOT NULL,
    model_requested TEXT NOT NULL,
    model_actual TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_rub NUMERIC(14, 6),
    cost_source TEXT NOT NULL DEFAULT 'unavailable',
    status TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_request_log_created_at ON request_log (created_at);
CREATE INDEX IF NOT EXISTS idx_request_log_project_task_type ON request_log (project, task_type);
CREATE INDEX IF NOT EXISTS idx_request_log_request_id ON request_log (request_id);
