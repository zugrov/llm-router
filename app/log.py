"""
Логирование запросов в request_log. Пишем только метаданные — без сырого prompt/response,
без X-Internal-Secret и без ROUTERAI_API_KEY. Схема таблицы — db/init.sql.
"""
from __future__ import annotations

import uuid

from sqlalchemy import text

from app.db import async_session


async def log_request(
    *,
    request_id: str,
    project: str,
    task_type: str,
    model_requested: str,
    model_actual: str | None,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    estimated_cost_rub: float | None,
    cost_source: str,
    status: str,
    latency_ms: int,
) -> None:
    async with async_session() as session:
        await session.execute(
            text(
                "INSERT INTO request_log ("
                "id, request_id, project, task_type, model_requested, model_actual, "
                "input_tokens, output_tokens, total_tokens, estimated_cost_rub, cost_source, "
                "status, latency_ms"
                ") VALUES ("
                ":id, :request_id, :project, :task_type, :model_requested, :model_actual, "
                ":input_tokens, :output_tokens, :total_tokens, :estimated_cost_rub, :cost_source, "
                ":status, :latency_ms"
                ")"
            ),
            {
                "id": uuid.uuid4(),
                "request_id": request_id,
                "project": project,
                "task_type": task_type,
                "model_requested": model_requested,
                "model_actual": model_actual,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_rub": estimated_cost_rub,
                "cost_source": cost_source,
                "status": status,
                "latency_ms": latency_ms,
            },
        )
        await session.commit()
