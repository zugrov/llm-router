"""
llm-router — единая точка входа для LLM-вызовов из нескольких MVP (v1: только cfo-autopilot).

POST /v1/complete — маршрутизация по task_type (routing.py), retry/fallback на технических
ошибках (upstream.py), логирование метаданных без prompt/response (log.py).
"""
from __future__ import annotations

import logging
import secrets
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.db import check_db_health
from app.log import log_request
from app.routing import resolve_models
from app.upstream import NonRetryableUpstreamError, UpstreamExhaustedError, complete

settings = get_settings()
logger = logging.getLogger("llm_router")

app = FastAPI(title="llm-router")


class CompleteRequest(BaseModel):
    # extra="forbid" — клиент не может передать model/provider/url или произвольную routing policy,
    # модель выбирает исключительно routing.py по task_type.
    model_config = ConfigDict(extra="forbid")

    project: str
    task_type: str
    system: str
    prompt: str
    max_tokens: int = Field(default=2000, gt=0, le=32000)
    request_id: str | None = None


class CompleteResponse(BaseModel):
    request_id: str
    content: str
    model_requested: str
    model_actual: str


def _check_internal_secret(x_internal_secret: str | None) -> None:
    expected = settings.internal_secret
    if not expected or not x_internal_secret or not secrets.compare_digest(x_internal_secret, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED"},
        )


def _check_project_allowed(project: str) -> None:
    if project not in settings.allowed_projects_set:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_code": "PROJECT_NOT_ALLOWED"},
        )


async def _safe_log(
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
    status_value: str,
    started: float,
) -> None:
    """Логирование не должно ронять основной запрос при недоступности БД."""
    latency_ms = int((time.monotonic() - started) * 1000)
    try:
        await log_request(
            request_id=request_id,
            project=project,
            task_type=task_type,
            model_requested=model_requested,
            model_actual=model_actual,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_rub=estimated_cost_rub,
            cost_source=cost_source,
            status=status_value,
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("Не удалось записать request_log (request_id=%s)", request_id)


@app.get("/health")
async def health() -> dict:
    try:
        await check_db_health()
    except Exception:
        raise HTTPException(status_code=503, detail={"error_code": "DB_UNAVAILABLE"}) from None
    return {"status": "ok"}


@app.post("/v1/complete", response_model=CompleteResponse)
async def v1_complete(
    body: CompleteRequest,
    x_internal_secret: str | None = Header(default=None, alias="X-Internal-Secret"),
) -> CompleteResponse:
    _check_internal_secret(x_internal_secret)
    _check_project_allowed(body.project)

    request_id = body.request_id or str(uuid.uuid4())
    started = time.monotonic()
    pair = resolve_models(body.task_type)
    model_requested = pair.primary if pair else body.task_type

    try:
        result = await complete(
            task_type=body.task_type,
            system=body.system,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
        )
    except NonRetryableUpstreamError as exc:
        await _safe_log(
            request_id=request_id,
            project=body.project,
            task_type=body.task_type,
            model_requested=model_requested,
            model_actual=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_rub=None,
            cost_source="unavailable",
            status_value="error",
            started=started,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"request_id": request_id, "error_code": "INVALID_REQUEST"},
        ) from exc
    except UpstreamExhaustedError as exc:
        await _safe_log(
            request_id=request_id,
            project=body.project,
            task_type=body.task_type,
            model_requested=model_requested,
            model_actual=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_rub=None,
            cost_source="unavailable",
            status_value="error",
            started=started,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"request_id": request_id, "error_code": "MODEL_UNAVAILABLE"},
        ) from exc

    await _safe_log(
        request_id=request_id,
        project=body.project,
        task_type=body.task_type,
        model_requested=result.model_requested,
        model_actual=result.model_actual,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        estimated_cost_rub=result.estimated_cost_rub,
        cost_source=result.cost_source,
        status_value="ok",
        started=started,
    )

    return CompleteResponse(
        request_id=request_id,
        content=result.content,
        model_requested=result.model_requested,
        model_actual=result.model_actual,
    )
