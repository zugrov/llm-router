"""
Вызов RouterAI через AsyncOpenAI + retry/fallback state machine.

Политика (см. план llm_router_через_routerai):
- НЕ ретраим и НЕ переключаем модель на 400/401/402/403/404/422 — это ошибки клиентского
  запроса, смена модели их не исправит. Отдаём NonRetryableUpstreamError сразу.
- Fallback на другую модель разрешён только для 408/429/500/502/503/504 и сетевых ошибок
  (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError с одним из этих статусов).
- APIStatusError означает, что RouterAI уже завершил свой внутренний provider fallback и всё
  равно вернул финальную ошибку -> сразу переключаем модель, без повтора той же модели.
- APITimeoutError/APIConnectionError означает, что неясно, дошёл ли запрос до RouterAI вообще ->
  даём ОДНУ повторную попытку той же модели, и только затем переключаем модель.
- Лимиты на один request_id: максимум 2 разные модели, максимум 3 внешних запроса суммарно.
- Reasoning-модели (напр. deepseek-v4-pro-0813) могут потратить весь max_tokens на reasoning
  и вернуть пустой content с finish_reason="length" при HTTP 200 — это техническая ошибка,
  приравниваем к сбою, требующему переключения модели.

Стриминг (stream_complete) следует той же политике с одним отличием: fallback/retry
возможен только пока клиенту не отдано ни одного chunk. После первого chunk откат
модели невозможен (клиент уже увидел частичный ответ) — при сбое поток завершается
событием ошибки.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

import openai
from openai import AsyncOpenAI

from app.config import get_settings
from app.routing import resolve_models

settings = get_settings()

_client = AsyncOpenAI(api_key=settings.routerai_api_key, base_url=settings.routerai_base_url)

# HTTP-статусы, на которые fallback ЗАПРЕЩЁН — ошибка клиентского запроса/авторизации/оплаты.
_NON_RETRYABLE_STATUS = {400, 401, 402, 403, 404, 422}

# HTTP-статусы, на которые fallback РАЗРЕШЁН — технический сбой на стороне апстрима.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ErrorAction(str, Enum):
    NON_RETRYABLE = "non_retryable"
    RETRY_SAME = "retry_same"
    SWITCH_MODEL = "switch_model"


class EmptyReasoningOutputError(Exception):
    """RouterAI вернул 200, но content пуст — reasoning-модель исчерпала max_tokens на рассуждения."""


class NonRetryableUpstreamError(Exception):
    """Ошибка апстрима, на которую fallback запрещён политикой (см. модуль)."""

    def __init__(self, status_code: int | None, message: str):
        self.status_code = status_code
        super().__init__(message)


class UpstreamExhaustedError(Exception):
    """Все разрешённые попытки (primary + fallback) исчерпаны технической ошибкой."""

    def __init__(self, status_code: int | None, message: str):
        self.status_code = status_code
        super().__init__(message)


@dataclass
class CompletionResult:
    content: str
    model_requested: str
    model_actual: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_rub: float | None
    cost_source: str


def _is_reasoning_model(model: str) -> bool:
    return model in settings.reasoning_models_set


def classify_exception(exc: Exception) -> tuple[ErrorAction, int | None]:
    """Классифицирует исключение OpenAI SDK по политике retry/fallback."""
    if isinstance(exc, EmptyReasoningOutputError):
        return ErrorAction.SWITCH_MODEL, 200
    if isinstance(exc, openai.BadRequestError):
        return ErrorAction.NON_RETRYABLE, exc.status_code
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return ErrorAction.RETRY_SAME, None
    if isinstance(exc, openai.RateLimitError):
        return ErrorAction.SWITCH_MODEL, 429
    if isinstance(exc, openai.APIStatusError):
        status_code = exc.status_code
        if status_code in _NON_RETRYABLE_STATUS:
            return ErrorAction.NON_RETRYABLE, status_code
        if status_code in _RETRYABLE_STATUS:
            return ErrorAction.SWITCH_MODEL, status_code
        # Неизвестный статус апстрима — консервативно не ретраим.
        return ErrorAction.NON_RETRYABLE, status_code
    # Непредвиденное исключение — не ретраим, чтобы не зациклиться на баге вместо реальной ошибки.
    return ErrorAction.NON_RETRYABLE, None


def _extract_cost(usage: object) -> float | None:
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None)
        if extra:
            cost = extra.get("cost")
    return float(cost) if isinstance(cost, (int, float)) else None


async def _call_model(model: str, messages: list[dict[str, str]], max_tokens: int) -> CompletionResult:
    """Один внешний HTTP-вызов к RouterAI под конкретной моделью, без ретраев внутри."""
    request_max_tokens = max_tokens
    if _is_reasoning_model(model):
        request_max_tokens = max_tokens + settings.reasoning_token_buffer

    response = await _client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=request_max_tokens,
    )

    choice = response.choices[0]
    content = (choice.message.content or "").strip()

    if choice.finish_reason == "length" and not content:
        raise EmptyReasoningOutputError(
            f"model={model}: пустой content при finish_reason=length "
            f"(не хватило max_tokens={request_max_tokens} на reasoning)"
        )

    usage = response.usage
    return CompletionResult(
        content=content,
        model_requested=model,
        model_actual=model,
        input_tokens=(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
        output_tokens=(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        total_tokens=(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
        estimated_cost_rub=_extract_cost(usage),
        cost_source="routerai_response" if _extract_cost(usage) is not None else "unavailable",
    )


async def complete(task_type: str, messages: list[dict[str, str]], max_tokens: int) -> CompletionResult:
    """
    Оркестрирует вызов RouterAI с retry/fallback по политике модуля:
    максимум 2 модели (primary, fallback), максимум 3 внешних запроса суммарно.
    """
    pair = resolve_models(task_type)
    if pair is None:
        raise NonRetryableUpstreamError(None, f"неизвестный или нереализованный task_type: {task_type}")

    last_status: int | None = None
    last_message = "unknown error"

    # (модель, максимум попыток именно этой модели): primary — до 2 (1 + 1 retry на network error),
    # fallback — 1 (переключение уже финальное, без повторной ретраи RouterAI provider fallback).
    for model, max_attempts in ((pair.primary, 2), (pair.fallback, 1)):
        for attempt in range(max_attempts):
            try:
                result = await _call_model(model, messages, max_tokens)
                result.model_requested = pair.primary
                return result
            except Exception as exc:  # классифицируем ниже по типу исключения OpenAI SDK
                action, status_code = classify_exception(exc)
                last_status, last_message = status_code, str(exc)
                if action == ErrorAction.NON_RETRYABLE:
                    raise NonRetryableUpstreamError(status_code, str(exc)) from exc
                if action == ErrorAction.RETRY_SAME and attempt == 0:
                    continue  # ещё одна попытка той же модели (только сетевые ошибки)
                break  # переходим к следующей модели (primary -> fallback)

    raise UpstreamExhaustedError(last_status, last_message)


@dataclass
class StreamChunk:
    """Событие потока: либо текстовая дельта, либо финальный результат, либо ошибка."""

    delta: str | None = None
    done: bool = False
    error: Exception | None = None
    result: CompletionResult | None = None


async def stream_complete(
    task_type: str, messages: list[dict[str, str]], max_tokens: int
) -> AsyncIterator[StreamChunk]:
    """
    Стриминговый аналог complete(). Retry/fallback работает только до первого
    отданного клиенту chunk'а — после этого при сбое поток завершается StreamChunk(error=...).
    """
    pair = resolve_models(task_type)
    if pair is None:
        yield StreamChunk(error=NonRetryableUpstreamError(None, f"неизвестный или нереализованный task_type: {task_type}"))
        return

    last_status: int | None = None
    last_message = "unknown error"
    chunks_emitted = 0

    for model, max_attempts in ((pair.primary, 2), (pair.fallback, 1)):
        for attempt in range(max_attempts):
            request_max_tokens = max_tokens
            if _is_reasoning_model(model):
                request_max_tokens = max_tokens + settings.reasoning_token_buffer

            content_parts: list[str] = []
            usage = None
            try:
                stream = await _client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=request_max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for event in stream:
                    if event.usage is not None:
                        usage = event.usage
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta.content or ""
                    if delta:
                        content_parts.append(delta)
                        chunks_emitted += 1
                        yield StreamChunk(delta=delta)

                content = "".join(content_parts).strip()
                if not content:
                    raise EmptyReasoningOutputError(
                        f"model={model}: пустой content в потоке "
                        f"(не хватило max_tokens={request_max_tokens} на reasoning)"
                    )

                yield StreamChunk(
                    done=True,
                    result=CompletionResult(
                        content=content,
                        model_requested=pair.primary,
                        model_actual=model,
                        input_tokens=(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                        output_tokens=(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                        total_tokens=(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
                        estimated_cost_rub=_extract_cost(usage),
                        cost_source="routerai_response" if _extract_cost(usage) is not None else "unavailable",
                    ),
                )
                return
            except Exception as exc:
                if chunks_emitted > 0:
                    # Клиент уже получил часть ответа — сменить модель нельзя, поток завершаем ошибкой.
                    yield StreamChunk(error=exc)
                    return
                action, status_code = classify_exception(exc)
                last_status, last_message = status_code, str(exc)
                if action == ErrorAction.NON_RETRYABLE:
                    yield StreamChunk(error=NonRetryableUpstreamError(status_code, str(exc)))
                    return
                if action == ErrorAction.RETRY_SAME and attempt == 0:
                    continue
                break

    yield StreamChunk(error=UpstreamExhaustedError(last_status, last_message))
