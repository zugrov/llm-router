"""Тесты retry/fallback state machine (app/upstream.py) — политика из плана llm-router."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from app.upstream import (
    EmptyReasoningOutputError,
    ErrorAction,
    NonRetryableUpstreamError,
    UpstreamExhaustedError,
    _call_model,
    classify_exception,
    complete,
)

PRIMARY = "z-ai/glm-5.3-flash"
FALLBACK = "deepseek/deepseek-v4-pro-0813"


def _status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://routerai.ru/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": "boom"})
    return openai.APIStatusError("boom", response=response, body={"error": "boom"})


def _bad_request_error() -> openai.BadRequestError:
    request = httpx.Request("POST", "https://routerai.ru/api/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": "bad"})
    return openai.BadRequestError("bad", response=response, body={"error": "bad"})


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://routerai.ru/api/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": "slow down"})
    return openai.RateLimitError("slow down", response=response, body={"error": "slow down"})


def _timeout_error() -> openai.APITimeoutError:
    request = httpx.Request("POST", "https://routerai.ru/api/v1/chat/completions")
    return openai.APITimeoutError(request=request)


def _connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://routerai.ru/api/v1/chat/completions")
    return openai.APIConnectionError(request=request)


def _fake_response(content: str, finish_reason: str = "stop", cost: float = 0.001):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=cost)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestClassifyException:
    @pytest.mark.parametrize("status_code", [400, 401, 402, 403, 404, 422])
    def test_non_retryable_statuses_no_fallback(self, status_code):
        action, code = classify_exception(_status_error(status_code))
        assert action == ErrorAction.NON_RETRYABLE
        assert code == status_code

    @pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
    def test_retryable_statuses_switch_model(self, status_code):
        action, code = classify_exception(_status_error(status_code))
        assert action == ErrorAction.SWITCH_MODEL
        assert code == status_code

    def test_bad_request_error_is_non_retryable(self):
        action, code = classify_exception(_bad_request_error())
        assert action == ErrorAction.NON_RETRYABLE
        assert code == 400

    def test_rate_limit_error_switches_model(self):
        action, code = classify_exception(_rate_limit_error())
        assert action == ErrorAction.SWITCH_MODEL
        assert code == 429

    def test_timeout_retries_same_model(self):
        action, _ = classify_exception(_timeout_error())
        assert action == ErrorAction.RETRY_SAME

    def test_connection_error_retries_same_model(self):
        action, _ = classify_exception(_connection_error())
        assert action == ErrorAction.RETRY_SAME

    def test_empty_reasoning_output_switches_model(self):
        action, _ = classify_exception(EmptyReasoningOutputError("empty"))
        assert action == ErrorAction.SWITCH_MODEL


class TestCallModelEmptyReasoning:
    @pytest.mark.asyncio
    async def test_length_with_empty_content_raises(self):
        response = _fake_response(content="", finish_reason="length")
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            with pytest.raises(EmptyReasoningOutputError):
                await _call_model(PRIMARY, "system", "prompt", 20)

    @pytest.mark.asyncio
    async def test_length_with_nonempty_content_ok(self):
        response = _fake_response(content="4", finish_reason="length")
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            result = await _call_model(PRIMARY, "system", "prompt", 20)
            assert result.content == "4"

    @pytest.mark.asyncio
    async def test_real_cost_from_usage(self):
        response = _fake_response(content="4", cost=0.10128836718)
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            result = await _call_model(PRIMARY, "system", "prompt", 20)
            assert result.estimated_cost_rub == pytest.approx(0.10128836718)
            assert result.cost_source == "routerai_response"


class TestCompleteStateMachine:
    """Проверка лимитов: максимум 2 модели, максимум 3 внешних запроса на request_id."""

    @pytest.mark.asyncio
    async def test_success_on_primary_first_try(self):
        response = _fake_response("42")
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            result = await complete("extraction", "sys", "prompt", 100)
            assert result.model_actual == PRIMARY
            assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_fallback_single_call(self):
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(side_effect=_status_error(402))
            with pytest.raises(NonRetryableUpstreamError) as exc_info:
                await complete("extraction", "sys", "prompt", 100)
            assert exc_info.value.status_code == 402
            assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_bad_request_400_no_fallback(self):
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(side_effect=_bad_request_error())
            with pytest.raises(NonRetryableUpstreamError):
                await complete("extraction", "sys", "prompt", 100)
            assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_503_switches_to_fallback_model_immediately(self):
        response = _fake_response("ok")
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[_status_error(503), response]
            )
            result = await complete("extraction", "sys", "prompt", 100)
            assert result.model_actual == FALLBACK
            assert result.model_requested == PRIMARY
            assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_retries_same_model_once_then_succeeds(self):
        response = _fake_response("ok")
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[_timeout_error(), response]
            )
            result = await complete("extraction", "sys", "prompt", 100)
            assert result.model_actual == PRIMARY
            assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_twice_then_switches_to_fallback(self):
        response = _fake_response("ok")
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[_timeout_error(), _timeout_error(), response]
            )
            result = await complete("extraction", "sys", "prompt", 100)
            assert result.model_actual == FALLBACK
            # Ровно 3 внешних запроса: primary x2 + fallback x1 — верхняя граница лимита.
            assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_never_exceeds_3_attempts_or_2_models(self):
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    _timeout_error(),
                    _timeout_error(),
                    _status_error(503),
                    _fake_response("should not be reached"),
                ]
            )
            with pytest.raises(UpstreamExhaustedError):
                await complete("extraction", "sys", "prompt", 100)
            assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_both_models_fail_technically_raises_exhausted(self):
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[_status_error(503), _status_error(502)]
            )
            with pytest.raises(UpstreamExhaustedError) as exc_info:
                await complete("extraction", "sys", "prompt", 100)
            assert exc_info.value.status_code == 502
            assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_on_fallback_still_raises_immediately(self):
        with patch("app.upstream._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[_status_error(503), _status_error(401)]
            )
            with pytest.raises(NonRetryableUpstreamError) as exc_info:
                await complete("extraction", "sys", "prompt", 100)
            assert exc_info.value.status_code == 401
            assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_unknown_task_type_raises_non_retryable(self):
        with pytest.raises(NonRetryableUpstreamError):
            await complete("unknown_task", "sys", "prompt", 100)
