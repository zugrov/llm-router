"""Тесты POST /v1/complete и GET /health — auth, allowlist, error codes, отсутствие утечек."""
import logging
from unittest.mock import AsyncMock, patch

import pytest

from app.upstream import CompletionResult, NonRetryableUpstreamError, UpstreamExhaustedError
from .conftest import AUTH_HEADERS

SECRET_PROMPT = "СЕКРЕТНЫЙ_ТЕКСТ_КОТОРОГО_НЕ_ДОЛЖНО_БЫТЬ_В_ЛОГАХ"


def _ok_result(model_actual: str = "z-ai/glm-5.3-flash") -> CompletionResult:
    return CompletionResult(
        content="ответ модели",
        model_requested="z-ai/glm-5.3-flash",
        model_actual=model_actual,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        estimated_cost_rub=0.0001,
        cost_source="routerai_response",
    )


class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_secret_401(self, client):
        resp = await client.post(
            "/v1/complete",
            json={"project": "cfo-autopilot", "task_type": "extraction", "system": "s", "prompt": "p"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"]["error_code"] == "UNAUTHORIZED"

    @pytest.mark.asyncio
    async def test_wrong_secret_401(self, client):
        resp = await client.post(
            "/v1/complete",
            json={"project": "cfo-autopilot", "task_type": "extraction", "system": "s", "prompt": "p"},
            headers={"X-Internal-Secret": "wrong-secret"},
        )
        assert resp.status_code == 401


class TestAllowlist:
    @pytest.mark.asyncio
    async def test_project_not_in_allowlist_403(self, client):
        with patch("app.upstream._client.chat.completions.create", new=AsyncMock()):
            resp = await client.post(
                "/v1/complete",
                json={"project": "grill", "task_type": "extraction", "system": "s", "prompt": "p"},
                headers=AUTH_HEADERS,
            )
        assert resp.status_code == 403
        assert resp.json()["detail"]["error_code"] == "PROJECT_NOT_ALLOWED"


class TestRequestSchema:
    @pytest.mark.asyncio
    async def test_extra_field_model_rejected(self, client):
        resp = await client.post(
            "/v1/complete",
            json={
                "project": "cfo-autopilot",
                "task_type": "extraction",
                "system": "s",
                "prompt": "p",
                "model": "anthropic/claude-opus-5",
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_extra_field_provider_rejected(self, client):
        resp = await client.post(
            "/v1/complete",
            json={
                "project": "cfo-autopilot",
                "task_type": "extraction",
                "system": "s",
                "prompt": "p",
                "provider": "anthropic",
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 422


class TestComplete:
    @pytest.mark.asyncio
    async def test_success_returns_request_id_and_models(self, client):
        with patch("app.main.complete", new=AsyncMock(return_value=_ok_result())):
            resp = await client.post(
                "/v1/complete",
                json={
                    "project": "cfo-autopilot",
                    "task_type": "extraction",
                    "system": "s",
                    "prompt": "p",
                    "request_id": "my-request-id",
                },
                headers=AUTH_HEADERS,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"] == "my-request-id"
        assert body["content"] == "ответ модели"
        assert body["model_actual"] == "z-ai/glm-5.3-flash"

    @pytest.mark.asyncio
    async def test_request_id_generated_when_absent(self, client):
        with patch("app.main.complete", new=AsyncMock(return_value=_ok_result())):
            resp = await client.post(
                "/v1/complete",
                json={"project": "cfo-autopilot", "task_type": "extraction", "system": "s", "prompt": "p"},
                headers=AUTH_HEADERS,
            )
        assert resp.status_code == 200
        assert resp.json()["request_id"]

    @pytest.mark.asyncio
    async def test_non_retryable_maps_to_422_invalid_request(self, client):
        with patch(
            "app.main.complete",
            new=AsyncMock(side_effect=NonRetryableUpstreamError(402, "недостаточно средств")),
        ):
            resp = await client.post(
                "/v1/complete",
                json={"project": "cfo-autopilot", "task_type": "extraction", "system": "s", "prompt": "p"},
                headers=AUTH_HEADERS,
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error_code"] == "INVALID_REQUEST"
        # Сырой текст ошибки апстрима не должен попадать в ответ клиенту.
        assert "недостаточно средств" not in resp.text

    @pytest.mark.asyncio
    async def test_upstream_exhausted_maps_to_502_model_unavailable(self, client):
        with patch(
            "app.main.complete",
            new=AsyncMock(side_effect=UpstreamExhaustedError(503, "внутренний текст ошибки RouterAI")),
        ):
            resp = await client.post(
                "/v1/complete",
                json={"project": "cfo-autopilot", "task_type": "extraction", "system": "s", "prompt": "p"},
                headers=AUTH_HEADERS,
            )
        assert resp.status_code == 502
        assert resp.json()["detail"]["error_code"] == "MODEL_UNAVAILABLE"
        assert "внутренний текст ошибки RouterAI" not in resp.text


class TestNoSecretsInLogs:
    @pytest.mark.asyncio
    async def test_prompt_and_secret_not_logged_on_success(self, client, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch("app.main.complete", new=AsyncMock(return_value=_ok_result())):
                resp = await client.post(
                    "/v1/complete",
                    json={
                        "project": "cfo-autopilot",
                        "task_type": "extraction",
                        "system": "system prompt",
                        "prompt": SECRET_PROMPT,
                    },
                    headers=AUTH_HEADERS,
                )
        assert resp.status_code == 200
        assert SECRET_PROMPT not in caplog.text
        assert AUTH_HEADERS["X-Internal-Secret"] not in caplog.text

    @pytest.mark.asyncio
    async def test_prompt_and_secret_not_logged_on_error(self, client, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch(
                "app.main.complete",
                new=AsyncMock(side_effect=UpstreamExhaustedError(502, "boom")),
            ):
                resp = await client.post(
                    "/v1/complete",
                    json={
                        "project": "cfo-autopilot",
                        "task_type": "extraction",
                        "system": "system prompt",
                        "prompt": SECRET_PROMPT,
                    },
                    headers=AUTH_HEADERS,
                )
        assert resp.status_code == 502
        assert SECRET_PROMPT not in caplog.text
        assert AUTH_HEADERS["X-Internal-Secret"] not in caplog.text


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        with patch("app.main.check_db_health", new=AsyncMock(return_value=True)):
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_db_unavailable(self, client):
        with patch("app.main.check_db_health", new=AsyncMock(side_effect=ConnectionError("db down"))):
            resp = await client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["detail"]["error_code"] == "DB_UNAVAILABLE"
        assert "db down" not in resp.text
