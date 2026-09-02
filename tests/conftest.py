"""Общие фикстуры и тестовое окружение llm-router. Импортируется pytest раньше test-модулей,
поэтому переменные окружения должны быть выставлены здесь до первого импорта app.*
"""
import os

os.environ.setdefault("ROUTERAI_API_KEY", "sk-test-key")
os.environ.setdefault("INTERNAL_SECRET", "test-internal-secret")
os.environ.setdefault("ALLOWED_PROJECTS", "cfo-autopilot,grill,maxima-consulting")
# Заведомо недоступный адрес — быстрый ECONNREFUSED вместо реальной БД в тестах.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@127.0.0.1:59999/test")

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

AUTH_HEADERS = {"X-Internal-Secret": "test-internal-secret"}


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
