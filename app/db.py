"""
Подключение к Postgres llm-router. Без ORM-моделей — прямой SQL через SQLAlchemy Core,
как в cfo_autopilot (см. backend/app/routers/*.py).
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import get_settings

settings = get_settings()

engine: AsyncEngine = create_async_engine(settings.database_url, pool_pre_ping=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def check_db_health() -> bool:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True
