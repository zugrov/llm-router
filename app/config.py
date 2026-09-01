"""
Настройки llm-router. Все значения читаются из .env / переменных окружения.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"

    # RouterAI (единственный апстрим в v1)
    routerai_api_key: str = ""
    routerai_base_url: str = "https://routerai.ru/api/v1"

    # Модели по task_type: primary / fallback (см. routing.py)
    model_extraction_primary: str = "z-ai/glm-5.3-flash"
    model_extraction_fallback: str = "deepseek/deepseek-v4-pro-0813"
    model_financial_analysis_primary: str = "anthropic/claude-sonnet-5"
    model_financial_analysis_fallback: str = "deepseek/deepseek-v4-pro-0813"

    # Reasoning-модели тратят часть max_tokens на рассуждения до content —
    # для них увеличиваем запрошенный max_tokens и проверяем на пустой ответ (см. upstream.py)
    reasoning_models: str = "deepseek/deepseek-v4-pro-0813"
    reasoning_token_buffer: int = 1500

    # Auth между сервисами (X-Internal-Secret, сверяется через secrets.compare_digest)
    internal_secret: str = ""

    # Allowlist проектов-потребителей — в v1 разрешён только cfo-autopilot
    allowed_projects: str = "cfo-autopilot"

    # Postgres (только request_log, без сырого prompt/response)
    database_url: str = "postgresql+asyncpg://llm_router:llm_router@db:5432/llm_router"

    @property
    def allowed_projects_set(self) -> set[str]:
        return {p.strip() for p in self.allowed_projects.split(",") if p.strip()}

    @property
    def reasoning_models_set(self) -> set[str]:
        return {m.strip() for m in self.reasoning_models.split(",") if m.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
