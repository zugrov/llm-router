"""
Статическая таблица маршрутизации task_type -> (primary, fallback) модель.

Порядок в TASK_TYPES_PRIORITY отражает приоритет task_type в системе.
Реализованы и покрыты тестами только те типы, что реально используются
потребителями (сейчас — только cfo-autopilot):
  - extraction           — извлечение данных из PDF/DOCX/CSV
  - financial_analysis   — AI-чат по финансовым данным компании (бывший "report")

Зарезервированы на будущее (grill, maxima consulting), без реализации,
чтобы не плодить мёртвый код до появления реальных вызовов:
  - code
  - chat
  - client_report        — будущая генерация PDF-отчётов через LLM, отдельно от financial_analysis
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings

TASK_TYPES_PRIORITY = ("extraction", "financial_analysis", "code", "chat")


@dataclass(frozen=True)
class ModelPair:
    primary: str
    fallback: str


def get_routing_table() -> dict[str, ModelPair]:
    settings = get_settings()
    return {
        "extraction": ModelPair(
            primary=settings.model_extraction_primary,
            fallback=settings.model_extraction_fallback,
        ),
        "financial_analysis": ModelPair(
            primary=settings.model_financial_analysis_primary,
            fallback=settings.model_financial_analysis_fallback,
        ),
    }


def resolve_models(task_type: str) -> ModelPair | None:
    """Возвращает пару моделей для task_type или None, если task_type неизвестен/не реализован."""
    return get_routing_table().get(task_type)
