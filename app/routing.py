"""
Статическая таблица маршрутизации task_type -> (primary, fallback) модель.

Порядок в TASK_TYPES_PRIORITY отражает приоритет task_type в системе.
Реализованы и покрыты тестами:
  - extraction           — извлечение данных из PDF/DOCX/CSV (cfo-autopilot)
  - financial_analysis   — AI-чат по финансовым данным компании (cfo-autopilot)
  - chat                 — многоходовой диалог (grill: пошаговая генерация бизнес-плана)
  - client_report        — одноразовая генерация PDF-отчёта для конечного клиента
                            (maxima consulting), отдельно от financial_analysis

Зарезервировано на будущее, без реализации, чтобы не плодить мёртвый код до
появления реальных вызовов:
  - code
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings

TASK_TYPES_PRIORITY = ("extraction", "financial_analysis", "chat", "client_report", "code")


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
        "chat": ModelPair(
            primary=settings.model_chat_primary,
            fallback=settings.model_chat_fallback,
        ),
        "client_report": ModelPair(
            primary=settings.model_client_report_primary,
            fallback=settings.model_client_report_fallback,
        ),
    }


def resolve_models(task_type: str) -> ModelPair | None:
    """Возвращает пару моделей для task_type или None, если task_type неизвестен/не реализован."""
    return get_routing_table().get(task_type)
