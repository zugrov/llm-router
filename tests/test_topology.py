"""Статическая проверка network topology docker-compose.yml (см. план, раздел «Docker и сети»)."""
from pathlib import Path

import yaml

COMPOSE_PATH = Path(__file__).resolve().parent.parent / "docker-compose.yml"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text())


class TestDockerComposeTopology:
    def test_db_has_no_published_ports(self):
        compose = _load_compose()
        db = compose["services"]["llm-router-db"]
        assert "ports" not in db, "Postgres не должен публиковать порты наружу"

    def test_db_only_on_internal_network(self):
        compose = _load_compose()
        db_networks = compose["services"]["llm-router-db"]["networks"]
        assert db_networks == ["llm_router_internal"]
        assert "cfo_autopilot_default" not in db_networks

    def test_db_service_name_does_not_collide_with_other_projects(self):
        """Регрессия: сервис "db" на shared внешней сети коллизирует с cfo_autopilot-db-1
        (одноимённый сервис из docker-compose.prod.yml cfo-autopilot) — Docker DNS резолвит
        "db" в чужой контейнер вместо своего, и llm-router-app подключается не к своей БД.
        """
        compose = _load_compose()
        assert "db" not in compose["services"], (
            "имя сервиса 'db' занято cfo_autopilot-db-1 в сети cfo_autopilot_default"
        )

    def test_app_has_no_published_ports_only_expose(self):
        compose = _load_compose()
        app = compose["services"]["llm-router"]
        assert "ports" not in app, "llm-router не должен публиковать порт на хост"
        assert app.get("expose") == ["8000"]

    def test_app_on_both_networks(self):
        compose = _load_compose()
        app_networks = compose["services"]["llm-router"]["networks"]
        assert "cfo_autopilot_default" in app_networks
        assert "llm_router_internal" in app_networks

    def test_cfo_autopilot_network_is_external(self):
        compose = _load_compose()
        network = compose["networks"]["cfo_autopilot_default"]
        assert network.get("external") is True

    def test_internal_network_is_internal(self):
        compose = _load_compose()
        network = compose["networks"]["llm_router_internal"]
        assert network.get("internal") is True

    def test_both_services_have_healthcheck(self):
        compose = _load_compose()
        assert "healthcheck" in compose["services"]["llm-router"]
        assert "healthcheck" in compose["services"]["llm-router-db"]

    def test_postgres_has_named_volume(self):
        compose = _load_compose()
        assert "llm_router_pgdata" in compose.get("volumes", {})
        db_volumes = compose["services"]["llm-router-db"]["volumes"]
        assert any("llm_router_pgdata" in v for v in db_volumes)
