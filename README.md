# llm-router

Единый LLM-роутер поверх [RouterAI](https://routerai.ru) для нескольких MVP. В v1 подключён
только `cfo-autopilot`. Маршрутизация модели — по `task_type`, без возможности для клиента
передать `model`/`provider`/`url` напрямую.

## Архитектура

```
cfo-autopilot backend --(POST /v1/complete + X-Internal-Secret)--> llm-router --(AsyncOpenAI)--> RouterAI
                                                                        |
                                                                        v
                                                                  Postgres (request_log)
```

## Сети (docker-compose.yml)

- `llm-router` (app) подключён к **двум** сетям:
  - `cfo_autopilot_default` (external, уже существует на VPS) — чтобы cfo-autopilot backend
    мог обращаться по имени контейнера `http://llm-router:8000`;
  - `llm_router_internal` (создаётся этим compose-файлом, `internal: true`) — для связи с
    собственным Postgres.
- `llm-router-db` (Postgres) подключён **только** к `llm_router_internal`. Без `ports:` —
  недоступен ни с хоста, ни из `cfo_autopilot_default`, только из контейнера `llm-router` по
  DNS-имени `llm-router-db`.
  > **Важно:** сервис намеренно называется `llm-router-db`, а не просто `db`. На VPS уже
  > крутится `cfo_autopilot-db-1` (сервис `db` в `docker-compose.prod.yml` cfo-autopilot) в
  > сети `cfo_autopilot_default`. Так как `llm-router` app подключён к этой же сети, короткое
  > имя `db` резолвилось бы в ЧУЖОЙ Postgres cfo-autopilot (embedded Docker DNS не изолирует
  > алиасы по compose-проекту в рамках одной сети) — было поймано на этапе деплоя как
  > `asyncpg.exceptions.InvalidPasswordError` (роли `llm_router` в чужой БД просто нет).
  > При подключении следующих MVP (grill, maxima) к общей сети используйте такие же
  > project-scoped имена сервисов, а не generic `db`/`redis`/`app`.
- `llm-router` не публикует порт на хост — только `expose: 8000`. Проверка health и любые
  ручные запросы делаются изнутри docker-сети:
  ```bash
  docker exec cfo_autopilot-backend-1 python3 -c \
    "import urllib.request; print(urllib.request.urlopen('http://llm-router:8000/health').read())"
  # или из любого контейнера в сети cfo_autopilot_default:
  docker run --rm --network cfo_autopilot_default curlimages/curl -f http://llm-router:8000/health
  ```

## Запуск

```bash
cp .env.example .env
# заполнить ROUTERAI_API_KEY, INTERNAL_SECRET (openssl rand -hex 32), POSTGRES_PASSWORD

# сеть cfo_autopilot_default должна существовать заранее (создаётся стеком cfo-autopilot)
docker compose up -d --build
docker compose ps   # оба сервиса должны быть healthy
```

## Backup / restore Postgres

```bash
# backup
docker exec llm-router-db-1 pg_dump -U llm_router llm_router | gzip > backup_$(date +%F).sql.gz

# restore
gunzip -c backup_2026-09-01.sql.gz | docker exec -i llm-router-db-1 psql -U llm_router llm_router
```

## Тесты

```bash
poetry install
poetry run pytest
```

## Явно вне scope v1

- Интеграция grill и maxima consulting (task_type `code`/`chat` зарезервированы, не реализованы).
- Независимый аудит ответов моделей (risk-based pass/revise/fail).
- Бюджетные лимиты / автоотключение по расходу.
- Динамическое обнаружение цен через `/models/{author}/{slug}/endpoints` — не нужно, RouterAI
  возвращает точную стоимость в `usage.cost` каждого ответа.
