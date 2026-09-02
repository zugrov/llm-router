# llm-router

Единый LLM-роутер поверх [RouterAI](https://routerai.ru) для нескольких MVP: `cfo-autopilot`,
`grill`, `maxima-consulting`. Маршрутизация модели — по `task_type`, без возможности для клиента
передать `model`/`provider`/`url` напрямую.

## Архитектура

```
cfo-autopilot backend --(POST /v1/complete        + X-Internal-Secret)--> llm-router --(AsyncOpenAI)--> RouterAI
grill (Docker)        --(POST /v1/complete/stream  + X-Internal-Secret)-->     |
consulting-agent (systemd, хост) --(тот же путь через 127.0.0.1:8020)-->       |
                                                                                v
                                                                          Postgres (request_log)
```

`POST /v1/complete` — одноразовый ответ целиком (cfo-autopilot: `extraction`, `financial_analysis`).
`POST /v1/complete/stream` — тот же контракт запроса, ответ по SSE построчно
(`data: {"delta": "..."}`, финал — `{"done": true, ...}` либо `{"error": true, "error_code": ...}`).
Нужен grill (`chat`, многоходовой диалог через `messages`) и maxima consulting (`client_report`).
Retry/fallback на другую модель в стриме возможен только пока клиенту не отдан ни один chunk —
дальше при сбое поток обрывается событием ошибки (см. `app/upstream.py`).

## Сети (docker-compose.yml)

- `llm-router` (app) подключён к **трём** сетям:
  - `cfo_autopilot_default` (external) — cfo-autopilot backend обращается по
    `http://llm-router:8000`;
  - `grill-ideas_default` (external) — grill обращается по тому же имени `llm-router:8000`;
  - `llm_router_internal` (создаётся этим compose-файлом, `internal: true`) — для связи с
    собственным Postgres.
- `llm-router-db` (Postgres) подключён **только** к `llm_router_internal`. Без `ports:` —
  недоступен ни с хоста, ни из внешних сетей, только из контейнера `llm-router` по DNS-имени
  `llm-router-db`.
  > **Важно:** сервис намеренно называется `llm-router-db`, а не просто `db`. На VPS уже
  > крутится `cfo_autopilot-db-1` (сервис `db` в `docker-compose.prod.yml` cfo-autopilot) в
  > сети `cfo_autopilot_default`. Так как `llm-router` app подключён к этой же сети, короткое
  > имя `db` резолвилось бы в ЧУЖОЙ Postgres cfo-autopilot (embedded Docker DNS не изолирует
  > алиасы по compose-проекту в рамках одной сети) — было поймано на этапе деплоя как
  > `asyncpg.exceptions.InvalidPasswordError` (роли `llm_router` в чужой БД просто нет).
  > При подключении следующих MVP к общей сети используйте такие же project-scoped имена
  > сервисов, а не generic `db`/`redis`/`app`.
- `llm-router` публикует `127.0.0.1:8020:8000` — это единственный способ достучаться из
  `consulting-agent` (systemd-процесс на хосте VPS, не в Docker). Наружу (`0.0.0.0`) порт не
  публикуется — за это отвечает allowlist (`ALLOWED_PROJECTS`) + `X-Internal-Secret`, а не сеть.
  Проверка health и ручные запросы:
  ```bash
  curl -f http://127.0.0.1:8020/health
  # или изнутри docker-сети:
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

## Явно вне scope

- `task_type: code` зарезервирован, не реализован — нет реального потребителя.
- Независимый аудит ответов моделей (risk-based pass/revise/fail).
- Бюджетные лимиты / автоотключение по расходу.
- Динамическое обнаружение цен через `/models/{author}/{slug}/endpoints` — не нужно, RouterAI
  возвращает точную стоимость в `usage.cost` каждого ответа.
