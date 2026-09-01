FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry==1.8.0 && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY . .

EXPOSE 8000
