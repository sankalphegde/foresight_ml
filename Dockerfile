FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app/src

COPY pyproject.toml uv.lock ./
RUN uv venv /app/.venv && uv sync --frozen --no-dev --no-install-project

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY src/ /app/src/

# Critical
ENV PYTHONPATH=/app/src

# Safe default (can be overridden by job)
CMD ["python", "-m", "ingestion.fred_job"]
