FROM ghcr.io/astral-sh/uv:python3.11-slim

# Use /app as the home base
WORKDIR /app

# 1. Install dependencies first to leverage Docker layer caching
# This uses bind mounts so the toml/lock files aren't permanently in the layer yet
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# 2. Copy the entire src directory into /app/src
COPY src/ /app/src/
COPY pyproject.toml uv.lock ./

# 3. Use 'uv run' to execute. 
# This automatically handles the virtualenv and ensures PYTHONPATH includes /app/src
# if your pyproject.toml is configured correctly.
CMD ["uv", "run", "python", "-m", "ingestion.fred_job"]