FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY server ./server
RUN mkdir -p /app/server/cache

WORKDIR /app/server

EXPOSE 3030

CMD ["uv", "run", "python", "app.py"]
