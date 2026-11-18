FROM ghcr.io/astral-sh/uv:X.Y.Z AS uv

FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --only-group tracking
EXPOSE 5000
