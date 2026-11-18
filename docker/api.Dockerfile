FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf PROJECT_ROOT=/app \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Exact versions from uv.lock; CPU-only torch comes from the PyTorch index set in pyproject.toml
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --only-group api

# Bake the extractor weights into the image: no download at start-up, reproducible deploys
ARG EXTRACTOR=mobilenetv3_large_100
RUN python -c "import timm; timm.create_model('${EXTRACTOR}', pretrained=True)" && chmod -R a+rX /opt/hf

COPY params.yaml params.yaml
COPY src/ src/
RUN useradd -m app && mkdir -p /app/data && chown app /app/data
USER app

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
