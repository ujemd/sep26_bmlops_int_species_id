FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf PROJECT_ROOT=/app
WORKDIR /app

# CPU-only torch: ~200 MB instead of ~2.5 GB for the default CUDA build
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements/api.txt requirements/api.txt
RUN pip install -r requirements/api.txt

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
