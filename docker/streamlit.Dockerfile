FROM ghcr.io/astral-sh/uv:X.Y.Z AS uv

FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --only-group streamlit
COPY streamlit_app/ streamlit_app/
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
