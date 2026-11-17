FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.22.0 "sqlalchemy>=2.0,<2.1" psycopg2-binary==2.9.13
EXPOSE 5000
