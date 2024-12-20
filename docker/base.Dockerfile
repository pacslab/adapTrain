FROM python:3.10-slim

WORKDIR /app

COPY ../requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY ../src/configurations /app/src/configurations
COPY ../src/exceptions /app/src/exceptions
COPY ../src/models /app/src/models
COPY ../src/utils /app/src/utils
COPY ../src/dataset /app/src/dataset
COPY ../src/logger /app/src/logger
COPY ../src/__init__.py /app/src/

ENV PYTHONPATH="/app/src"