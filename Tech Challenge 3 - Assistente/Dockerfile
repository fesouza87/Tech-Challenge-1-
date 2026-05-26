FROM python:3.11-slim AS base

WORKDIR /app

COPY requirements.runtime.txt /app/requirements.runtime.txt
COPY requirements.txt /app/requirements.txt

COPY backend /app/backend
COPY scripts /app/scripts
COPY data /app/data

ENV TC3_DATA_DIR=/app/data
ENV TC3_PROTOCOL_DIR=/app/data/protocols
ENV TC3_VECTORSTORE_DIR=/app/data/vectorstore
ENV TC3_PATIENT_DB=/app/data/patients.db
ENV TC3_AUDIT_LOG=/app/logs/audit.jsonl

WORKDIR /app/backend
EXPOSE 8000

FROM base AS runtime
RUN pip install --no-cache-dir -r /app/requirements.runtime.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS hf
RUN pip install --no-cache-dir -r /app/requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
