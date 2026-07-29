FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY phase3_monitoring/inference_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY phase3_monitoring/inference_service /app/phase3_monitoring/inference_service/

# We need the source code from Phase 1/2 for graph building and model
COPY src /app/src
COPY config.yaml /app/config.yaml
COPY artifacts /app/artifacts

# Set Python Path
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "phase3_monitoring.inference_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
