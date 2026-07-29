FROM python:3.11-slim

WORKDIR /app

COPY phase3_monitoring/flow_feeder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY phase3_monitoring/flow_feeder /app/phase3_monitoring/flow_feeder/
ENV PYTHONPATH=/app

CMD ["python", "phase3_monitoring/flow_feeder/replay_flows.py"]
