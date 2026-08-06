# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs/application ./configs/application

RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.6.0" \
    && python -m pip install --no-cache-dir \
      "numpy>=1.26,<2.0" \
      "pandas>=2.0" \
      "PyYAML>=6.0" \
      "scikit-learn>=1.3" \
      "torch-geometric>=2.5" \
      "fastapi>=0.110" \
      "pydantic>=2.7" \
      "uvicorn>=0.30" \
    && python -m pip install --no-cache-dir --no-deps . \
    && python -m compileall -q src \
    && useradd --uid 49999 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app

USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn"]
CMD ["src.application.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
