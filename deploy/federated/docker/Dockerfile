# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

# Keep the application image CPU-only. SuperLink and SuperNode remain the
# upstream, version-pinned Flower infrastructure images.
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.6.0" \
    && python -m pip install --no-cache-dir \
      "flwr==1.32.1" \
      "numpy>=1.26,<2.0" \
      "pandas>=2.0" \
      "matplotlib>=3.7" \
      "scikit-learn>=1.3" \
      "pyyaml>=6.0" \
      "torch-geometric>=2.5" \
    && python -m pip install --no-cache-dir --no-deps . \
    && python -m compileall -q src \
    && useradd --uid 49999 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app

USER app
ENTRYPOINT ["flower-superexec"]
