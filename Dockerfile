FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/app/.cache/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY packages/ packages/
RUN pip install -U pip setuptools wheel && \
    for pkg in pcgrl-core pcgrl-runtime pcgrl-rendering pcgrl-bridge pcgrl-learning pcgrl-app; do \
        pip install -e "packages/$pkg"; \
    done && \
    pip install -e ".[dev]"

COPY . .
RUN bash scripts/bridge.sh

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
