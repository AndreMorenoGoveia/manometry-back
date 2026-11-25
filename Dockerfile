FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is used by the CPU builds of PyTorch/scikit-learn. Install the
# dependencies before copying the application so source-only changes keep the
# expensive wheel layer cached by Cloud Build.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-cpu.txt

COPY app ./app
COPY assets/model ./assets/model
COPY assets/reference_manifest.json ./assets/reference_manifest.json

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run sends traffic to PORT=8080 by default. One worker is intentional:
# every worker would hold another copy of the model in memory.
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
