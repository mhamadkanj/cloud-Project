# Carbon-aware scheduler API
# Build from the REPOSITORY ROOT (it needs model/ and api/):
#   docker build -f api/Dockerfile -t carbon-api .

# ---------- Stage 1: install the Python packages ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY api/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && find /install -type d -name tests -prune -exec rm -rf {} + \
 && find /install -type d -name __pycache__ -prune -exec rm -rf {} +

# ---------- Stage 2: the small image that actually runs ----------
FROM python:3.12-slim
# CI passes the commit SHA here, so GET /version shows which commit is running
ARG API_VERSION=dev
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model/model-1.0.0.joblib \
    API_VERSION=${API_VERSION}
WORKDIR /app

# only the installed packages come from stage 1, no build leftovers
COPY --from=builder /install /usr/local
# shared feature code + trained model + API code
COPY model/features.py ./features.py
COPY model/artifacts/model-1.0.0.joblib ./model/model-1.0.0.joblib
COPY api/app ./app

# security: do not run as root
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]