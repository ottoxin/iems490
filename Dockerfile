# -------- Base image --------
FROM python:3.11-slim

# -------- Env & OS deps --------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

# -------- Create non-root user --------
RUN useradd -m appuser
WORKDIR /app

# -------- Python deps --------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------- Project files --------
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY data/ ./data/

# Ensure outputs dir exists & fix ownership
RUN mkdir -p outputs && chown -R appuser:appuser /app
USER appuser

# -------- Default command --------
# Leave it interactive by default; you can override CMD at runtime.
CMD ["bash"]