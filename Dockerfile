FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY arxiv_classifier ./arxiv_classifier
COPY configs ./configs
COPY train_artifacts ./train_artifacts 2>/dev/null || true

# Expose API port
EXPOSE 8000

# Run API server by default
CMD ["python", "-m", "arxiv_classifier.commands", "serve"]
