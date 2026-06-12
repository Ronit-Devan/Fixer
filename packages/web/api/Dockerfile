FROM python:3.11-slim

# Hugging Face Spaces expects the container to run on port 7860
ENV PORT=7860

# Hugging Face mounts /data with writable space; everything else is read-only
# We put the app in /app which is writable during build
WORKDIR /app

# Copy engine and API from the monorepo
COPY packages/engine /app/packages/engine
COPY packages/web/api /app/packages/web/api

WORKDIR /app/packages/web/api

# Install uv and dependencies
RUN pip install --no-cache-dir uv
RUN uv sync --frozen

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "et_api.server:app", "--host", "0.0.0.0", "--port", "7860"]
