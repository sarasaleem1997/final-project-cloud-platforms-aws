# Base image: slim Python 3.13
FROM python:3.13-slim

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory inside the container
WORKDIR /app

# Copy dependency files first (separate layer for caching)
# If only app code changes, Docker reuses the cached dependency layer
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (no dev dependencies, no editable install)
RUN uv sync --frozen --no-dev --no-install-project

# Copy only what's needed to run the app
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/gold/ ./data/gold/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit on all interfaces so it's accessible outside the container
CMD [".venv/bin/streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
