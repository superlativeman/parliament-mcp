FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies via Poetry
RUN poetry config virtualenvs.create false \
  && poetry install --no-root --no-interaction --no-ansi

EXPOSE 8080

CMD ["python3", "-m", "parliament_mcp.mcp_server.main"]
