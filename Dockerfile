FROM python:3.12-slim

WORKDIR /app

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy application code
COPY pyproject.toml .
COPY server.py .

# Expose the default MCP port
EXPOSE 8081

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8081/healthcheck || exit 1

# Run the server with SSE transport
CMD ["uv", "run", "--with", "mcp[cli]", "--with", "sympy", "mcp", "run", "/app/server.py", "--transport", "sse"] 