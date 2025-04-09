# Stage 0: Base image
FROM python:3.13-slim AS base

# Install minimal system dependencies, UV, and Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    git \
    swig \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    sqlite3 \
    jq \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    # Install Node.js and npm
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Install UV package manager
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Setup Docker group for Docker-in-Docker support with MCP
# Ensure the docker group exists (GID might vary, Docker typically uses 999 or similar)
# We don't need to add the user since we'll run as root.
RUN groupadd -r docker || getent group docker || groupadd -g 999 docker

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_CACHE_DIR=/app/.uv-cache

# Stage 1: Builder stage for backend
FROM base AS backend-builder

# Set working directory
WORKDIR /app

# Copy backend requirements
COPY backend/pyproject.toml backend/requirements.txt /app/backend/

# Install Python dependencies using UV
RUN cd /app/backend && \
    uv venv /app/.venv && \
    uv pip install -e . && \
    ln -s /app/.venv/bin/python /usr/local/bin/python && \
    ln -s /app/.venv/bin/uvicorn /usr/local/bin/uvicorn

# Stage 2: Builder stage for frontend
FROM base AS frontend-builder

# Set working directory
WORKDIR /app/frontend


# Copy frontend code (will respect .dockerignore for node_modules)
WORKDIR /app
# Copy frontend source code (selective copy for better caching)
COPY frontend/.prettierrc frontend/components.json frontend/next-env.d.ts \
     frontend/next.config.js frontend/package.json frontend/pnpm-lock.yaml \
     frontend/postcss.config.js frontend/tailwind.config.js frontend/tsconfig.json \
     /app/frontend/
COPY frontend/components /app/frontend/components/
COPY frontend/contexts /app/frontend/contexts/
COPY frontend/hooks /app/frontend/hooks/
COPY frontend/pages /app/frontend/pages/
COPY frontend/public /app/frontend/public/
COPY frontend/services /app/frontend/services/
COPY frontend/styles /app/frontend/styles/
COPY frontend/types /app/frontend/types/
COPY frontend/utils /app/frontend/utils/
# Install frontend dependencies
RUN cd /app/frontend && pnpm install --frozen-lockfile

# Build frontend for production
RUN cd /app/frontend && NODE_ENV=production pnpm run build
# Build frontend for production
RUN cd /app/frontend && NODE_ENV=production pnpm run build

# Stage 3: Development stage
FROM base AS development

# Install Node.js using nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && \
    . "$HOME/.nvm/nvm.sh" && \
    nvm install 22 && \
    npm install -g pnpm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/backend \
             /app/frontend \
             /app/frontend/node_modules \
             /app/frontend/.next \
             /app/.pnpm-store \
             /app/backend/tests

# Create virtual environment
RUN uv venv /app/.venv

# Copy entrypoint scripts
COPY entrypoint.sh /app/
COPY entrypoint.prod.sh /app/
RUN chmod +x /app/entrypoint.sh && \
    chmod +x /app/entrypoint.prod.sh

# Copy backend source code
COPY backend /app/backend/

# Copy frontend source code (selective copy for better caching)
COPY frontend/.prettierrc frontend/components.json frontend/next-env.d.ts \
     frontend/next.config.js frontend/package.json frontend/pnpm-lock.yaml \
     frontend/postcss.config.js frontend/tailwind.config.js frontend/tsconfig.json \
     /app/frontend/
COPY frontend/components /app/frontend/components/
COPY frontend/contexts /app/frontend/contexts/
COPY frontend/hooks /app/frontend/hooks/
COPY frontend/pages /app/frontend/pages/
COPY frontend/public /app/frontend/public/
COPY frontend/services /app/frontend/services/
COPY frontend/styles /app/frontend/styles/
COPY frontend/types /app/frontend/types/
COPY frontend/utils /app/frontend/utils/

# Ownership is handled by running as root, no chown needed

# Install development tools with UV
RUN cd /app/backend && \
    uv pip install -e ".[dev]"

# Create config files
RUN echo 'exclude_dirs: ["/venv", "/tests"]' > /app/backend/bandit.yaml

# Environment variables for development
ENV NODE_ENV=development \
    FASTAPI_ENV=development \
    PNPM_HOME=".local/share/pnpm" \
    PATH="/app/.venv/bin:${PATH}" \
    MCP_NETWORK=mcp-network \
    MCP_DATA_DIR=/var/lib/doogie-chat/mcp \
    MCP_ENABLE_DOCKER=true

# Add pnpm to PATH
ENV PATH="${PNPM_HOME}:${PATH}"

# Development-specific command
CMD ["/app/entrypoint.sh"]

# Stage 4: Production stage
FROM base AS production

# Build arguments for metadata
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG GITHUB_REPOSITORY=toosmooth/doogiebot

# Add metadata
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="Mike Reeves" \
      org.opencontainers.image.url="https://github.com/${GITHUB_REPOSITORY}" \
      org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.vendor="TOoSmOotH" \
      org.opencontainers.image.title="DoogieBot" \
      org.opencontainers.image.description="DoogieBot - An intelligent chat platform"

# Set working directory
WORKDIR /app

# Install Node.js using nvm
RUN apt-get update && apt-get install -y curl && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    nvm install 22 && \
    nvm use 22 && \
    npm install -g pnpm && \
    echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc && \
    echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.bashrc && \
    echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> ~/.bashrc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add node and npm to PATH
ENV PATH="/root/.nvm/versions/node/v22.0.0/bin:${PATH}"

# Create virtual environment
RUN uv venv /app/.venv && \
    mkdir -p /app/backend

# Copy backend code
COPY backend/pyproject.toml /app/backend/
COPY backend/app /app/backend/app
COPY backend/main.py /app/backend/main.py
COPY backend/alembic.ini /app/backend/alembic.ini
COPY backend/alembic /app/backend/alembic

# Install backend dependencies
RUN cd /app/backend && \
    uv pip install -e .

# Copy frontend build from builder
COPY --from=frontend-builder /app/frontend/.next /app/frontend/.next
COPY --from=frontend-builder /app/frontend/public /app/frontend/public
COPY --from=frontend-builder /app/frontend/node_modules /app/frontend/node_modules
COPY frontend/package.json frontend/next.config.js /app/frontend/

# Copy entrypoint script
COPY entrypoint.prod.sh /app/
RUN chmod +x /app/entrypoint.prod.sh

# Ownership is handled by running as root, no chown needed

# Environment variables for production
ENV NODE_ENV=production \
    FASTAPI_ENV=production \
    PATH="/app/.venv/bin:${PATH}"

# Health check
HEALTHCHECK --interval=5m --timeout=3s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose ports
EXPOSE 3000 8000


# Run as root to allow Docker socket access
# USER ${USER_ID}:${GROUP_ID} # Removed to run as root

# Run the application
ENTRYPOINT ["/app/entrypoint.prod.sh"]