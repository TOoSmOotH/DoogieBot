# Stage 0: Base image
FROM python:3.13-slim AS base

# User configuration with defaults
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG DOCKER_GID=999 # Use 999 as a common default, pass from compose if different
ARG USER_NAME=appuser

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
    # Docker CLI install moved to specific stages (dev/prod) where needed
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Install UV package manager
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Create non-root user with configurable UID/GID (passed from docker-compose)
RUN groupadd -g ${GROUP_ID} ${USER_NAME} || true && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash -m ${USER_NAME} && \
    mkdir -p /app && \
    chown -R ${USER_ID}:${GROUP_ID} /app

# --- MODIFIED Docker Group Setup ---
# Create a group with the specific GID passed from the build argument
# This group needs to match the GID of the host's docker group to access the socket
RUN DOCKER_GROUP_NAME=$(getent group ${DOCKER_GID} | cut -d: -f1) && \
    if [ -n "$DOCKER_GROUP_NAME" ]; then \
        echo "Adding user ${USER_NAME} to existing group ${DOCKER_GROUP_NAME} (GID: ${DOCKER_GID})" && \
        usermod -aG ${DOCKER_GROUP_NAME} ${USER_NAME}; \
    else \
        echo "Creating group 'docker' with GID ${DOCKER_GID} and adding user ${USER_NAME}" && \
        groupadd -g ${DOCKER_GID} docker && \
        usermod -aG docker ${USER_NAME}; \
    fi
# --- END MODIFIED Docker Group Setup ---

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
ENV UV_HTTP_TIMEOUT=300
RUN cd /app/backend && \
    uv venv /app/.venv && \
    uv pip install -e . && \
    ln -s /app/.venv/bin/python /usr/local/bin/python && \
    ln -s /app/.venv/bin/uvicorn /usr/local/bin/uvicorn

# Stage 2: Builder stage for frontend
FROM base AS frontend-builder

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g pnpm

# Set working directory
WORKDIR /app/frontend

# Copy package files first for cache optimization
COPY frontend/package.json frontend/pnpm-lock.yaml ./

# Install frontend dependencies
RUN pnpm install --frozen-lockfile # Use --frozen-lockfile for reliability

# Copy the rest of the frontend code AFTER installing dependencies
COPY frontend/next.config.js ./
COPY frontend/postcss.config.js ./
COPY frontend/tailwind.config.js ./
COPY frontend/tsconfig.json ./
COPY frontend/next-env.d.ts ./
COPY frontend/.prettierrc ./
COPY frontend/components ./components
COPY frontend/contexts ./contexts
COPY frontend/hooks ./hooks
COPY frontend/pages ./pages
COPY frontend/public ./public
COPY frontend/services ./services
COPY frontend/styles ./styles
COPY frontend/types ./types
COPY frontend/utils ./utils

# Build frontend for production
RUN cd /app/frontend && NODE_ENV=production pnpm run build

# Stage 3: Test stage
FROM backend-builder AS test

# Install test dependencies with UV
RUN cd /app/backend && \
    uv pip install -e ".[test]" && \
    uv pip install black pylint mypy

# Copy backend code and test files
COPY backend/ /app/backend/
COPY tests/ /app/tests/

# Set test entrypoint
ENTRYPOINT ["uv", "run", "pytest"]
CMD ["backend/tests"]

# Stage 4: Development stage
FROM base AS development

# Install Node.js and Docker CLI (needed for MCP)
RUN apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y nodejs docker-ce-cli && \
    # pnpm will be installed by entrypoint.sh if needed
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories early
RUN mkdir -p /app/backend \
             /app/frontend \
             /app/frontend/node_modules \
             /app/frontend/.next \
             /app/.pnpm-store \
             /app/backend/tests

# Create virtual environment
RUN uv venv /app/.venv

# Copy backend dependency files first
COPY backend/pyproject.toml backend/requirements.txt backend/uv.lock* /app/backend/
# Install backend dev dependencies BEFORE copying all source code
ENV UV_HTTP_TIMEOUT=300
RUN cd /app/backend && \
    uv pip install -e ".[dev]" && \
    echo "--- Contents of /app/.venv/bin after dev install ---" && \
    ls -la /app/.venv/bin && \
    echo "--- End of /app/.venv/bin contents ---"

# Copy frontend dependency files
COPY frontend/package.json frontend/pnpm-lock.yaml /app/frontend/

# Copy entrypoint scripts
COPY entrypoint.sh /app/
COPY entrypoint.prod.sh /app/
# Removed chmod +x, Docker handles execution via ENTRYPOINT/CMD

# Copy the rest of the backend source code
COPY backend /app/backend/

# Copy the rest of the frontend source code (granular copy)
COPY frontend/.prettierrc /app/frontend/
COPY frontend/components.json /app/frontend/
COPY frontend/next-env.d.ts /app/frontend/
COPY frontend/next.config.js /app/frontend/
COPY frontend/package.json /app/frontend/
COPY frontend/pnpm-lock.yaml /app/frontend/
COPY frontend/postcss.config.js /app/frontend/
COPY frontend/tailwind.config.js /app/frontend/
COPY frontend/tsconfig.json /app/frontend/
COPY frontend/components /app/frontend/components/
COPY frontend/contexts /app/frontend/contexts/
COPY frontend/hooks /app/frontend/hooks/
COPY frontend/pages /app/frontend/pages/
COPY frontend/public /app/frontend/public/
COPY frontend/services /app/frontend/services/
COPY frontend/styles /app/frontend/styles/
COPY frontend/types /app/frontend/types/
COPY frontend/utils /app/frontend/utils/

# Set ownership for the entire app directory before installing dependencies
# This needs to happen *after* user/group are created with correct IDs
RUN chown -R ${USER_ID}:${GROUP_ID} /app
 # Backend dev dependencies already installed above for better caching

 # Frontend dependencies will be installed by entrypoint.sh for dev consistency with bind mounts
 # Install frontend dependencies (using pnpm install, should use lock file)
# No need to run this here if using bind mounts for dev, but good for standalone image
# RUN cd /app/frontend && pnpm install

# Create config files (after backend code is copied)
RUN echo 'exclude_dirs: ["/venv", "/tests"]' > /app/backend/bandit.yaml

# Environment variables for development
ENV NODE_ENV=development \
    FASTAPI_ENV=development \
    # PNPM_HOME removed, using standard install path below
    PATH="/app/.venv/bin:${PATH}" \
    # MCP configuration
    MCP_NETWORK=mcp-network \
    MCP_DATA_DIR=/var/lib/doogie-chat/mcp \
    MCP_ENABLE_DOCKER=true

# Add pnpm to PATH
# pnpm should be globally available in PATH from the curl install, no need for user-specific path mod

# Ensure UV cache directory exists and has correct permissions BEFORE switching user
RUN mkdir -p /app/.uv-cache && chown -R ${USER_ID}:${GROUP_ID} /app/.uv-cache

# Switch to configured user AFTER all root operations (installations, chown) are done
USER ${USER_ID}:${GROUP_ID}

# Development-specific command
CMD ["/app/entrypoint.sh"]

# Stage 5: Production stage
FROM base AS production

# Build arguments for metadata
ARG BUILD_DATE
ARG VERSION=1.0.0

# Add metadata with correct author information
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="Mike Reeves" \
      org.opencontainers.image.url="https://github.com/TOoSmOotH/DoogieBot" \
      org.opencontainers.image.source="https://github.com/TOoSmOotH/DoogieBot" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.vendor="TOoSmOotH" \
      org.opencontainers.image.title="DoogieBot" \
      org.opencontainers.image.description="DoogieBot"

# Set working directory
WORKDIR /app

# Install Node.js, pnpm, and Docker CLI (needed for MCP)
RUN apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y nodejs docker-ce-cli && \
    curl -fsSL https://get.pnpm.io/install.sh | SHELL=/bin/bash sh - && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment and install dependencies with UV
RUN uv venv /app/.venv && \
    mkdir -p /app/backend /app/frontend

# Copy backend dependency files first
COPY backend/pyproject.toml backend/requirements.txt backend/uv.lock* /app/backend/
# Install backend production dependencies (no dev extras)
# Copy necessary backend source code for runtime BEFORE installing dependencies
COPY backend/app /app/backend/app
# Install backend production dependencies (no dev extras)
ENV UV_HTTP_TIMEOUT=300
RUN cd /app/backend && \
    uv pip install -e .

# Copy frontend build artifacts from builder stage
COPY --from=frontend-builder /app/frontend/.next /app/frontend/.next
COPY --from=frontend-builder /app/frontend/public /app/frontend/public
COPY --from=frontend-builder /app/frontend/node_modules /app/frontend/node_modules
# Copy necessary frontend config files for runtime
COPY frontend/package.json frontend/next.config.js /app/frontend/

# (Backend source code copied earlier)
COPY backend/main.py /app/backend/main.py
COPY backend/alembic.ini /app/backend/alembic.ini
COPY backend/alembic /app/backend/alembic
# Add any other necessary top-level files/dirs from backend/ here if needed

# Set ownership for backend AFTER copying source
RUN chown -R ${USER_ID}:${GROUP_ID} /app/backend

# Copy unified entrypoint script
COPY entrypoint.sh /app/
RUN chown ${USER_ID}:${GROUP_ID} /app/entrypoint.sh
# Removed chmod +x, Docker handles execution via ENTRYPOINT/CMD

# Ensure frontend build files have correct ownership
RUN chown -R ${USER_ID}:${GROUP_ID} /app/frontend/.next && \
    chown -R ${USER_ID}:${GROUP_ID} /app/frontend/public && \
    chown -R ${USER_ID}:${GROUP_ID} /app/frontend/node_modules

# Environment variables for production
ENV NODE_ENV=production \
    FASTAPI_ENV=production \
    PATH="/app/.venv/bin:${PATH}"

# Health check
HEALTHCHECK --interval=5m --timeout=3s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose ports
EXPOSE 3000 8000

# Switch to configured user
USER ${USER_ID}:${GROUP_ID}

# Run the application using the unified entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
