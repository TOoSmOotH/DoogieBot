#!/bin/bash
set -e

# Ensure required directories exist
ensure_directories() {
    echo "Ensuring required directories exist..."
    # Use the persistent data directory for db and indexes
    mkdir -p /app/data/db
    mkdir -p /app/data/indexes
    mkdir -p /app/backend/uploads # Uploads can stay within backend for now

    # Permissions should be handled by Dockerfile chown and user/group mapping
    # Removed chmod 777/755 calls

    # Create a new UV cache directory (permissions handled by user context)
    mkdir -p /tmp/uv-cache-new

    # Set UV to use our new cache directory
    export UV_CACHE_DIR=/tmp/uv-cache-new

    echo "Directories checked and created if needed."
    echo "UV cache directory set to: $UV_CACHE_DIR"
}

# Function to check Docker socket access (simplified)
check_docker_permissions() {
    echo "Checking Docker socket permissions..."
    DOCKER_SOCKET=/var/run/docker.sock
    if [ -S "$DOCKER_SOCKET" ]; then
        if [ -w "$DOCKER_SOCKET" ]; then
            echo "User ($(id -u):$(id -g)) appears to have write access to the Docker socket."
            # Verify Docker command execution
            echo "Verifying Docker command execution..."
            if docker ps &>/dev/null; then
                echo "Docker command execution verified successfully."
            else
                echo "WARNING: Docker socket is writable but 'docker ps' command failed."
                echo "This may indicate issues with Docker daemon connectivity or configuration."
            fi
        else
            echo "WARNING: User ($(id -u):$(id -g)) may not have write access to the Docker socket ($DOCKER_SOCKET)."
            echo "Ensure the container user's group ID matches the Docker socket's group ID on the host."
            ls -l $DOCKER_SOCKET
        fi
    else
        echo "Docker socket $DOCKER_SOCKET not found. Skipping permission check."
    fi
}


# Run database migrations with retry logic
run_migrations() {
    echo "Running database migrations..."
    cd /app/backend # Ensure we are in the correct directory for alembic.ini

    # Ensure the database file exists in the persistent location
    DB_FILE="/app/data/db/doogie.db"
    mkdir -p "$(dirname "$DB_FILE")"
    # Create file if it doesn't exist, permissions handled by user context
    touch "$DB_FILE"
    # Removed chmod 666
    echo "Ensured database file exists at $DB_FILE"

    # Ensure UV environment variable is exported here too
    export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-new}
    echo "Using UV cache directory: $UV_CACHE_DIR for migrations"

    # Apply all migrations based on files in alembic/versions
    echo "Applying database migrations..."
    local max_attempts=5
    local attempt=1
    local success=false

    # Removed chown attempt as venv is now outside the mounted directory

    # Activate the virtual environment from the new location
    VENV_PATH="/app/.venv"
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "Activating virtual environment at $VENV_PATH..."
        source "$VENV_PATH/bin/activate"
    else
        echo "ERROR: Virtual environment activation script not found at $VENV_PATH/bin/activate. Exiting."
        exit 1
    fi

    while [ $attempt -le $max_attempts ] && [ "$success" = false ]; do
        echo "Attempt $attempt of $max_attempts to apply migrations..."
        # Run upgrade head
        # Run alembic directly now that the venv is activated
        if alembic upgrade head; then
            success=true
            echo "Database migrations applied successfully."
        else
            echo "Migration attempt $attempt failed. Waiting before retry..."
            # Print UV cache directory info for debugging
            echo "UV cache directory contents:"
            ls -la $UV_CACHE_DIR || echo "Cannot list UV cache directory"
            sleep 5
            attempt=$((attempt+1))
        fi
    done

    if [ "$success" = false ]; then
        echo "ERROR: Failed to apply migrations after $max_attempts attempts. Exiting."
        exit 1 # Exit if migrations fail
    fi

    # Verification step is less critical now as autogenerate + upgrade should handle it
    # but we can keep a basic check
    echo "Verifying database connection..."
    if [ -f "$DB_FILE" ]; then
        # Run a simple SQL query to verify connection
        if sqlite3 "$DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='users';" | grep -q "users"; then
            echo "Database verification successful (users table found)."
        else
            echo "WARNING: Users table not found after migrations. Check Alembic configuration and model definitions."
        fi
    else
        echo "ERROR: Database file $DB_FILE not found after migration attempt."
    fi
}

# Function to start the backend (handles dev/prod)
start_backend() {
    echo "Starting backend server in $FASTAPI_ENV mode..."
    cd /app/backend || { echo "ERROR: Failed to cd to /app/backend"; exit 1; }

    # Ensure UV environment variable is set here too
    export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-new}
    echo "Using UV cache directory: $UV_CACHE_DIR for backend"

    # Activate the virtual environment from the new location
    VENV_PATH="/app/.venv"
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "Activating virtual environment at $VENV_PATH..."
         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate venv"; exit 1; }
    else
        echo "ERROR: Virtual environment activation script not found at $VENV_PATH/bin/activate. Exiting."
        exit 1
    fi

    if [ "$FASTAPI_ENV" = "production" ]; then
        # Production: Use multiple workers, no reload
        echo "Starting production backend..."
        uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --timeout-keep-alive 300 &
        BACKEND_PID=$!
        echo "Production backend server started with PID: $BACKEND_PID"
    else
        # Development: Use single worker with reload, run foreground
        echo ">>> Attempting to start development backend..." # ADDED
        export PYTHONMALLOC=debug
        export PYTHONWARNINGS=always
        echo ">>> Checking python version..." # ADDED
        python --version # ADDED
        echo ">>> Running Uvicorn command..." # ADDED
        # Run uvicorn directly now that the venv is activated
        /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload --workers 1 --timeout-keep-alive 300 --timeout-graceful-shutdown 300 --log-level debug --limit-concurrency 20 --backlog 50
        # If the script reaches here, Uvicorn has exited.
        UVICORN_EXIT_CODE=$? # ADDED - Capture exit code
        echo ">>> Uvicorn process finished with exit code: $UVICORN_EXIT_CODE" # ADDED
        # The Uvicorn command runs in the foreground and blocks.
        # The script will exit when Uvicorn stops.
        # The trap handler will manage cleanup.
    fi
}

# Removed prepare_frontend_dev function entirely as requested


# Function to start the frontend (handles dev/prod)
start_frontend() {
    echo "Starting frontend server in $FASTAPI_ENV mode..."
    cd /app/frontend

    # Define the expected path for pnpm installed via curl script
    PNPM_EXEC="/home/appuser/.local/share/pnpm/pnpm"

    if [ "$FASTAPI_ENV" = "production" ]; then
        # Production: Start built application
        echo "Starting production frontend..."
        # Assume pnpm is globally available in production image PATH
        pnpm start &
        FRONTEND_PID=$!
        echo "Production frontend server started with PID: $FRONTEND_PID"
    else
        # Development: Start dev server with turbo, using specific path
        echo "Starting development frontend..."
        # Use the specific path to pnpm executable
        NODE_OPTIONS="--max_old_space_size=4096" "$PNPM_EXEC" dev --turbo &
        FRONTEND_PID=$!
        echo "Development frontend server started with PID: $FRONTEND_PID"
    fi
}

# --- Main Execution ---

# Determine environment (default to development if not set)
export FASTAPI_ENV=${FASTAPI_ENV:-development}
echo "Running entrypoint in $FASTAPI_ENV mode..."

# Ensure required directories exist
ensure_directories

# Check Docker permissions (informational)
check_docker_permissions

# Run migrations (common to both modes)
run_migrations

# Environment-specific startup logic
if [ "$FASTAPI_ENV" = "production" ]; then
    # --- Production Startup ---
    echo "Starting production services..."
    start_backend # Starts in background
    start_frontend # Starts in background

    # Handle shutdown for background processes
    shutdown() {
        echo "Shutting down production services..."
        if [ ! -z "$BACKEND_PID" ]; then kill -TERM $BACKEND_PID; fi
        if [ ! -z "$FRONTEND_PID" ]; then kill -TERM $FRONTEND_PID; fi
        exit 0
    }
    trap shutdown SIGTERM SIGINT

    # Keep container alive while background processes run
    echo "Production services started. Waiting for processes..."
    wait $BACKEND_PID $FRONTEND_PID

else
    # --- Development Startup ---
    echo "Starting development services..."

    # Define the expected path for pnpm installed via curl script
    PNPM_EXEC="/home/appuser/.local/share/pnpm/pnpm"

    # Install pnpm using curl script if not found at the expected path
    echo "Checking for pnpm at $PNPM_EXEC..."
    if [ ! -f "$PNPM_EXEC" ]; then
        echo "pnpm not found at expected path. Installing pnpm using curl script (development mode)..."
        # Ensure the target directory exists and is owned by appuser
        mkdir -p /home/appuser/.local/share/pnpm
        # chown appuser:appuser /home/appuser/.local/share/pnpm # Should already be owned by appuser
        curl -fsSL https://get.pnpm.io/install.sh | SHELL=/bin/bash sh -
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install pnpm using curl script." >&2
            # Attempt fallback with npm if curl fails? Or just exit? Let's exit for now.
            # echo "Attempting fallback installation with npm..."
            # npm install -g pnpm --prefix /home/appuser/.local # Try installing locally? Risky.
            exit 1
        fi
        # Verify installation
        if [ ! -f "$PNPM_EXEC" ]; then
             echo "Error: pnpm installation script ran, but executable not found at $PNPM_EXEC." >&2
             exit 1
        fi
        echo "pnpm installed successfully to $PNPM_EXEC."
    else
        echo "pnpm already installed at $PNPM_EXEC."
    fi

    # Prepare frontend dependencies specifically for dev
    # Skipping prepare_frontend_dev call as requested

    # Start frontend in background first
    start_frontend
    # Removed fixed sleep wait

    echo "Waiting 60 seconds before starting backend to allow Docker Desktop socket time..."
    sleep 60

    # Start backend in foreground (this will block until stopped)
    start_backend

    # Shutdown handling for dev (only need to kill frontend if backend exits)
    shutdown() {
        echo "Shutting down development services..."
        # Backend runs foreground, so trap mainly catches frontend
        if [ ! -z "$FRONTEND_PID" ]; then kill -TERM $FRONTEND_PID; fi
        exit 0
    }
    trap shutdown SIGTERM SIGINT

    # No 'wait' or final echo needed as start_backend runs in foreground and blocks

fi

# Script should block in start_backend (when in dev mode) or wait (when in prod mode)
# No code should execute after the fi unless an error occurs or processes exit.
