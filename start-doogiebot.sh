#!/bin/bash

# Colors for terminal output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${YELLOW}Starting DoogieBot setup...${NC}"

# Create entrypoint.sh if it doesn't exist
if [ ! -f "./entrypoint.sh" ]; then
    echo -e "${YELLOW}Creating entrypoint.sh script...${NC}"
    cat > ./entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Ensure required directories exist
ensure_directories() {
    echo "Ensuring required directories exist..."
    mkdir -p /app/backend/db
    mkdir -p /app/backend/indexes
    mkdir -p /app/backend/uploads

    # Set proper permissions
    chmod -R 755 /app/backend/db
    chmod -R 755 /app/backend/indexes
    chmod -R 755 /app/backend/uploads

    echo "Directories checked and created if needed."
}

# Run database migrations with retry logic
run_migrations() {
    echo "Running database migrations..."
    cd /app/backend

    # Create database directory if it doesn't exist
    mkdir -p "$PWD/db"
    touch "$PWD/db/doogie.db"
    chmod 666 "$PWD/db/doogie.db"

    # Try to run migrations with retries
    local max_attempts=5
    local attempt=1
    local success=false

    while [ $attempt -le $max_attempts ] && [ "$success" = false ]; do
        echo "Attempt $attempt of $max_attempts to run migrations..."
        if uv run alembic upgrade head; then
            success=true
            echo "Database migrations completed successfully."
        else
            echo "Migration attempt $attempt failed. Waiting before retry..."
            sleep 5
            attempt=$((attempt+1))
        fi
    done

    if [ "$success" = false ]; then
        echo "ERROR: Failed to run migrations after $max_attempts attempts."
        echo "Will continue startup, but application may not work correctly."
    fi

    # Verify database tables exist
    echo "Verifying database tables..."
    if [ -f "db/doogie.db" ]; then
        # Run a simple SQL query to verify the database file exists and has tables
        sqlite3 db/doogie.db "SELECT name FROM sqlite_master WHERE type='table';"
        if ! sqlite3 db/doogie.db ".tables" | grep -q "users"; then
            echo "WARNING: Users table not found in database. Using SQLAlchemy initialization."
            echo "Initializing database schema with SQLAlchemy..."
            # Create a simple script to initialize the database
            cat > init_db.py << EOFILE
from app.db.base import init_db
init_db()
EOFILE
            # Run the script
            uv run python init_db.py
            echo "Database initialization completed."
        else
            echo "Database verification successful."
        fi
    else
        echo "WARNING: Database file not found. Using SQLAlchemy initialization."
        echo "Initializing database schema with SQLAlchemy..."
        # Create a simple script to initialize the database
        cat > init_db.py << EOFILE
from app.db.base import init_db
init_db()
EOFILE
        # Run the script
        uv run python init_db.py
        echo "Database initialization completed."
    fi
}

# Function to start the backend
start_backend() {
    echo "Starting backend server..."
    cd /app/backend
    # Use a single worker with memory limits to prevent crashes
    # Set PYTHONMALLOC=debug to help catch memory issues
    export PYTHONMALLOC=debug
    # Set memory limits
    export PYTHONWARNINGS=always
    # Use a single worker with memory limits and run via uv
    uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload --workers 1 --timeout-keep-alive 300 --timeout-graceful-shutdown 300 --log-level debug --limit-concurrency 20 --backlog 50 &
    BACKEND_PID=$!
    echo "Backend server started with PID: $BACKEND_PID"
}

# Function to prepare frontend dependencies
prepare_frontend() {
    cd /app/frontend

    # Configure pnpm to use a specific store directory with proper permissions
    echo "Configuring pnpm store..."
    pnpm config set store-dir /app/.pnpm-store

    # Check if we have write permissions
    if [ ! -w "." ] || [ ! -w "/app/.pnpm-store" ]; then
        echo "Warning: Permission issues detected. Attempting to fix..."
        # mkdir -p node_modules .next # Directory creation moved to Dockerfile
    fi

    # Install frontend dependencies if needed
    # Check for a common binary instead of the whole .bin directory
    if [ ! -f "node_modules/.bin/next" ]; then
        echo "Installing frontend dependencies..."
        # Use --shamefully-hoist for better compatibility in Docker
        # Use --no-strict-peer-dependencies to avoid peer dependency issues
        pnpm install --shamefully-hoist --no-strict-peer-dependencies
    else
        echo "Frontend dependencies already installed."
    fi
}


# Function to start the frontend
start_frontend() {
    echo "Starting frontend server..."
    cd /app/frontend

    # Start Next.js development server with turbo mode
    NODE_OPTIONS="--max_old_space_size=4096" pnpm dev --turbo &
    FRONTEND_PID=$!
    echo "Frontend server started with PID: $FRONTEND_PID"
}

# Ensure required directories exist
ensure_directories

# Run migrations
run_migrations

# Add a delay to ensure database is ready
echo "Waiting for 3 seconds to ensure database is ready..."
sleep 3

# Prepare frontend before starting services
prepare_frontend # Re-added this call

# Start services
start_backend
echo "Waiting 5 seconds for backend to initialize..."
sleep 5
start_frontend

# Handle shutdown
shutdown() {
    echo "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill -TERM $BACKEND_PID
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill -TERM $FRONTEND_PID
    fi
    exit 0
}

# Trap SIGTERM and SIGINT
trap shutdown SIGTERM SIGINT

# Keep the container running
echo "All services started. Container is now running..."
wait
EOF
fi

# Create entrypoint.prod.sh if it doesn't exist
if [ ! -f "./entrypoint.prod.sh" ]; then
    echo -e "${YELLOW}Creating entrypoint.prod.sh script...${NC}"
    cat > ./entrypoint.prod.sh << 'EOF'
#!/bin/bash
set -e

# Run database migrations
run_migrations() {
    echo "Running database migrations..."
    cd /app/backend
    # Alembic should find alembic.ini in the current directory
    python -m alembic upgrade head
    # The create_tag_tables.py script is redundant as Alembic handles all table creation.

    echo "Database migrations completed."
}

# Function to start the backend in production mode
start_backend() {
    echo "Starting backend server in production mode..."
    cd /app/backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --timeout-keep-alive 300 &
    BACKEND_PID=$!
    echo "Backend server started with PID: $BACKEND_PID"
}

# Function to start the frontend in production mode
start_frontend() {
    echo "Starting frontend server in production mode..."
    cd /app/frontend
    pnpm start &
    FRONTEND_PID=$!
    echo "Frontend server started with PID: $FRONTEND_PID"
}

# Run migrations
run_migrations

# Start services in production mode
start_backend
start_frontend

# Handle shutdown
shutdown() {
    echo "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill -TERM $BACKEND_PID
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill -TERM $FRONTEND_PID
    fi
    exit 0
}

# Trap SIGTERM and SIGINT
trap shutdown SIGTERM SIGINT

# Keep the container running
echo "All services started in production mode. Container is now running..."
wait
EOF
fi

# Make scripts executable
echo -e "${YELLOW}Making scripts executable...${NC}"
chmod +x ./entrypoint.sh
chmod +x ./entrypoint.prod.sh

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p ./data/db
mkdir -p ./data/indexes

# Make sure directories have proper permissions
echo -e "${YELLOW}Setting proper permissions...${NC}"
chmod -R 755 ./data

# Check if using development or production
if [ "$1" == "prod" ]; then
    echo -e "${YELLOW}Starting DoogieBot in production mode...${NC}"
    docker compose -f docker-compose.prod.yml up -d
else
    echo -e "${YELLOW}Starting DoogieBot in development mode...${NC}"
    docker compose up -d
fi

echo -e "${GREEN}DoogieBot is now starting up. Check logs with 'docker compose logs -f'${NC}"