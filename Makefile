.PHONY: all clean install dev lint docker-lint format docker-format test docker-test security-check docker-security-check docker-build docker-up docker-down docker-up-prod help migrate frontend-build frontend-dev backend-dev generate-docs sync lock fix-permissions fix-docker fix-all

# Default target
all: install lint test

# Environment variables
ENV ?= dev
ENV_FILE ?= .env.$(ENV)

# Load environment variables from .env file if it exists
ifneq (,$(wildcard $(ENV_FILE)))
    include $(ENV_FILE)
    export
endif

# Python settings
UV = uv
PYTHON_VERSION = 3.13
VENV = .venv
UV_RUN = $(UV) run

# Local command settings (Note: Docker targets are preferred for consistency)
BACKEND_LINT = $(UV_RUN) pylint
BACKEND_FORMAT = $(UV_RUN) black
BACKEND_ISORT = $(UV_RUN) isort
BACKEND_TEST = $(UV_RUN) pytest
BACKEND_SECURITY_CHECK = $(UV_RUN) bandit

# Docker settings based on environment
ifeq ($(ENV),prod)
    DOCKER_COMPOSE_FILE = docker-compose.prod.yml
    BUILD_ARGS = --build-arg NODE_ENV=production --build-arg FASTAPI_ENV=production
else
    DOCKER_COMPOSE_FILE = docker-compose.yml
    BUILD_ARGS = --build-arg NODE_ENV=development --build-arg FASTAPI_ENV=development
endif

# Check if docker compose or docker-compose should be used
DOCKER_COMPOSE_CMD = $(shell command -v docker-compose >/dev/null 2>&1 && echo "docker-compose" || echo "docker compose")

ifeq ($(DOCKER_COMPOSE_CMD),docker-compose)
    DOCKER_COMPOSE = docker-compose -f $(DOCKER_COMPOSE_FILE)
else
    DOCKER_COMPOSE = docker compose -f $(DOCKER_COMPOSE_FILE)
endif

IMAGE_NAME = ghcr.io/toosmooth/doogiebot
CONTAINER_NAME = doogie-chat-container
# BUILD_ENV = # Removed, use BUILD_ARGS instead

# Version management
VERSION = $(shell grep -m 1 version backend/pyproject.toml | cut -d'"' -f2)
GIT_HASH = $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE = $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# System detection
OS = $(shell uname -s)

# Colors for terminal output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color


# Prerequisite checks
check-prereqs:
	@echo "${YELLOW}Checking prerequisites...${NC}"
	@command -v docker >/dev/null 2>&1 || { echo >&2 "${RED}Error: docker is not installed.${NC}"; exit 1; }
	@command -v $(DOCKER_COMPOSE_CMD) >/dev/null 2>&1 || { echo >&2 "${RED}Error: $(DOCKER_COMPOSE_CMD) is not installed or not in PATH.${NC}"; exit 1; }
	# Optional: Add checks for uv and pnpm if local targets are kept and used
	# @command -v $(UV) >/dev/null 2>&1 || { echo >&2 "${RED}Error: uv is not installed.${NC}"; exit 1; }
	# @command -v pnpm >/dev/null 2>&1 || { echo >&2 "${RED}Error: pnpm is not installed.${NC}"; exit 1; }
	@echo "${GREEN}All prerequisites satisfied.${NC}"

# Help target
help:
	@echo "${GREEN}Security Onion Chatbot Makefile${NC}"
	@echo ""
	@echo "${YELLOW}Available targets:${NC}"
	@echo " ${GREEN}all${NC}             : Install dependencies, run linters, and tests"
	@echo " ${GREEN}clean${NC}           : Clean up build artifacts, caches, and virtual environment"
	@echo " ${GREEN}fix-permissions${NC} : Fix permissions for Docker volumes"
	@echo " ${GREEN}fix-docker${NC}      : Fix Docker Compose file formatting issues"
	@echo " ${GREEN}fix-all${NC}         : Fix all Docker environment issues"
	@echo " ${GREEN}install${NC}         : Install backend and frontend dependencies"
	@echo " ${GREEN}lock${NC}            : Generate lock file for reproducible builds"
	@echo " ${GREEN}sync${NC}            : Sync dependencies from lock file"
	@echo " ${GREEN}dev${NC}             : Start development environment with Docker"
	@echo " ${GREEN}lint${NC}            : Run linters locally using virtual environment"
	@echo " ${GREEN}docker-lint${NC}     : Run linters in Docker container"
	@echo " ${GREEN}format${NC}          : Format code locally using virtual environment"
	@echo " ${GREEN}docker-format${NC}   : Format code in Docker container"
	@echo " ${GREEN}test${NC}            : Run tests locally using virtual environment"
	@echo " ${GREEN}docker-test${NC}     : Run tests in Docker container"
	@echo " ${GREEN}security-check${NC}  : Run security checks locally using virtual environment"
	@echo " ${GREEN}docker-security${NC} : Run security checks in Docker container"
	@echo " ${GREEN}docker-build${NC}    : Build Docker image with cache (default)"
	@echo " ${GREEN}docker-build-fresh${NC}: Build Docker image without cache"
	@echo " ${GREEN}docker-up${NC}       : Start Docker container (uses ENV=dev/prod)"
	# Removed docker-up-prod help text
	@echo " ${GREEN}docker-down${NC}     : Stop Docker container (uses ENV=dev/prod)"
	@echo " ${GREEN}migrate${NC}         : Run database migrations in Docker"
	@echo " ${GREEN}frontend-build${NC}  : Build frontend for production in Docker"
	@echo " ${GREEN}frontend-dev${NC}    : Start frontend development server"
	@echo " ${GREEN}backend-dev${NC}     : Start backend development server"
	@echo " ${GREEN}sync${NC}            : Sync codebase to a remote machine"

# Clean up
clean:
	@echo "${YELLOW}Cleaning up...${NC}"
	rm -rf $(VENV) *.egg-info dist build __pycache__ .pytest_cache .coverage
	rm -rf frontend/.next frontend/node_modules
	rm -rf backend/__pycache__ backend/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "${GREEN}Cleanup complete.${NC}"

# Generate lockfile
lock:
	@echo "${YELLOW}Generating lock file...${NC}"
	cd backend && $(UV) lock
	@echo "${GREEN}Lock file generated.${NC}"

# Sync dependencies
sync:
	@echo "${YELLOW}Syncing dependencies from lock file...${NC}"
	cd backend && $(UV) sync
	@echo "${GREEN}Dependencies synced.${NC}"


docker-build: check-prereqs
	@echo "${YELLOW}Building ${IMAGE_NAME} image with cache...${NC}"
	docker build $(BUILD_ARGS) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VERSION=$(VERSION) \
		--tag ${IMAGE_NAME}:latest \
		--tag ${IMAGE_NAME}:$(VERSION) \
		--tag ${IMAGE_NAME}:$(GIT_HASH) \
		--cache-from ${IMAGE_NAME}:latest \
		-f Dockerfile .
	@echo "${GREEN}Docker image built and tagged as ${IMAGE_NAME}:latest, ${IMAGE_NAME}:$(VERSION), and ${IMAGE_NAME}:$(GIT_HASH).${NC}"

# Docker builds - Fresh build without cache
docker-build-fresh: check-prereqs
	@echo "${YELLOW}Building fresh ${IMAGE_NAME} image with no cache...${NC}"
	docker build $(BUILD_ARGS) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VERSION=$(VERSION) \
		--no-cache \
		--tag ${IMAGE_NAME}:latest \
		--tag ${IMAGE_NAME}:$(VERSION) \
		--tag ${IMAGE_NAME}:$(GIT_HASH) \
		-f Dockerfile .
	@echo "${GREEN}Fresh Docker image built and tagged as ${IMAGE_NAME}:latest, ${IMAGE_NAME}:$(VERSION), and ${IMAGE_NAME}:$(GIT_HASH).${NC}"


# Start development environment (Simplified: dev now just runs docker-up)
dev: docker-up

# Start Docker container (handles both dev and prod based on ENV)
docker-up: check-prereqs
	@echo "${YELLOW}Starting Docker container in $(ENV) mode...${NC}"
	$(DOCKER_COMPOSE) up -d # Run detached
	@echo "${YELLOW}Waiting for services to be healthy...${NC}"
	@timeout=120; counter=0; \
	until $(DOCKER_COMPOSE) ps --filter name=app --filter status=running | grep -q 'running'; do \
		sleep 2; \
		counter=$$((counter + 2)); \
		if [ $$counter -ge $$timeout ]; then \
			echo "${RED}Timed out waiting for services to start or become healthy.${NC}"; \
			$(DOCKER_COMPOSE) logs app; \
			exit 1; \
		fi; \
		echo -n "."; \
	done; \
	echo "\n${GREEN}Services are now running and healthy.${NC}"


# Stop Docker container (handles both dev and prod based on ENV)
docker-down: check-prereqs
	@echo "${YELLOW}Stopping Docker container...${NC}"
	$(DOCKER_COMPOSE) down
	@echo "${GREEN}Docker container stopped.${NC}"

# Primary commands should use the Docker environment for consistency

# Linting (Local - commented out)
# lint:
#	@echo "${YELLOW}Running backend linters locally...${NC}"
#	cd backend && $(BACKEND_LINT) app --disable=C0111
#	@echo "${YELLOW}Running frontend linters locally...${NC}"
#	cd frontend && npm run lint
#	@echo "${GREEN}Linting complete.${NC}"

# Linting (Docker) - Preferred method
docker-lint: check-prereqs
	@echo "${YELLOW}Running backend linters in Docker...${NC}"
	$(DOCKER_COMPOSE) exec -u root app bash -c "chown -R appuser:appuser /app/.venv || true"
	-$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && source /app/.venv/bin/activate && uv pip install pylint && python -m pylint app --disable=C0111,R0801"
	@echo "${YELLOW}Running frontend linters in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm run lint"
	@echo "${GREEN}Linting complete.${NC}"

# Formatting (Local - commented out)
# format:
#	@echo "${YELLOW}Formatting backend code locally...${NC}"
#	cd backend && $(BACKEND_FORMAT) app
#	cd backend && $(BACKEND_ISORT) app
#	@if [ -d "backend/tests" ]; then \
#		cd backend && $(BACKEND_FORMAT) tests; \
#		cd backend && $(BACKEND_ISORT) tests; \
#	fi
#	@echo "${YELLOW}Formatting frontend code locally...${NC}"
#	cd frontend && npm run format
#	@echo "${GREEN}Formatting complete.${NC}"

# Formatting (Docker) - Preferred method
docker-format: check-prereqs
	@echo "${YELLOW}Formatting backend code in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && uv run black app && uv run isort app && if [ -d 'tests' ]; then uv run black tests && uv run isort tests; fi"
	@echo "${YELLOW}Formatting frontend code in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm run format"
	@echo "${GREEN}Formatting complete.${NC}"

# Testing (Local - commented out)
# test:
#	@echo "${YELLOW}Running tests locally...${NC}"
#	cd backend && $(BACKEND_TEST) tests
#	cd frontend && npm test
#	@echo "${GREEN}Tests complete.${NC}"

# Testing (Docker) - Preferred method
docker-test: check-prereqs
	@echo "${YELLOW}Running tests in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && uv run pytest --cov=app --cov-report=term-missing backend/tests" # Use uv run to execute pytest
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm test"
	@echo "${GREEN}Tests complete.${NC}"

# Security checks (Local - commented out)
# security-check:
#	@echo "${YELLOW}Running security checks locally...${NC}"
#	cd backend && $(BACKEND_SECURITY_CHECK) -r app -c bandit.yaml
#	cd frontend && npm audit
#	@echo "${GREEN}Security checks complete.${NC}"

# Security checks (Docker) - Preferred method
docker-security: check-prereqs
	@echo "${YELLOW}Running security checks in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && uv run bandit -r app -c bandit.yaml"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm audit"
	@echo "${GREEN}Security checks complete.${NC}"

# Run database migrations (Docker) - Preferred method
migrate: check-prereqs
	@echo "${YELLOW}Running database migrations...${NC}"
	# Delete existing DB and old migration file to ensure clean initial generation
	$(DOCKER_COMPOSE) exec app bash -c "rm -f /app/data/db/doogie.db && rm -f /app/backend/alembic/versions/*.py"
	# Autogenerate the initial migration based on models
	$(DOCKER_COMPOSE) exec -e UV_CACHE_DIR=/tmp/uv-cache-new app bash -c "source /app/.venv/bin/activate && cd /app/backend && alembic revision --autogenerate -m 'Initial schema'"
	# Apply the generated migration
	$(DOCKER_COMPOSE) exec -e UV_CACHE_DIR=/tmp/uv-cache-new app bash -c "source /app/.venv/bin/activate && cd /app/backend && alembic upgrade head"
	@echo "${GREEN}Migrations complete.${NC}"

# Build frontend for production (Docker) - Preferred method
frontend-build: check-prereqs
	@echo "${YELLOW}Building frontend for production...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm run build"
	@echo "${GREEN}Frontend build complete.${NC}"

# Start frontend dev server
frontend-dev:
	@echo "${YELLOW}Starting frontend development server...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm run dev"

# Start backend dev server
backend-dev:
	@echo "${YELLOW}Starting backend development server...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

# Generate documentation
generate-docs:
	@echo "${YELLOW}Generating documentation...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && sphinx-apidoc -o docs/source app"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend/docs && make html"
	@echo "${GREEN}Documentation generated.${NC}"

# Sync code to remote server
sync:
	@echo "${YELLOW}Syncing codebase to remote machine...${NC}"
	@read -p "Enter destination (e.g., user@server:/path/to/project): " destination; \
	./sync-doogie.sh $$destination
	@echo "${GREEN}Sync complete.${NC}"

# Debug target to build, wait, check logs, and run fetch test with local server
debug:
	docker compose down
	@echo "${YELLOW}Starting debug sequence...${NC}"
	# Removed unnecessary local HTTP server startup/shutdown
	@echo "${YELLOW}Rebuilding Docker image without cache to apply changes...${NC}"
	$(DOCKER_COMPOSE) build app
	@echo "${YELLOW}Starting Docker container with fresh image...${NC}"
	$(DOCKER_COMPOSE) up -d
	@echo "${YELLOW}Waiting for app container to become healthy (max 120s)...${NC}"
	@timeout=120; counter=0; \
	until docker compose ps --filter name=app --filter status=running | grep -q '(healthy)'; do \
	    sleep 2; \
	    counter=$$((counter + 2)); \
	    if [ $$counter -ge $$timeout ]; then \
	        echo "${RED}Timed out waiting for app container to become healthy.${NC}"; \
	        docker compose logs app; \
	        exit 1; \
	    fi; \
	    echo -n "."; \
	done;
	echo "\n${GREEN}App container is healthy. Running fetch tool test against public URL...${NC}"
	TEST_URL="https://example.com" ./backend/tests/test_fetch_tool.sh
	# Removed comment about unnecessary local HTTP server shutdown
	@echo "${YELLOW}Displaying recent logs...${NC}"
	@$(DOCKER_COMPOSE) logs app --since 5m || true
	@echo "${GREEN}Debug sequence complete.${NC}"

# CI/CD Targets
ci: check-prereqs docker-lint docker-security docker-test
	@echo "${GREEN}CI checks completed successfully${NC}"

docker-push: check-prereqs
	@echo "${YELLOW}Pushing ${IMAGE_NAME} image to container registry...${NC}"
	docker push ${IMAGE_NAME}:latest
	docker push ${IMAGE_NAME}:$(VERSION)
	docker push ${IMAGE_NAME}:$(GIT_HASH)
	@echo "${GREEN}Docker images pushed to registry.${NC}"

cd: check-prereqs docker-build docker-push
	@echo "${GREEN}CD pipeline completed successfully${NC}"

# Security Scanning (requires trivy to be installed locally)
security-scan: check-prereqs
	@echo "${YELLOW}Scanning Docker image for vulnerabilities...${NC}"
	@command -v trivy >/dev/null 2>&1 || { echo >&2 "${RED}Error: trivy is not installed. Please install trivy to use this target.${NC}"; exit 1; }
	trivy image ${IMAGE_NAME}:latest --severity HIGH,CRITICAL
	@echo "${GREEN}Security scan complete.${NC}"

# Utility to run a script (use with caution)
run-script:
	@echo "${YELLOW}Running script: $(SCRIPT)${NC}"
	@if [ -z "$(SCRIPT)" ]; then \
		echo "${RED}Error: No script specified. Usage: make run-script SCRIPT=path/to/script.sh${NC}"; \
		exit 1; \
	fi
	@if [ ! -f "$(SCRIPT)" ]; then \
		echo "${RED}Error: Script $(SCRIPT) not found.${NC}"; \
		exit 1; \
	fi
	@echo "${YELLOW}Checking script for potential issues (requires shellcheck)...${NC}"
	@command -v shellcheck >/dev/null 2>&1 && shellcheck $(SCRIPT) || echo "${YELLOW}WARNING: shellcheck not found or script has issues. Proceed with caution.${NC}"
	@chmod +x $(SCRIPT)
	@./$(SCRIPT)

