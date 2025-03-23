.PHONY: all clean install dev lint docker-lint format docker-format test docker-test security-check docker-security-check docker-build docker-up docker-down docker-up-prod help migrate frontend-build frontend-dev backend-dev generate-docs sync

# Default target
all: install lint test

# Python settings
PYTHON = python3
VENV = .venv
PIP = $(VENV)/bin/pip

# Local command settings
BACKEND_LINT = $(VENV)/bin/pylint
BACKEND_FORMAT = $(VENV)/bin/black
BACKEND_ISORT = $(VENV)/bin/isort
BACKEND_TEST = $(VENV)/bin/pytest
BACKEND_SECURITY_CHECK = $(VENV)/bin/bandit

# Docker settings
IMAGE_NAME = doogie-chat
CONTAINER_NAME = doogie-chat-container
DOCKER_COMPOSE = docker compose
BUILD_ENV = 

# Version management
VERSION = $(shell grep -m 1 version backend/pyproject.toml | cut -d'"' -f2)

# System detection
OS = $(shell uname -s)

# Colors for terminal output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Help target
help:
	@echo "${GREEN}Security Onion Chatbot Makefile${NC}"
	@echo ""
	@echo "${YELLOW}Available targets:${NC}"
	@echo " ${GREEN}all${NC}             : Install dependencies, run linters, and tests"
	@echo " ${GREEN}clean${NC}           : Clean up build artifacts, caches, and virtual environment"
	@echo " ${GREEN}install${NC}         : Install backend and frontend dependencies"
	@echo " ${GREEN}dev${NC}             : Start development environment with Docker"
	@echo " ${GREEN}lint${NC}            : Run linters locally using virtual environment"
	@echo " ${GREEN}docker-lint${NC}     : Run linters in Docker container"
	@echo " ${GREEN}format${NC}          : Format code locally using virtual environment"
	@echo " ${GREEN}docker-format${NC}   : Format code in Docker container"
	@echo " ${GREEN}test${NC}            : Run tests locally using virtual environment"
	@echo " ${GREEN}docker-test${NC}     : Run tests in Docker container"
	@echo " ${GREEN}security-check${NC}  : Run security checks locally using virtual environment"
	@echo " ${GREEN}docker-security${NC} : Run security checks in Docker container"
	@echo " ${GREEN}docker-build${NC}    : Build Docker image"
	@echo " ${GREEN}docker-up${NC}       : Start Docker container in development mode"
	@echo " ${GREEN}docker-up-prod${NC}  : Start Docker container in production mode"
	@echo " ${GREEN}docker-down${NC}     : Stop Docker container"
	@echo " ${GREEN}migrate${NC}         : Run database migrations"
	@echo " ${GREEN}frontend-build${NC}  : Build frontend for production"
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

# Installation (used for local dev without Docker)
install:
	@echo "${YELLOW}Setting up virtual environment...${NC}"
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	
	@echo "${YELLOW}Installing backend dependencies...${NC}"
	$(PIP) install -r backend/requirements.txt
	$(PIP) install pylint black isort pytest bandit
	
	@echo "${YELLOW}Installing frontend dependencies...${NC}"
	cd frontend && npm install
	
	@echo "${GREEN}Installation complete.${NC}"

# Docker builds
docker-build:
	@echo "${YELLOW}Building Docker image...${NC}"
	$(DOCKER_COMPOSE) build $(BUILD_ENV)
	@echo "${GREEN}Docker build complete.${NC}"

# Start development environment
dev: docker-up

# Start Docker in development mode
docker-up:
	@echo "${YELLOW}Starting Docker container in development mode...${NC}"
	$(DOCKER_COMPOSE) up $(BUILD_ENV)

# Start Docker in production mode
docker-up-prod:
	@echo "${YELLOW}Starting Docker container in production mode...${NC}"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up $(BUILD_ENV)

# Stop Docker
docker-down:
	@echo "${YELLOW}Stopping Docker container...${NC}"
	$(DOCKER_COMPOSE) down
	@echo "${GREEN}Docker container stopped.${NC}"

# Primary commands use the local environment by default

# Linting (Local as primary command)
lint:
	@echo "${YELLOW}Running backend linters locally...${NC}"
	$(BACKEND_LINT) backend/app --disable=C0111
	@echo "${YELLOW}Running frontend linters locally...${NC}"
	cd frontend && npm run lint
	@echo "${GREEN}Linting complete.${NC}"

# Linting (Docker)
docker-lint:
	@echo "${YELLOW}Running backend linters in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && pip install pylint && pylint app --disable=C0111"
	@echo "${YELLOW}Running frontend linters in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm run lint"
	@echo "${GREEN}Linting complete.${NC}"

# Formatting (Local as primary command)
format:
	@echo "${YELLOW}Formatting backend code locally...${NC}"
	$(BACKEND_FORMAT) backend/app
	$(BACKEND_ISORT) backend/app
	@if [ -d "backend/tests" ]; then \
		$(BACKEND_FORMAT) backend/tests; \
		$(BACKEND_ISORT) backend/tests; \
	fi
	@echo "${YELLOW}Formatting frontend code locally...${NC}"
	cd frontend && npm run format
	@echo "${GREEN}Formatting complete.${NC}"

# Formatting (Docker)
docker-format:
	@echo "${YELLOW}Formatting backend code in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && pip install black isort && black app && isort app && if [ -d 'tests' ]; then black tests && isort tests; fi"
	@echo "${YELLOW}Formatting frontend code in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm run format"
	@echo "${GREEN}Formatting complete.${NC}"

# Testing (Local as primary command)
test:
	@echo "${YELLOW}Running tests locally...${NC}"
	$(BACKEND_TEST) backend/tests
	cd frontend && npm test
	@echo "${GREEN}Tests complete.${NC}"

# Testing (Docker)
docker-test:
	@echo "${YELLOW}Running tests in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && pip install pytest && pytest"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm test"
	@echo "${GREEN}Tests complete.${NC}"

# Security checks (Local as primary command)
security-check:
	@echo "${YELLOW}Running security checks locally...${NC}"
	$(BACKEND_SECURITY_CHECK) -r backend/app -c backend/bandit.yaml
	cd frontend && npm audit
	@echo "${GREEN}Security checks complete.${NC}"

# Security checks (Docker)
docker-security:
	@echo "${YELLOW}Running security checks in Docker...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && pip install bandit && bandit -r app -c bandit.yaml"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/frontend && npm audit"
	@echo "${GREEN}Security checks complete.${NC}"

# Run database migrations
migrate:
	@echo "${YELLOW}Running database migrations...${NC}"
	$(DOCKER_COMPOSE) exec app bash -c "cd /app/backend && python -m alembic upgrade head"
	@echo "${GREEN}Migrations complete.${NC}"

# Build frontend for production
frontend-build:
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