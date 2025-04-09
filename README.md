# DoogieBot

This project is a chatbot application built with Next.js (frontend) and FastAPI (backend). It allows users to interact with documents and a Large Language Model (LLM). The project initially focused on addressing markdown formatting issues in chatbot responses but has evolved into a full-fledged application.

## Project Overview
This application provides a user interface for:
- Managing documents (uploading, processing, and retrieving)
- Configuring LLM settings
- Interacting with an LLM through a chat interface
- Managing user accounts

## Technology Stack

- **Frontend:** Next.js, React, Tailwind CSS
- **Backend:** FastAPI, Python
- **Database:** (Inferred from migrations - likely PostgreSQL or similar)
- **LLM Integration:** Supports multiple LLMs (OpenAI, Anthropic, Ollama, etc.)
- **Vector Database:** (Inferred from code - likely FAISS)
- **Deployment:** Docker, Docker Compose, Makefile

## Environment Setup

This project uses environment variables for configuration, particularly for secrets like API keys and the application `SECRET_KEY`.

1.  **Copy Example:** Copy the `.env.example` file to `.env.dev` for development and `.env.prod` for production.
    ```bash
    cp .env.example .env.dev
    cp .env.example .env.prod
    ```
2.  **Edit Files:** Edit `.env.dev` and `.env.prod` to add your specific API keys (OpenAI, Anthropic, etc.) and generate a strong, unique `SECRET_KEY` for each environment.
    *   **Important:** The `SECRET_KEY` is crucial for security and **must** be set in both `.env.dev` and `.env.prod`.
3.  **Git Ignore:** These `.env.*` files are included in `.gitignore` and should **never** be committed to version control.

The `Makefile` automatically loads the appropriate `.env.*` file based on the `ENV` variable (defaulting to `dev`).

## Running the Application (Docker & Makefile)

The primary way to build, run, and manage the application is through the provided `Makefile` targets, which leverage Docker and Docker Compose for a consistent environment.

**Prerequisites:**
*   Docker Engine
*   Docker Compose (V2 `docker compose` or V1 `docker-compose`)

**Key Makefile Targets:**

*   `make docker-build`: Builds the Docker image using layer caching (recommended for faster builds). Tags as `latest`, version, and git hash.
*   `make docker-build-fresh`: Builds the Docker image without using cache.
*   `make docker-up`: Starts the application containers in **development** mode (using `docker-compose.yml` and `.env.dev`). Builds the image if not present. Waits for services to become healthy.
*   `make ENV=prod docker-up`: Starts the application containers in **production** mode (using `docker-compose.prod.yml` and `.env.prod`). Builds the image if not present. Waits for services to become healthy.
*   `make docker-down`: Stops the running application containers (uses `ENV` to determine which compose file).
*   `make docker-lint`: Runs linters for backend and frontend inside the Docker container.
*   `make docker-format`: Formats code for backend and frontend inside the Docker container.
*   `make docker-test`: Runs tests for backend and frontend inside the Docker container.
*   `make docker-security`: Runs security checks (bandit, npm audit) inside the Docker container.
*   `make migrate`: Runs database migrations (Alembic) inside the Docker container.
*   `make ci`: Runs a sequence of checks suitable for Continuous Integration (lint, security, test).
*   `make help`: Displays all available Makefile targets.

**Development Workflow:**

1.  Ensure prerequisites are installed.
2.  Create and populate `.env.dev` (see Environment Setup).
3.  Run `make docker-up`. This will:
    *   Build the image using the local `Dockerfile` if necessary.
    *   Start the container using `docker-compose.yml`.
    *   Mount local code (`./backend`, `./frontend`) into the container for live reloading.
    *   Run the unified `entrypoint.sh` script, which installs dependencies (if needed) and starts dev servers.
4.  Access the frontend at `http://localhost:3000` and the backend API at `http://localhost:8000`.
5.  To stop: `make docker-down`.

**Production Workflow:**

1.  Ensure prerequisites are installed.
2.  Create and populate `.env.prod` with production secrets.
3.  Build the image: `make docker-build` (or `make docker-build-fresh`).
4.  Push the image to a registry (optional but recommended): `make docker-push` (ensure `IMAGE_NAME` in Makefile points to your registry).
5.  Start the container: `make ENV=prod docker-up`. This uses `docker-compose.prod.yml` and the pre-built image (if `IMAGE_NAME` matches).
6.  To stop: `make ENV=prod docker-down`.

**Docker Socket Mount:**
The `docker-compose.yml` file mounts the host's Docker socket (`/var/run/docker.sock`) into the container.
*   **Purpose:** This is required for certain Model Context Protocol (MCP) servers running within the application container that need to interact with the Docker daemon (e.g., to start other containers).
*   **Security Warning:** Mounting the Docker socket effectively grants the container root-level access to the host system via the Docker daemon. This is a significant security risk. Only run this configuration in trusted environments and be aware of the implications.

## Memory Bank
This project utilizes a memory bank system to maintain context and documentation across sessions. The memory bank consists of Markdown files located in the `memory-bank/` directory. These files contain information about the project's goals, architecture, context, and progress. Key files include:

- `projectbrief.md`: Defines the core requirements and goals of the project.
- `productContext.md`: Explains the purpose and functionality of the application.
- `activeContext.md`: Tracks the current focus, recent changes, and next steps.
- `systemPatterns.md`: Describes the system architecture and design patterns.
- `techContext.md`: Lists the technologies used and development setup.
- `progress.md`: Summarizes the current status and known issues.
