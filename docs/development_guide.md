# Doogie Chat Bot - Development Guide

This guide provides instructions for setting up the development environment, running the application, executing tests, and adhering to coding standards for the Doogie Chat Bot project.

## Prerequisites

*   **Docker:** Required for running the application environment. Ensure Docker Desktop or Docker Engine is installed and running.
*   **Docker Compose:** Used for orchestrating the application container. Version V2 (`docker compose`) is preferred over V1 (`docker-compose`).
*   **Make:** Used for running common development tasks defined in the `Makefile`.
*   **Git:** For version control.
*   **Code Editor:** VS Code with recommended extensions (Python, Pylance, ESLint, Prettier, Black, isort) is suggested.

## Project Structure

*   `backend/`: Contains the Python FastAPI backend application.
    *   `app/`: Core application code (API routes, services, models, schemas).
    *   `alembic/`: Database migration scripts.
    *   `tests/`: Backend tests.
*   `frontend/`: Contains the Next.js frontend application.
    *   `pages/`: Next.js page routes.
    *   `components/`: Reusable React components.
    *   `services/`: Frontend API interaction logic.
    *   `hooks/`: Custom React hooks.
    *   `contexts/`: React context providers.
*   `docs/`: Project documentation.
*   `docker-compose.yml`: Docker Compose configuration for development.
*   `docker-compose.prod.yml`: Docker Compose configuration for production simulation.
*   `Dockerfile`: Defines the single Docker image for both dev and prod.
*   `Makefile`: Defines common development commands.
*   `.clinerules`: Project-specific rules enforced during development.

## Development Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd doogie-chat
    ```

2.  **Environment Variables:**
    *   Copy the example environment file: `cp .env.example .env`
    *   Review and update `.env` with necessary configurations (e.g., API keys for LLM services if not using defaults, database path if changed). **Do not commit `.env`**.

3.  **Build Docker Image:**
    *   This step builds the single Docker image containing both backend and frontend dependencies and code.
    ```bash
    make docker-build
    # or directly: docker compose build
    ```

4.  **Install Hooks (Optional but Recommended):**
    *   Consider setting up pre-commit hooks to automatically run linters/formatters before committing code.

## Running the Application

### Development Mode

This mode uses `docker compose` with the `docker-compose.yml` file. It includes features like hot-reloading for both frontend and backend.

1.  **Start the Container:**
    ```bash
    make dev
    # or directly: docker compose up
    ```
2.  **Access the Application:**
    *   Frontend: `http://localhost:3000`
    *   Backend API Docs: `http://localhost:8000/docs`
3.  **Hot Reloading:** Changes made to files in `backend/` or `frontend/` should automatically trigger reloads within the container.
4.  **Stopping the Container:**
    ```bash
    make docker-down
    # or directly: docker compose down
    ```

### Production Simulation Mode

This mode uses `docker compose` with both `docker-compose.yml` and `docker-compose.prod.yml`. It runs the application using production-like settings (e.g., `gunicorn` for the backend, optimized frontend build).

1.  **Start the Container:**
    ```bash
    make docker-up-prod
    # or directly: docker compose -f docker-compose.yml -f docker-compose.prod.yml up
    ```
2.  **Access the Application:**
    *   Frontend: `http://localhost:3000`
3.  **Stopping the Container:**
    ```bash
    make docker-down
    # or directly: docker compose -f docker-compose.yml -f docker-compose.prod.yml down
    ```

## Database Migrations

The backend uses Alembic for database migrations (SQLite).

*   **Run Migrations:** Apply pending migrations. This is usually done automatically on container start in development, but can be run manually if needed.
    ```bash
    make migrate
    # or inside the container: alembic upgrade head
    ```
*   **Creating New Migrations:**
    1.  Modify SQLAlchemy models in `backend/app/models/`.
    2.  Run the following command *inside the running development container*:
        ```bash
        # docker compose exec doogie-chat bash
        alembic revision --autogenerate -m "Your migration description"
        # exit
        ```
    3.  Review the generated migration script in `backend/alembic/versions/`.
    4.  Apply the new migration: `make migrate`.

## Testing

All tests should be run within the Docker container to ensure consistency.

*   **Run All Tests:**
    ```bash
    make docker-test
    ```
*   This command typically executes `pytest` within the container.

## Code Quality and Formatting

Linters and formatters are used to maintain code consistency and quality.

*   **Linters:**
    *   Python: `pylint`, `flake8` (via `ruff`)
    *   TypeScript/JavaScript: `eslint`
*   **Formatters:**
    *   Python: `black`, `isort` (via `ruff format`)
    *   TypeScript/JavaScript: `prettier`
*   **Security Checks:**
    *   Python: `bandit`
    *   Node: `npm audit` (run during frontend install/build)

*   **Run Linters:**
    ```bash
    make docker-lint
    ```
*   **Run Formatters:**
    ```bash
    make docker-format
    ```
*   **Run Security Checks:**
    ```bash
    make docker-security
    ```
*   **Run All Checks (Lint, Format, Test, Security):**
    ```bash
    make all
    ```

## Coding Standards & Best Practices

*   **Follow Style Guides:** Adhere to the configurations defined for `black`, `isort`, `pylint`, `eslint`, and `prettier`. Use `make docker-format` regularly.
*   **Descriptive Naming:** Use clear and meaningful names for variables, functions, classes, components, etc.
*   **Modularity:** Organize code logically into modules (Python) and components/hooks/services (TypeScript). Separate concerns.
*   **Error Handling:** Implement robust error handling. Log errors appropriately using the `logging` module in Python. Provide user-friendly error feedback in the frontend.
*   **API URLs:** Strictly follow the rules defined in `.clinerules` for frontend API calls. Use the utilities in `frontend/services/api.ts`. Never hardcode `/api/v1`.
*   **Docker:** Adhere to the Docker rules in `.clinerules` (single container, bind mounts, `docker compose`).
*   **Documentation:** Add comments for complex logic. Write docstrings for Python functions/classes/modules. Document React components and hooks. Keep `docs/` updated.
*   **Logging:** Use structured logging in the backend where appropriate, providing context.

## Troubleshooting

*   **Permissions Issues (Docker Socket):** If Docker commands fail inside the container, ensure the Docker socket (`/var/run/docker.sock`) is correctly mounted and permissions allow the container user (root in dev) to access it.
*   **Hot Reloading Not Working:** Verify file mounting in `docker-compose.yml`. Ensure development servers (`uvicorn --reload`, `next dev`) are running correctly within the container. Check container logs.
*   **Dependency Conflicts:** Run `make clean` and `make docker-build` to rebuild the image and reinstall dependencies cleanly.