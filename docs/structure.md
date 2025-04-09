# Doogie Chat Bot - Detailed Project Structure

## Project Overview

Doogie Chat Bot is a hybrid RAG-based chatbot application with multi-user capabilities. The application uses a combination of BM25 and Vector Search (Annoy) for information retrieval and integrates with various LLM services. The project is structured as a full-stack application with a FastAPI backend and Next.js frontend, all containerized in a single Docker container.

## Complete Directory Structure

```
/DoogieBot/
├── .clinerules               # Project-specific rules
├── .dockerignore             # Docker ignore patterns
├── .env.example              # Environment variables template
├── .git/                     # Git repository
├── .github/                  # GitHub workflows
│   └── workflows/             # GitHub workflow definitions
│       └── ci-cd.yml          # CI/CD workflow configuration
├── .gitignore                # Git ignore patterns
├── .pylintrc                 # Python linter configuration
├── .venv/                    # Python virtual environment
├── Dockerfile                # Multi-stage Docker configuration
├── LICENSE                   # Project license
├── Makefile                  # Development task automation
├── README.md                 # Project documentation
├── backend/                  # Backend code (FastAPI)
│   ├── alembic/              # Database migrations
│   │   ├── versions/         # Migration scripts
│   │   ├── env.py            # Alembic environment configuration
│   │   └── script.py.mako    # Migration script template
│   ├── app/                  # Main application code
│   │   ├── api/              # API endpoints
│   │   │   ├── routes/       # Route handlers by feature
│   │   │   │   ├── auth.py   # Authentication routes
│   │   │   │   ├── chats.py  # Chat management routes
│   │   │   │   ├── documents.py # Document handling routes
│   │   │   │   ├── embedding.py # Embedding routes
│   │   │   │   ├── llm.py    # LLM configuration routes
│   │   │   │   ├── rag/      # RAG-specific routes
│   │   │   │   ├── rag.py    # Main RAG routes
│   │   │   │   ├── reranking.py # Reranking routes
│   │   │   │   ├── system.py # System management routes
│   │   │   │   ├── tags.py   # Tag management routes
│   │   │   │   └── users.py  # User management routes
│   │   │   └── api.py        # Main API router
│   │   ├── core/             # Core configuration
│   │   │   └── config.py     # Application settings
│   │   ├── db/               # Database code
│   │   │   └── base.py       # SQLAlchemy setup
│   │   ├── llm/              # LLM integration
│   │   │   ├── anthropic_client.py # Anthropic API client
│   │   │   ├── base.py       # Base LLM client interface
│   │   │   ├── factory.py    # LLM client factory
│   │   │   ├── google_gemini_client.py # Google Gemini API client
│   │   │   ├── ollama_client.py # Ollama API client
│   │   │   ├── openai_client.py # OpenAI API client
│   │   │   └── openrouter_client.py # OpenRouter API client
│   │   ├── models/           # Database models
│   │   │   ├── chat.py       # Chat and message models
│   │   │   ├── document.py   # Document model
│   │   │   ├── embedding_config.py # Embedding configuration model
│   │   │   ├── indexes.py    # Index models
│   │   │   ├── llm_config.py # LLM configuration model
│   │   │   ├── rag_config.py # RAG configuration model
│   │   │   ├── reranking_config.py # Reranking configuration model
│   │   │   ├── tag.py        # Tag model
│   │   │   └── user.py       # User model
│   │   ├── rag/              # RAG implementation
│   │   │   ├── bm25_index.py # BM25 search implementation
│   │   │   ├── document_chunker.py # Document chunking logic
│   │   │   ├── document_parser.py # Document parsing
│   │   │   ├── document_processor.py # Document processing pipeline
│   │   │   ├── faiss_store.py # Vector store implementation
│   │   │   ├── graph_interface.py # Graph interface
│   │   │   ├── graph_rag.py  # GraphRAG implementation
│   │   │   ├── graphrag/     # GraphRAG components
│   │   │   ├── hybrid_retriever.py # Hybrid search implementation
│   │   │   ├── networkx/     # NetworkX implementation
│   │   │   └── singleton.py  # RAG singleton pattern
│   │   ├── schemas/          # Pydantic schemas
│   │   │   ├── chat.py       # Chat schemas
│   │   │   ├── document.py   # Document schemas
│   │   │   ├── embedding.py  # Embedding schemas
│   │   │   ├── llm.py        # LLM schemas
│   │   │   ├── rag.py        # RAG schemas
│   │   │   ├── reranking.py  # Reranking schemas
│   │   │   ├── system.py     # System schemas
│   │   │   ├── tag.py        # Tag schemas
│   │   │   ├── token.py      # Authentication token schemas
│   │   │   └── user.py       # User schemas
│   │   ├── services/         # Business logic services
│   │   │   ├── chat.py       # Chat service
│   │   │   ├── document.py   # Document service
│   │   │   ├── embedding_config.py # Embedding configuration service
│   │   │   ├── llm.py        # LLM service
│   │   │   ├── llm_config.py # LLM configuration service
│   │   │   ├── rag_config.py # RAG configuration service
│   │   │   ├── reranking_config.py # Reranking configuration service
│   │   │   ├── system.py     # System service
│   │   │   ├── tag.py        # Tag service
│   │   │   ├── user.py       # User service
│   │   │   └── zip_processor.py # ZIP file processing service
│   │   └── utils/            # Utility functions
│   │       ├── deps.py       # Dependency injection utilities
│   │       ├── middleware.py # Custom middleware
│   │       └── security.py   # Security utilities
│   ├── tests/                # Test files
│   │   ├── api/              # API tests
│   │   └── selenium/         # Selenium tests
│   ├── alembic.ini           # Alembic configuration
│   ├── clear_users.py        # Utility to clear users
│   ├── create_llm_config.py  # Utility to create LLM config
│   ├── create_tag_tables.py  # Utility to create tag tables
│   ├── doogie.db             # SQLite database
│   ├── main.py               # Application entry point
│   ├── pyproject.toml        # Python project metadata
│   ├── requirements.txt      # Python dependencies
│   └── run_migrations.sh     # Migration script
├── docker-compose.prod.yml   # Production Docker Compose
├── docker-compose.yml        # Development Docker Compose
├── docs/                     # Project documentation
│   ├── projectoverview.md    # Project overview
│   └── structure.md          # This file
├── entrypoint.prod.sh        # Production entrypoint script
├── entrypoint.sh             # Development entrypoint script
├── frontend/                 # Frontend code (Next.js)
│   ├── .next/                # Next.js build directory
│   ├── .prettierrc           # Prettier configuration
│   ├── components/           # React components
│   │   ├── admin/            # Admin components
│   │   ├── chat/             # Chat components
│   │   │   ├── ChatInput.tsx     # Message input component
│   │   │   ├── ChatPage.tsx      # Main chat page layout
│   │   │   ├── ChatSidebar.tsx   # Sidebar for navigation
│   │   │   ├── DocumentReferences.tsx # Document references display
│   │   │   ├── FeedbackButton.tsx # Message feedback component
│   │   │   ├── ImprovedChatInput.tsx # Enhanced input component
│   │   │   ├── ImprovedMessageContent.tsx # Enhanced content display
│   │   │   ├── MarkdownEditor.tsx  # Markdown editing component
│   │   │   ├── MessageContent.tsx  # Message rendering component
│   │   │   ├── SearchBar.tsx       # Search component
│   │   │   ├── TagSearchBar.tsx    # Tag search component
│   │   │   └── TagSelector.tsx     # Tag selection component
│   │   ├── document/         # Document components
│   │   ├── layout/           # Layout components
│   │   └── ui/               # UI components
│   │       ├── Breadcrumbs.tsx   # Breadcrumb navigation
│   │       ├── Button.tsx        # Button component
│   │       ├── Card.tsx          # Card container
│   │       ├── ConfirmDialog.tsx # Confirmation dialog
│   │       ├── Dialog.tsx        # Modal dialog
│   │       ├── ErrorBoundary.tsx # Error handling
│   │       ├── Input.tsx         # Input component
│   │       ├── Spinner.tsx       # Loading spinner
│   │       ├── Tag.tsx           # Tag component
│   │       ├── Toast.tsx         # Notification toast
│   │       └── ... (other UI components)
│   ├── contexts/             # React contexts
│   │   ├── AuthContext.tsx   # Authentication context
│   │   ├── NotificationContext.tsx # Notification context
│   │   ├── OnboardingContext.tsx # Onboarding context
│   │   ├── ShortcutContext.tsx # Keyboard shortcuts context
│   │   └── ThemeContext.tsx  # Theme context
│   ├── hooks/                # Custom React hooks
│   ├── next-env.d.ts         # Next.js type declarations
│   ├── next.config.js        # Next.js configuration
│   ├── node_modules/         # Node.js dependencies
│   ├── package.json          # Node.js package configuration
│   ├── pages/                # Next.js pages
│   │   ├── _app.tsx          # Application wrapper
│   │   ├── admin/            # Admin pages
│   │   ├── chat/             # Chat pages
│   │   ├── index.tsx         # Home page
│   │   ├── login.tsx         # Login page
│   │   ├── profile.tsx       # User profile page
│   │   └── register.tsx      # Registration page
│   ├── public/               # Static assets
│   ├── services/             # API services
│   │   ├── api.ts            # API utilities
│   │   ├── auth.ts           # Authentication services
│   │   ├── chat.ts           # Chat services
│   │   ├── document.ts       # Document services
│   │   ├── llm.ts            # LLM services
│   │   ├── rag.ts            # RAG services
│   │   ├── system.ts         # System services
│   │   └── user.ts           # User services
│   ├── styles/               # CSS styles
│   ├── tailwind.config.js    # Tailwind CSS configuration
│   ├── tsconfig.json         # TypeScript configuration
│   ├── types/                # TypeScript type definitions
│   └── utils/                # Utility functions
│       ├── accessibilityUtils.ts # Accessibility utilities
│       ├── errorHandling.ts  # Error handling utilities
│       ├── exportUtils.ts    # Export utilities
│       └── ... (other utilities)
├── memory-bank/              # Project context and notes
│   ├── activeContext.md      # Active context information
│   ├── graphrag_implementation_plan.md # GraphRAG plan
│   ├── productContext.md     # Product context
│   ├── progress.md           # Project progress
│   ├── project_documentation.md # Project documentation
│   ├── projectbrief.md       # Project brief
│   ├── systemPatterns.md     # System patterns
│   └── techContext.md        # Technical context
├── scripts/                  # Utility scripts
│   ├── fix-all.sh             # Script to fix all environment issues
│   ├── fix-docker-compose.sh  # Script to fix Docker Compose formatting
│   └── fix-permissions.sh     # Script to fix permissions for Docker volumes
├── sync-doogie.sh            # Sync script
└── uploads/                  # File uploads directory
```

## Package Versions and Dependencies

### Backend Dependencies (Python 3.12+)

#### Web Framework
- **fastapi**: >=0.108.0
- **uvicorn[standard]**: >=0.30.0
- **aiohttp**: >=3.9.3

#### Database
- **sqlalchemy**: >=2.0.28
- **alembic**: >=1.15.1

#### Authentication
- **python-jose[cryptography]**: >=3.3.0
- **passlib[bcrypt]**: ==1.7.4 (pinned for stability)
- **bcrypt**: ==4.0.1 (downgraded for compatibility)
- **python-multipart**: >=0.0.9

#### Data Validation
- **pydantic**: >=2.10.6
- **pydantic-settings**: >=2.2.1
- **email-validator**: >=2.1.1

#### RAG Components
- **annoy**: >=1.17.3 (Alternative to faiss-cpu for vector search)
- **rank-bm25**: >=0.2.2
- **networkx**: >=3.2.1
- **sentence-transformers**: >=3.4.1
- **scikit-learn**: >=1.4.1

#### Document Processing
- **PyPDF2**: >=3.0.1
- **python-docx**: >=1.1.0
- **markdown**: >=3.6
- **python-frontmatter**: >=1.1.0
- **PyYAML**: >=6.0.1

#### Utilities
- **python-dotenv**: >=1.0.1
- **httpx**: >=0.27.0
- **tenacity**: >=8.3.0
- **loguru**: >=0.7.2
- **requests**: >=2.32.0
- **GitPython**: >=3.1.43

#### LLM Clients
- **anthropic**: >=0.21.3
- **google-generativeai**: >=0.4.0

### Frontend Dependencies (Node.js 20.x)

#### Framework and Core
- **next**: 15.2.3
- **react**: 19.0.0
- **react-dom**: 19.0.0
- **typescript**: 5.8.2

#### API and Data Fetching
- **axios**: ^1.7.1
- **@tanstack/react-query**: ^5.21.5
- **jwt-decode**: ^4.0.0

#### UI Components and Styling
- **@radix-ui/react-dialog**: ^1.0.5
- **class-variance-authority**: ^0.7.0
- **clsx**: ^2.1.0
- **tailwind-merge**: ^2.2.1
- **tailwindcss**: ^3.4.1

#### Forms and Validation
- **react-hook-form**: ^7.51.2
- **@hookform/resolvers**: ^3.3.4
- **zod**: ^3.22.4

#### Content Rendering
- **react-markdown**: ^9.0.1
- **remark-gfm**: ^4.0.0
- **rehype-raw**: ^7.0.0
- **react-syntax-highlighter**: ^15.6.1
- **prismjs**: ^1.29.0
- **react-window**: ^1.8.11

## Backend Structure

### Main Application Files

- **`main.py`**: Application entry point that:
  - Sets up the FastAPI application
  - Configures middleware (CORS, trailing slash)
  - Defines lifespan events for startup/shutdown
  - Initializes the application with the admin user and LLM configuration
  - Sets up error handlers for HTTP exceptions and general exceptions
  - Registers API routers

- **`alembic.ini`** and **`/alembic`**: Database migration configuration and migration scripts using Alembic
- **`run_migrations.sh`**: Shell script to run database migrations
- **`clear_users.py`**: Utility script to clear user data
- **`create_llm_config.py`**: Script to create default LLM configuration
- **`create_tag_tables.py`**: Script to set up tag-related database tables

### Core Modules

#### Config (/app/core)

- **`config.py`**: Application configuration using Pydantic Settings:
  - Defines application settings like API paths, security credentials, and default configurations
  - Sets up logging configuration
  - Handles environment variables
  - Contains default values for required settings

#### API Routes (/app/api)

- **`api.py`**: Main API router that includes all sub-routers
- **`/routes`**: Directory containing route handlers:
  - **`auth.py`**: Authentication endpoints (register, login, refresh token)
  - **`users.py`**: User management endpoints
  - **`chats.py`**: Chat-related endpoints
  - **`documents.py`**: Document management endpoints
  - **`rag.py`**: RAG-related endpoints
  - **`llm.py`**: LLM configuration endpoints
  - **`tags.py`**: Tag management endpoints
  - **`system.py`**: System-related endpoints
  - **`embedding.py`**: Embedding-related endpoints
  - **`reranking.py`**: Reranking-related endpoints

#### Database (/app/db)

- **`base.py`**: SQLAlchemy setup with:
  - Base class for models
  - Session management
  - Database connection utilities
- **`init_db.py`**: Database initialization functions

#### Models (/app/models)

SQLAlchemy ORM models defining the database schema:

- **`user.py`**: User model with:
  - User roles (USER, ADMIN)
  - User status (PENDING, ACTIVE, INACTIVE)
  - Password hashing
  - Timestamps for creation, update, last login
- **`chat.py`**: Chat and message models
- **`document.py`**: Document model for storing uploaded files
- **`embedding_config.py`**: Configuration for embedding models
- **`indexes.py`**: Index configuration for the RAG system
- **`llm_config.py`**: LLM provider configuration
- **`rag_config.py`**: RAG system configuration
- **`reranking_config.py`**: Reranking configuration
- **`tag.py`**: Tags for organizing content

#### Schemas (/app/schemas)

Pydantic schemas for request validation and response serialization:
- Request models
- Response models
- Internal data models
- Validation logic

#### Services (/app/services)

Business logic services:
- **`user.py`**: User management service with:
  - Authentication
  - User creation and modification
  - Password handling
- **`llm_config.py`**: LLM configuration service
- Document processing services
- RAG search services
- Chat functionality services

#### RAG System (/app/rag)

- **`singleton.py`**: RAG singleton pattern for shared resources
- BM25 implementation
- Annoy vector store integration
- Document processors
- GraphRAG utilities

#### LLM Integration (/app/llm)

- Provider-specific adapter classes:
  - OpenAI
  - Anthropic
  - OpenRouter
  - DeepSeek
  - Ollama
  - LM Studio
- Configuration classes
- Prompt management

#### Utilities (/app/utils)

- **`middleware.py`**: Custom middleware classes, including trailing slash handling
- **`security.py`**: Security utilities for authentication and tokens
- Error handling utilities
- Path and file handling utilities

### Database Schema

- **Users**: User accounts with roles and permissions
- **Chats**: Chat conversations
- **Messages**: Individual messages within chats
- **Documents**: Uploaded files and documents
- **Tags**: Organizational tags for chats and documents
- **LLMConfig**: Configuration for LLM providers
- **RAGConfig**: Configuration for the RAG system
- **EmbeddingConfig**: Configuration for embedding models
- **RerankingConfig**: Configuration for reranking

## Frontend Structure

### Core Files

- **`next.config.js`**: Next.js configuration
- **`package.json`**: Node.js dependencies and scripts
- **`tsconfig.json`**: TypeScript configuration
- **`tailwind.config.js`**: Tailwind CSS configuration

### Components

#### UI Components (/components/ui)

Reusable UI components:
- **`Button.tsx`**: Button component with various variants and states:
  - Variants: default, destructive, outline, secondary, ghost, link
  - Sizes: default, sm, lg, icon
  - Loading states with spinners
  - Accessibility attributes
- **`Input.tsx`**: Input component
- **`Card.tsx`**: Card container component
- **`Dialog.tsx`**: Modal dialog component
- **`Spinner.tsx`**: Loading spinner
- **`Toast.tsx`**: Notification toast
- **`Tag.tsx`**: Tag component for displaying tags
- Many other UI components

#### Chat Components (/components/chat)

- **`ChatInput.tsx`**: Message input component with:
  - Text input
  - File upload functionality
  - Message sending
  - Loading states
- **`ChatPage.tsx`**: Main chat page layout
- **`ChatSidebar.tsx`**: Sidebar for chat navigation
- **`MessageContent.tsx`**: Message rendering component
- **`FeedbackButton.tsx`**: UI for providing feedback on messages
- **`TagSelector.tsx`**: Component for selecting and managing tags

#### Document Components (/components/document)

- Document upload and management components
- Document preview components
- Document processing status indicators

#### Layout Components (/components/layout)

- Layout components for page structure
- Navigation components
- Header and footer components

#### Admin Components (/components/admin)

- Admin dashboard components
- User management components
- System configuration components

### Pages

Next.js pages following the file-based routing convention:
- **`index.tsx`**: Home page
- **`login.tsx`**: Login page
- **`chat/[id].tsx`**: Chat page
- **`admin/...`**: Admin dashboard pages
- **`api/...`**: API route handlers for server-side operations

### Services (/services)

API service clients that communicate with the backend:

- **`api.ts`**: Core API utilities for consistent URL handling:
  - **`getApiUrl()`**: Generates API URLs with proper prefixing
  - **`get()`**, **`post()`**, **`put()`**, **`del()`**: Standard HTTP methods
  - Token handling and refresh logic
  - Error handling

- **`auth.ts`**: Authentication services:
  - **`login()`**: User login
  - **`register()`**: User registration
  - **`refreshToken()`**: Token refresh
  - **`getCurrentUser()`**: Get current user information

- **`chat.ts`**: Chat-related services:
  - **`getChats()`**: Get all chats
  - **`getChat()`**: Get a single chat
  - **`createChat()`**: Create a new chat
  - **`updateChat()`**: Update chat details
  - **`deleteChat()`**: Delete a chat
  - **`sendMessage()`**: Send a message
  - **`streamMessage()`**: Stream a message from the LLM

- **`document.ts`**: Document management services
- **`llm.ts`**: LLM configuration services
- **`rag.ts`**: RAG-related services
- **`system.ts`**: System-related services
- **`user.ts`**: User management services

### Contexts (/contexts)

React context providers:
- Authentication context
- Chat context
- Theme context
- Notification context

### Hooks (/hooks)

Custom React hooks for:
- API data fetching
- Authentication
- Form handling
- Theming

### Types (/types)

TypeScript type definitions:
- API request and response types
- Component prop types
- State types
- Common interfaces

### Utils (/utils)

Utility functions:
- **`errorHandling.ts`**: Comprehensive error handling with:
  - Error categories (network, authentication, validation, etc.)
  - Standardized error objects
  - User-friendly error messages
  - Notification integration
- Form utilities
- Date formatting
- String formatting

## Docker Configuration

The project follows a single-container architecture with multi-stage Docker builds for efficiency.

### Dockerfile Stages

1. **base**: Base Python image with common dependencies
   - Python 3.12 slim
   - Essential build tools
   - UV package manager for faster Python dependency installation
   - Non-root user setup

2. **backend-builder**: Builds and installs backend dependencies
   - Python virtual environment
   - Backend dependencies installation

3. **frontend-builder**: Builds the frontend for production
   - Node.js 20.x installation
   - pnpm package manager
   - Frontend dependencies installation
   - Next.js build process

4. **test**: Configuration for running tests
   - Test-specific dependencies
   - Test entrypoint

5. **development**: Development configuration
   - Combined frontend and backend development environment
   - Auto-reload for both services
   - Development dependencies
   - Bind mounts for local development

6. **production**: Production-ready image
   - Optimized builds
   - Minimal dependencies
   - Health checks
   - Production entrypoint

### Docker Compose Configuration

- **`docker-compose.yml`**: Development configuration
  - Bind mounts local directories for auto-reloading
  - Sets up development environment variables
  - Configures ports for frontend (3000) and backend (8000)
  - Health check configuration

- **`docker-compose.prod.yml`**: Production configuration
  - Optimized for production deployment
  - Environment variable handling
  - Volume management
  - Health check configuration

### Entrypoint Scripts

- **`entrypoint.sh`**: Development entrypoint that:
  - Runs database migrations
  - Starts the backend FastAPI server with hot reload
  - Starts the frontend Next.js server with hot reload
  - Sets up proper signal handling for clean shutdown

- **`entrypoint.prod.sh`**: Production entrypoint that:
  - Runs database migrations
  - Starts the backend with multiple workers
  - Starts the frontend in production mode
  - Handles proper process management and signals

## API Endpoints

### Authentication Endpoints

- **POST /api/v1/auth/register**: Register a new user
- **POST /api/v1/auth/login**: Login with username and password
- **POST /api/v1/auth/refresh**: Refresh access token

### User Endpoints

- **GET /api/v1/users/me**: Get current user information
- **PUT /api/v1/users/me**: Update current user
- **GET /api/v1/users**: Get all users (admin only)
- **POST /api/v1/users**: Create a new user (admin only)
- **GET /api/v1/users/{user_id}**: Get user by ID (admin only)
- **PUT /api/v1/users/{user_id}**: Update user (admin only)
- **DELETE /api/v1/users/{user_id}**: Delete user (admin only)

### Chat Endpoints

- **GET /api/v1/chats**: Get all chats for current user
- **POST /api/v1/chats**: Create a new chat
- **GET /api/v1/chats/{chat_id}**: Get chat by ID
- **PUT /api/v1/chats/{chat_id}**: Update chat
- **DELETE /api/v1/chats/{chat_id}**: Delete chat
- **POST /api/v1/chats/{chat_id}/messages**: Add a message to a chat
- **POST /api/v1/chats/{chat_id}/llm**: Send a message to the LLM
- **GET /api/v1/chats/{chat_id}/stream**: Stream a response from the LLM

### Document Endpoints

- **GET /api/v1/documents**: Get all documents
- **POST /api/v1/documents**: Upload a document
- **GET /api/v1/documents/{document_id}**: Get document by ID
- **DELETE /api/v1/documents/{document_id}**: Delete document

### RAG Endpoints

- **POST /api/v1/rag/query**: Query the RAG system
- **GET /api/v1/rag/status**: Get RAG system status
- **POST /api/v1/rag/reindex**: Rebuild the RAG indexes

### LLM Endpoints

- **GET /api/v1/llm/config**: Get LLM configuration
- **PUT /api/v1/llm/config**: Update LLM configuration
- **GET /api/v1/llm/providers**: Get available LLM providers
- **GET /api/v1/llm/models**: Get available models for a provider

### Tag Endpoints

- **GET /api/v1/tags**: Get all tags for current user
- **POST /api/v1/tags**: Create a new tag
- **PUT /api/v1/tags/{tag_id}**: Update tag
- **DELETE /api/v1/tags/{tag_id}**: Delete tag
- **PUT /api/v1/tags/chats/{chat_id}/tags**: Update tags for a chat

## Development Workflow

The project includes a comprehensive Makefile with commands for common development tasks:

### Building
- **`make all`**: Install dependencies, run linters, and tests
- **`make clean`**: Clean up build artifacts
- **`make install`**: Install backend and frontend dependencies
- **`make dev`**: Start development environment with Docker
- **`make docker-build`**: Build Docker image

### Testing
- **`make test`**: Run tests locally
- **`make docker-test`**: Run tests in Docker container

### Code Quality
- **`make lint`**: Run linters locally
- **`make docker-lint`**: Run linters in Docker container
- **`make format`**: Format code locally
- **`make docker-format`**: Format code in Docker container
- **`make security-check`**: Run security checks locally
- **`make docker-security`**: Run security checks in Docker container

### Deployment
- **`make docker-up`**: Start Docker container in development mode
- **`make docker-up-prod`**: Start Docker container in production mode
- **`make docker-down`**: Stop Docker container

### Database
- **`make migrate`**: Run database migrations

## Security Features

- **Authentication**: JWT-based authentication with access and refresh tokens
- **Password Hashing**: Secure password storage using bcrypt
- **Role-Based Access Control**: Different permissions for users and admins
- **API Key Security**: Environment variable-based API key storage
- **CORS Protection**: Configurable CORS settings
- **Input Validation**: Comprehensive request validation using Pydantic
- **Error Handling**: Consistent error responses with proper status codes
- **Docker Security**: Non-root user in container, health checks
