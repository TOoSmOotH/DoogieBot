"""
Main API router for the application.

This module collects all API routes and combines them into a single router.
"""

from fastapi import APIRouter
from app.api.routes import auth, users, chats, documents, rag, llm, tags, system, embedding, reranking, docker, mcp # Added docker
# Create the main API router
api_router = APIRouter()

# Health check endpoint for API V1
@api_router.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy"}

# Include routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(chats.router, prefix="/chats", tags=["chats"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(tags.router, prefix="/tags", tags=["tags"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(embedding.router, prefix="/embedding", tags=["embedding"])
api_router.include_router(reranking.router, prefix="/reranking", tags=["reranking"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
api_router.include_router(docker.router, prefix="/docker", tags=["docker"]) # Added docker router
