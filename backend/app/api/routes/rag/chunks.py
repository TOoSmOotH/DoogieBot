from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel

from app.db.base import get_db
from app.models.user import User
from app.models.document import DocumentChunk, Document
from app.models.graph import GraphNode, GraphEdge
from app.rag.singleton import rag_singleton
from app.utils.deps import get_current_admin_user, get_current_user

router = APIRouter()

# Define response model for chunk info
class ChunkInfoResponse(BaseModel):
    chunk_id: str
    chunk_index: int
    document_id: str
    document_title: str
    document_type: str
    document_filename: str
    created_at: str
    
    class Config:
        from_attributes = True

@router.delete("/chunks", response_model=dict)
async def delete_all_chunks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Delete all document chunks. Admin only.
    """
    # Delete all chunks
    db.query(DocumentChunk).delete()
    db.commit()
    
    return {
        "status": "success",
        "message": "All document chunks deleted"
    }

@router.post("/reset", response_model=dict)
async def reset_rag_system(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Reset the RAG system. This will clear all indexes and delete all document chunks. Admin only.
    """
    # Clear all indexes
    rag_singleton.clear_all()
    
    # Delete all chunks
    db.query(DocumentChunk).delete()
    
    # Delete all graph nodes and edges
    db.query(GraphEdge).delete()
    db.query(GraphNode).delete()
    
    db.commit()
    
    return {
        "status": "success",
        "message": "RAG system reset successfully"
    }

@router.get("/chunks/{chunk_id}", response_model=ChunkInfoResponse)
async def get_chunk_info(
    chunk_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user), # Regular user access allowed
) -> Any:
    """
    Get information about a document chunk, including its source document.
    """
    try:
        # Query for the chunk with join to document
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk with id {chunk_id} not found"
            )
        
        # Get the related document
        document = db.query(Document).filter(Document.id == chunk.document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with id {chunk.document_id} not found"
            )
        
        # Create the response object
        return {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "document_id": document.id,
            "document_title": document.title,
            "document_type": document.type,
            "document_filename": document.filename,
            "created_at": chunk.created_at.isoformat() if chunk.created_at else ""  
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error retrieving chunk info for {chunk_id}: {str(e)}")
        
        # Return a 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chunk info: {str(e)}"
        )