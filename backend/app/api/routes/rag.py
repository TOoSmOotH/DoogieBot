import os
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.base import get_db
from app.models.document import Document, DocumentChunk, GraphEdge, GraphNode
from app.models.user import User
from app.rag.bm25_index import BM25Index
from app.rag.faiss_store import FAISSStore
from app.rag.graph_rag import GraphRAG
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.singleton import rag_singleton
from app.schemas.rag import (RAGBuildOptions, RAGComponentToggle,
                             RAGRetrieveOptions)
from app.services.document import DocumentService
from app.services.rag_config import RAGConfigService
from app.utils.deps import get_current_admin_user, get_current_user
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/stats", response_model=Dict[str, Any])
async def get_rag_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get RAG statistics - alias for get_rag_status.
    """
    return await get_rag_status(db, current_user)


@router.get("/status", response_model=Dict[str, Any])
async def get_rag_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get the status of the RAG system.
    """
    # Get document and chunk counts
    document_count = db.query(Document).count()
    chunk_count = db.query(DocumentChunk).count()

    # Get last updated timestamp
    latest_chunk = (
        db.query(DocumentChunk).order_by(DocumentChunk.created_at.desc()).first()
    )
    last_updated = latest_chunk.created_at if latest_chunk else None

    # Ensure singleton is initialized
    rag_singleton.initialize()

    # Get references to singleton instances
    bm25_index = rag_singleton.get_bm25_index()
    faiss_store = rag_singleton.get_faiss_store()
    graph_rag = rag_singleton.get_graph_rag()

    # Get counts from the singleton instances
    bm25_doc_count = len(bm25_index.doc_ids) if bm25_index else 0
    faiss_doc_count = len(faiss_store.doc_ids) if faiss_store else 0
    graph_node_count = len(graph_rag.graph.nodes) if graph_rag else 0
    graph_edge_count = len(graph_rag.graph.edges) if graph_rag else 0

    # Get RAG component configuration
    rag_config = RAGConfigService.get_config(db)

    return {
        "document_count": document_count,
        "chunk_count": chunk_count,
        "last_updated": last_updated,
        "bm25_status": {
            "enabled": rag_config.bm25_enabled,
            "document_count": bm25_doc_count,
            "last_indexed": None,  # Would need to store this in the index
            "status": "idle",
        },
        "faiss_status": {
            "enabled": rag_config.faiss_enabled,
            "document_count": faiss_doc_count,
            "last_indexed": None,  # Would need to store this in the index
            "status": "idle",
        },
        "graph_status": {
            "enabled": rag_config.graph_enabled,
            "document_count": graph_node_count,  # Use node count as document count
            "node_count": graph_node_count,
            "edge_count": graph_edge_count,
            "last_indexed": None,  # Would need to store this in the index
            "status": "idle",
        },
    }


@router.get("/graph/analyze", response_model=Dict[str, Any])
async def analyze_graph(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Analyze the graph structure and return statistics.
    """
    # Get singleton instance
    graph_rag = rag_singleton.get_graph_rag()

    # Check if graph is loaded
    if not graph_rag or len(graph_rag.graph.nodes) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found or empty",
        )

    # Analyze graph
    analysis = graph_rag.analyze_graph()

    return analysis


@router.get("/graph/important-nodes", response_model=List[Dict[str, Any]])
async def get_important_nodes(
    method: str = "pagerank",
    top_n: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get the most important nodes in the graph using various centrality measures.

    Args:
        method: Centrality method to use ('pagerank', 'betweenness', 'degree', 'eigenvector')
        top_n: Number of top nodes to return
    """
    # Validate method
    valid_methods = ["pagerank", "betweenness", "degree", "eigenvector"]
    if method not in valid_methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid method. Must be one of: {', '.join(valid_methods)}",
        )

    # Get singleton instance
    graph_rag = rag_singleton.get_graph_rag()

    # Check if graph is loaded
    if not graph_rag or len(graph_rag.graph.nodes) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found or empty",
        )

    # Get important nodes
    important_nodes = graph_rag.get_important_nodes(top_n=top_n, method=method)

    return important_nodes


@router.post("/toggle-component", response_model=Dict[str, Any])
async def toggle_rag_component(
    toggle_data: RAGComponentToggle,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Toggle a RAG component on or off. Admin only.
    """
    # Validate component
    if toggle_data.component not in ["bm25", "faiss", "graph"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid component: {toggle_data.component}. Must be one of: bm25, faiss, graph",
        )

    # Update component status in database
    config = RAGConfigService.update_component_status(
        db, toggle_data.component, toggle_data.enabled
    )

    if not config:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update {toggle_data.component} status",
        )

    return {
        "status": "success",
        "component": toggle_data.component,
        "enabled": toggle_data.enabled,
    }


@router.post("/build-indexes", response_model=Dict[str, Any])
async def build_indexes(
    build_options: RAGBuildOptions,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Build or rebuild RAG indexes. Admin only.

    Args:
        build_options: Options for building indexes
    """
    # Get RAG configuration from database
    rag_config = RAGConfigService.get_config(db)

    # Override configuration with request parameters if provided
    # Otherwise use the database configuration
    use_bm25 = build_options.use_bm25 and rag_config.bm25_enabled
    use_faiss = build_options.use_faiss and rag_config.faiss_enabled
    use_graph = build_options.use_graph and rag_config.graph_enabled
    rebuild = build_options.rebuild
    batch_size = build_options.batch_size
    # Variables are now set above
    import asyncio
    import logging

    from app.core.config import settings

    logger = logging.getLogger(__name__)

    # Check if there are any document chunks to index
    chunk_count = db.query(DocumentChunk).count()
    if chunk_count == 0:
        return {
            "status": "warning",
            "message": "No document chunks found to index. Please upload and process documents first.",
            "document_count": 0,
            "chunk_count": 0,
        }

    # Create retriever
    retriever = HybridRetriever(
        db, use_bm25=use_bm25, use_faiss=use_faiss, use_graph=use_graph
    )

    # Build indexes in background with timeout
    async def build():
        try:
            # Set a timeout for the build process
            timeout = settings.RAG_INDEX_BUILD_TIMEOUT
            logger.info(f"Starting index building with timeout of {timeout} seconds")

            # Use asyncio.wait_for to set a timeout
            result = await asyncio.wait_for(
                retriever.build_indexes(rebuild=rebuild), timeout=timeout
            )

            logger.info(f"Index building completed successfully: {result}")

            # Store the result in a database or cache for later retrieval
            # For now, we just log it
            print(f"Index building completed: {result}")

        except asyncio.TimeoutError:
            logger.error(f"Index building timed out after {timeout} seconds")
            print(f"Index building timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error building indexes: {str(e)}")
            print(f"Error building indexes: {str(e)}")

    # Add the task to the background tasks
    background_tasks.add_task(build)

    # Estimate completion time based on chunk count
    # This is a rough estimate - actual time will vary based on hardware and data
    estimated_minutes = max(
        5, int(chunk_count / 100)
    )  # Rough estimate: 100 chunks per minute

    return {
        "status": "started",
        "message": f"Index building started in the background with {chunk_count} document chunks. This may take {estimated_minutes} minutes or more to complete.",
        "rebuild": rebuild,
        "chunk_count": chunk_count,
        "estimated_completion_minutes": estimated_minutes,
        "timeout_seconds": settings.RAG_INDEX_BUILD_TIMEOUT,
        "indexes": {
            "bm25": {
                "enabled": use_bm25,
                "status": "indexing" if use_bm25 else "skipped",
            },
            "faiss": {
                "enabled": use_faiss,
                "status": "indexing" if use_faiss else "skipped",
            },
            "graph": {
                "enabled": use_graph,
                "status": "indexing" if use_graph else "skipped",
            },
        },
    }


@router.post("/retrieve", response_model=List[dict])
async def retrieve(
    options: RAGRetrieveOptions,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Retrieve relevant documents using the hybrid approach.

    Args:
        options: Retrieval options
    """
    # Get RAG configuration from database
    rag_config = RAGConfigService.get_config(db)
    # Check if there are any document chunks to search
    chunk_count = db.query(DocumentChunk).count()
    if chunk_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No document chunks found to search. Please upload and process documents first.",
        )

    # Create retriever
    retriever = HybridRetriever(
        db,
        use_bm25=options.use_bm25,
        use_faiss=options.use_faiss,
        use_graph=options.use_graph,
    )

    # Retrieve documents
    results = await retriever.retrieve(
        query=options.query,
        query_embedding=options.query_embedding,
        top_k=options.top_k,
        rerank=options.rerank,
        fast_mode=options.fast_mode,
    )

    return results


@router.get("/graph/nodes/{node_id}", response_model=dict)
async def get_graph_node(
    node_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get a graph node by ID.
    """
    # Get singleton instance
    graph = rag_singleton.get_graph_rag()

    # Check if graph is loaded
    if not graph or len(graph.graph.nodes) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found or empty",
        )

    node = graph.get_node(node_id)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found",
        )

    return node


@router.get("/graph/nodes/{node_id}/neighbors", response_model=List[dict])
async def get_node_neighbors(
    node_id: str,
    relation_type: Optional[str] = None,
    max_depth: int = 1,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get neighbors of a graph node.
    """
    # Get singleton instance
    graph = rag_singleton.get_graph_rag()

    # Check if graph is loaded
    if not graph or len(graph.graph.nodes) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found or empty",
        )

    neighbors = graph.get_neighbors(
        node_id=node_id, relation_type=relation_type, max_depth=max_depth
    )

    return neighbors


@router.get("/chunks/{chunk_id}", response_model=Dict[str, Any])
async def get_chunk_info(
    chunk_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get information about a document chunk, including its document ID and title.
    """
    # Get the chunk
    chunk = DocumentService.get_chunk(db, chunk_id)
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found",
        )

    # Get the document
    document = DocumentService.get_document(db, chunk.document_id)

    # Return chunk info with document details
    return {
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "document_id": chunk.document_id,
        "document_title": document.title if document else "Unknown document",
        "document_type": document.type if document else None,
        "document_filename": document.filename if document else None,
        "created_at": chunk.created_at,
    }


@router.post("/graph/search", response_model=List[dict])
async def search_graph(
    query: str,
    node_types: Optional[List[str]] = None,
    relation_types: Optional[List[str]] = None,
    max_results: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Search the graph for nodes matching the query.
    """
    # Get singleton instance
    graph = rag_singleton.get_graph_rag()

    # Check if graph is loaded
    if not graph or len(graph.graph.nodes) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found or empty",
        )

    results = graph.search(
        query=query,
        node_types=node_types,
        relation_types=relation_types,
        max_results=max_results,
    )

    return results


@router.post("/graph/rebuild", response_model=dict)
async def rebuild_graph(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Rebuild the graph from database. Admin only.
    """
    import asyncio
    import logging

    from app.core.config import settings

    logger = logging.getLogger(__name__)

    # Check if there are any document chunks to use for graph building
    chunk_count = db.query(DocumentChunk).count()
    if chunk_count == 0:
        return {
            "status": "warning",
            "message": "No document chunks found to build graph. Please upload and process documents first.",
            "chunk_count": 0,
        }

    graph = GraphRAG()

    # Rebuild graph in background with timeout
    async def rebuild():
        try:
            # Set a timeout for the build process
            timeout = settings.RAG_INDEX_BUILD_TIMEOUT
            logger.info(f"Starting graph rebuilding with timeout of {timeout} seconds")

            # Use asyncio.wait_for to set a timeout
            async def build_and_save():
                nodes, edges = graph.build_from_database(db)
                graph.save()
                return nodes, edges

            nodes, edges = await asyncio.wait_for(build_and_save(), timeout=timeout)

            logger.info(
                f"Graph rebuilding completed successfully with {nodes} nodes and {edges} edges"
            )

        except asyncio.TimeoutError:
            logger.error(f"Graph rebuilding timed out after {timeout} seconds")
            print(f"Graph rebuilding timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error rebuilding graph: {str(e)}")
            print(f"Error rebuilding graph: {str(e)}")

    # Add the task to the background tasks
    background_tasks.add_task(rebuild)

    # Estimate completion time based on chunk count
    # This is a rough estimate - actual time will vary based on hardware and data
    estimated_minutes = max(
        5, int(chunk_count / 50)
    )  # Graph building is more intensive

    return {
        "status": "started",
        "message": f"Graph rebuilding started in the background with {chunk_count} document chunks. This may take {estimated_minutes} minutes or more to complete.",
        "chunk_count": chunk_count,
        "estimated_completion_minutes": estimated_minutes,
        "timeout_seconds": settings.RAG_INDEX_BUILD_TIMEOUT,
    }


@router.post("/graph/save-to-database", response_model=dict)
async def save_graph_to_database(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Save the graph to the database. Admin only.
    """
    # Get singleton instance
    graph = rag_singleton.get_graph_rag()

    # Check if graph is loaded
    if not graph or len(graph.graph.nodes) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph not found or empty",
        )

    nodes, edges = graph.save_to_database(db)

    return {"status": "success", "nodes": nodes, "edges": edges}


@router.post("/delete-all-chunks", response_model=Dict[str, Any])
async def delete_all_chunks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Delete all document chunks from the database. Admin only.
    """
    try:
        num_deleted = DocumentService.delete_all_chunks(db)
        return {
            "status": "success",
            "message": f"Successfully deleted {num_deleted} document chunks",
            "chunks_deleted": num_deleted,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chunks: {str(e)}",
        )


@router.post("/reset", response_model=Dict[str, Any])
async def reset_rag_system(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Completely reset the RAG system by deleting all index files. Admin only.
    """
    try:
        # Set up index paths
        index_dir = "./indexes"
        os.makedirs(index_dir, exist_ok=True)

        # Delete index files
        index_files = [
            os.path.join(index_dir, "bm25_index.pkl"),
            os.path.join(index_dir, "faiss_index.pkl"),
            os.path.join(index_dir, "faiss_index.pkl.bin"),
            os.path.join(index_dir, "graph_rag.pkl"),
        ]

        deleted_files = []
        for file_path in index_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(os.path.basename(file_path))

        # Clear singleton instances
        rag_singleton.clear_all()

        return {
            "status": "success",
            "message": f"Successfully reset RAG system. Deleted index files: {', '.join(deleted_files)}",
            "deleted_files": deleted_files,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset RAG system: {str(e)}",
        )
