import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.models.document import DocumentChunk
from app.rag.bm25_index import BM25Index
from app.rag.faiss_store import FAISSStore
from app.rag.graph_rag import GraphRAG
from app.rag.singleton import rag_singleton
from app.services.document import DocumentService
from app.services.rag_config import RAGConfigService
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that combines BM25, FAISS, and GraphRAG.
    """

    def __init__(
        self,
        db: Session,
        index_dir: str = "./indexes",
        use_bm25: bool = True,
        use_faiss: bool = True,
        use_graph: bool = True,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            db: Database session
            index_dir: Directory to store indexes
            use_bm25: Whether to use BM25 for retrieval
            use_faiss: Whether to use FAISS for retrieval
            use_graph: Whether to use GraphRAG for retrieval
        """
        self.db = db
        self.index_dir = index_dir
        self.use_bm25 = use_bm25
        self.use_faiss = use_faiss
        self.use_graph = use_graph

        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)

        # We don't initialize the singleton here anymore
        # It's initialized in the main.py startup event
        # We'll just use the singleton when needed

        # We'll use properties to lazily load these components
        self._bm25_index = None
        self._faiss_store = None
        self._graph_rag = None

    @property
    def bm25_index(self):
        """Lazy-loaded BM25 index"""
        if self._bm25_index is None and self.use_bm25:
            self._bm25_index = rag_singleton.get_bm25_index()
        return self._bm25_index

    @property
    def faiss_store(self):
        """Lazy-loaded FAISS store"""
        if self._faiss_store is None and self.use_faiss:
            self._faiss_store = rag_singleton.get_faiss_store()
        return self._faiss_store

    @property
    def graph_rag(self):
        """Lazy-loaded GraphRAG"""
        if self._graph_rag is None and self.use_graph:
            self._graph_rag = rag_singleton.get_graph_rag()
        return self._graph_rag

    async def build_indexes(self, rebuild: bool = False) -> Dict[str, Any]:
        """
        Build or rebuild all indexes.

        Args:
            rebuild: Whether to rebuild existing indexes

        Returns:
            Dictionary with build results
        """
        results = {
            "bm25": {"status": "skipped", "documents": 0},
            "faiss": {"status": "skipped", "documents": 0},
            "graph": {"status": "skipped", "nodes": 0, "edges": 0},
        }

        # Get RAG configuration from database
        rag_config = RAGConfigService.get_config(self.db)

        # Use configuration to determine which components to build
        use_bm25 = self.use_bm25 and rag_config.bm25_enabled
        use_faiss = self.use_faiss and rag_config.faiss_enabled
        use_graph = self.use_graph and rag_config.graph_enabled

        # Clear indexes if rebuilding
        if rebuild:
            if use_bm25:
                self.bm25_index.clear()

            if use_faiss:
                self.faiss_store.clear()

            if use_graph:
                self.graph_rag.clear()
                self.graph_rag.clear()

        # Get all document chunks
        chunks = self.db.query(DocumentChunk).all()

        # Build BM25 index
        if use_bm25:
            try:
                logger.info("Building BM25 index...")

                documents = []
                for chunk in chunks:
                    documents.append({"id": chunk.id, "content": chunk.content})

                if not documents:
                    logger.warning("No documents found for BM25 indexing")
                    results["bm25"] = {
                        "status": "warning",
                        "message": "No documents found to index",
                        "documents": 0,
                    }
                else:
                    self.bm25_index.add_documents(documents)
                    self.bm25_index.save()

                    results["bm25"] = {"status": "success", "documents": len(documents)}

                    logger.info(f"BM25 index built with {len(documents)} documents")
            except Exception as e:
                logger.error(f"Error building BM25 index: {str(e)}")
                results["bm25"] = {"status": "error", "message": str(e)}

        # Build FAISS index
        if use_faiss:
            try:
                logger.info("Building FAISS index...")

                # Log total number of chunks
                logger.info(f"Total chunks in database: {len(chunks)}")

                documents = []
                chunks_without_embeddings = 0

                for chunk in chunks:
                    # Skip chunks without embeddings
                    if not chunk.embedding:
                        chunks_without_embeddings += 1
                        continue

                    # Parse embedding from JSON
                    try:
                        logger.debug(
                            f"Parsing embedding for chunk {chunk.id}: {chunk.embedding[:30]}..."
                        )
                        embedding = json.loads(chunk.embedding)

                        # Update FAISS store dimension based on first embedding
                        if len(documents) == 0:
                            self.faiss_store.dimension = len(embedding)
                            logger.info(f"Setting FAISS dimension to {len(embedding)}")

                        documents.append(
                            {
                                "id": chunk.id,
                                "embedding": embedding,
                                "content": chunk.content,
                                "metadata": chunk.meta_data,
                            }
                        )
                    except Exception as parse_error:
                        logger.warning(
                            f"Could not parse embedding for chunk {chunk.id}: {str(parse_error)}"
                        )

                # Log detailed information about chunks and embeddings
                logger.info(
                    f"Found {chunks_without_embeddings} chunks without embeddings"
                )
                logger.info(
                    f"Successfully parsed embeddings for {len(documents)} chunks"
                )

                if not documents:
                    logger.warning(
                        "No documents with embeddings found for FAISS indexing"
                    )
                    results["faiss"] = {
                        "status": "warning",
                        "message": f"No documents with embeddings found to index. {chunks_without_embeddings} chunks exist without embeddings.",
                        "documents": 0,
                        "chunks_without_embeddings": chunks_without_embeddings,
                    }
                else:
                    self.faiss_store.add_embeddings(documents)
                    self.faiss_store.save()

                    results["faiss"] = {
                        "status": "success",
                        "documents": len(documents),
                    }

                    logger.info(f"FAISS index built with {len(documents)} documents")
            except Exception as e:
                logger.error(f"Error building FAISS index: {str(e)}")
                results["faiss"] = {"status": "error", "message": str(e)}

        # Build GraphRAG
        if use_graph:
            try:
                logger.info("Building GraphRAG...")

                # Build from database
                nodes, edges = self.graph_rag.build_from_database(self.db)
                self.graph_rag.save()

                results["graph"] = {"status": "success", "nodes": nodes, "edges": edges}

                logger.info(f"GraphRAG built with {nodes} nodes and {edges} edges")
            except Exception as e:
                logger.error(f"Error building GraphRAG: {str(e)}")
                results["graph"] = {"status": "error", "message": str(e)}

        return results

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        use_bm25: Optional[bool] = None,
        use_faiss: Optional[bool] = None,
        use_graph: Optional[bool] = None,
        rerank: bool = True,
        fast_mode: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the hybrid approach.

        Args:
            query: Query string
            query_embedding: Optional query embedding for vector search
            top_k: Number of results to return
            use_bm25: Whether to use BM25 (overrides instance setting)
            use_faiss: Whether to use FAISS (overrides instance setting)
            use_graph: Whether to use GraphRAG (overrides instance setting)
            rerank: Whether to rerank results
            fast_mode: Whether to use fast mode for large graphs

        Returns:
            List of relevant documents
        """
        import time

        start_time = time.time()

        # Get RAG configuration from database
        rag_config = RAGConfigService.get_config(self.db)

        # Use instance settings if not specified, otherwise use the provided values
        # If not provided, use the database configuration
        use_bm25 = (
            use_bm25
            if use_bm25 is not None
            else (self.use_bm25 and rag_config.bm25_enabled)
        )
        use_faiss = (
            use_faiss
            if use_faiss is not None
            else (self.use_faiss and rag_config.faiss_enabled)
        )
        use_graph = (
            use_graph
            if use_graph is not None
            else (self.use_graph and rag_config.graph_enabled)
        )

        # Initialize results
        all_results = []

        # BM25 retrieval
        bm25_results = []
        if use_bm25:
            bm25_start = time.time()
            try:
                bm25_results = self.bm25_index.search(query, top_k=top_k)
                for result in bm25_results:
                    result["source"] = "bm25"
                all_results.extend(bm25_results)
                logger.info(
                    f"BM25 retrieval completed in {time.time() - bm25_start:.2f}s, found {len(bm25_results)} results"
                )
            except Exception as e:
                logger.error(f"Error in BM25 retrieval: {str(e)}")

        # FAISS retrieval
        faiss_results = []
        if use_faiss and query_embedding:
            faiss_start = time.time()
            try:
                faiss_results = self.faiss_store.search(query_embedding, top_k=top_k)
                for result in faiss_results:
                    result["source"] = "faiss"
                all_results.extend(faiss_results)
                logger.info(
                    f"FAISS retrieval completed in {time.time() - faiss_start:.2f}s, found {len(faiss_results)} results"
                )
            except Exception as e:
                logger.error(f"Error in FAISS retrieval: {str(e)}")

        # GraphRAG retrieval - completely skip if we already have enough results from other methods
        graph_results = []
        if use_graph and len(all_results) < top_k:
            graph_start = time.time()
            try:
                # Use a simpler, more reliable approach for graph search
                # Instead of using the complex graph search, we'll do a simple keyword search on the graph nodes
                logger.info("Using simplified graph search to avoid memory issues")

                # Get a sample of nodes from the graph
                import random

                try:
                    # Get the total number of nodes
                    node_count = len(self.graph_rag.graph.nodes)

                    # If we have too many nodes, sample a subset
                    max_nodes_to_check = 200
                    if node_count > max_nodes_to_check:
                        # Sample random nodes
                        sampled_nodes = random.sample(
                            list(self.graph_rag.graph.nodes), max_nodes_to_check
                        )
                    else:
                        # Use all nodes
                        sampled_nodes = list(self.graph_rag.graph.nodes)

                    # Simple keyword matching
                    query_lower = query.lower()
                    query_terms = query_lower.split()

                    # Check each node for keyword matches
                    for node_id in sampled_nodes:
                        node_data = self.graph_rag.graph.nodes[node_id]
                        content = node_data.get("content", "").lower()

                        # Check if any query term is in the content
                        if any(term in content for term in query_terms):
                            # Calculate a simple score based on term frequency
                            score = sum(
                                content.count(term) for term in query_terms
                            ) / len(query_terms)

                            # Add to results
                            graph_results.append(
                                {
                                    "id": node_id,
                                    "content": node_data.get("content"),
                                    "type": node_data.get("type"),
                                    "score": score,
                                    "metadata": node_data.get("metadata", {}),
                                    "source": "graph",
                                }
                            )

                    # Sort by score and limit
                    graph_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                    graph_results = graph_results[:top_k]

                except Exception as graph_error:
                    logger.error(
                        f"Error in simplified graph search: {str(graph_error)}"
                    )
                    # Continue with empty results

                # Process results if we got any
                if graph_results:
                    all_results.extend(graph_results)
                    logger.info(
                        f"Simplified graph retrieval completed in {time.time() - graph_start:.2f}s, found {len(graph_results)} results"
                    )
                else:
                    logger.warning("No results from simplified graph search")
            except Exception as e:
                logger.error(f"Error in graph retrieval: {str(e)}")
                # Continue with other results even if graph search fails

        # Rerank results if needed
        if rerank and all_results:
            rerank_start = time.time()

            # Simple reranking by score
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Remove duplicates (keep highest score)
            seen_ids = set()
            unique_results = []

            for result in all_results:
                result_id = result.get("id")
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)

            all_results = unique_results
            logger.info(
                f"Reranking completed in {time.time() - rerank_start:.2f}s, {len(all_results)} unique results"
            )

        # Limit to top_k
        final_results = all_results[:top_k]

        # Log source distribution in final results
        source_counts = {}
        for result in final_results:
            source = result.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        total_time = time.time() - start_time
        logger.info(
            f"Total retrieval completed in {total_time:.2f}s. Final results: {len(final_results)} ({source_counts})"
        )

        return final_results

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Document chunk or None if not found
        """
        chunk = DocumentService.get_chunk(self.db, chunk_id)
        if not chunk:
            return None

        # Get document
        document = DocumentService.get_document(self.db, chunk.document_id)

        return {
            "id": chunk.id,
            "content": chunk.content,
            "document_id": chunk.document_id,
            "document_title": document.title if document else None,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.meta_data,
        }
