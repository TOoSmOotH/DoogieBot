import logging
import os
from typing import Optional

from app.rag.bm25_index import BM25Index
from app.rag.faiss_store import FAISSStore
from app.rag.graph_rag import GraphRAG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSingleton:
    """
    Singleton class to hold RAG components in memory.
    """

    # Class variables for the singleton pattern
    _instance = None
    _lock = False

    # Singleton instances of RAG components
    bm25_index: Optional[BM25Index] = None
    faiss_store: Optional[FAISSStore] = None
    graph_rag: Optional[GraphRAG] = None

    # Flag to track if components are initialized
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGSingleton, cls).__new__(cls)
        return cls._instance

    def initialize(self, index_dir: str = "./indexes"):
        """
        Initialize RAG components if not already initialized.
        Thread-safe initialization with double-checked locking pattern.

        Args:
            index_dir: Directory containing index files
        """
        # Quick check without acquiring lock
        if self._initialized:
            logger.debug("RAG components already initialized (quick check)")
            return

        # Use a lock to prevent multiple initializations
        if RAGSingleton._lock:
            logger.debug("Initialization in progress by another thread, waiting...")
            # Wait for initialization to complete
            while RAGSingleton._lock:
                import time

                time.sleep(0.1)
            return

        # Set lock
        RAGSingleton._lock = True

        try:
            # Double-check after acquiring lock
            if self._initialized:
                logger.debug("RAG components already initialized (after lock)")
                return

            logger.info("Initializing RAG components...")

            # Create index directory if it doesn't exist
            os.makedirs(index_dir, exist_ok=True)

            # Initialize BM25 index
            self.bm25_index = BM25Index(os.path.join(index_dir, "bm25_index.pkl"))
            bm25_loaded = self.bm25_index.load()
            if bm25_loaded:
                logger.info(
                    f"BM25 index loaded with {len(self.bm25_index.doc_ids)} documents"
                )
            else:
                logger.warning("BM25 index not loaded")

            # Initialize FAISS store
            self.faiss_store = FAISSStore(os.path.join(index_dir, "faiss_index.pkl"))
            faiss_loaded = self.faiss_store.load()
            if faiss_loaded:
                logger.info(
                    f"FAISS index loaded with {len(self.faiss_store.doc_ids)} vectors"
                )
            else:
                logger.warning("FAISS index not loaded")

            # Initialize GraphRAG
            self.graph_rag = GraphRAG(os.path.join(index_dir, "graph_rag.pkl"))
            graph_loaded = self.graph_rag.load()
            if graph_loaded:
                logger.info(
                    f"Graph loaded with {len(self.graph_rag.graph.nodes)} nodes and {len(self.graph_rag.graph.edges)} edges"
                )
            else:
                logger.warning("Graph not loaded")

            self._initialized = True
            logger.info("RAG components initialization complete")
        finally:
            # Release lock
            RAGSingleton._lock = False

    def get_bm25_index(self) -> BM25Index:
        """Get the BM25 index instance."""
        if not self._initialized:
            self.initialize()
        elif self.bm25_index is None:
            # If initialized but index is None, create a new instance without full initialization
            logger.warning(
                "BM25 index was None despite initialization, creating new instance"
            )
            self.bm25_index = BM25Index("./indexes/bm25_index.pkl")
            self.bm25_index.load()
        return self.bm25_index

    def get_faiss_store(self) -> FAISSStore:
        """Get the FAISS store instance."""
        if not self._initialized:
            self.initialize()
        elif self.faiss_store is None:
            # If initialized but store is None, create a new instance without full initialization
            logger.warning(
                "FAISS store was None despite initialization, creating new instance"
            )
            self.faiss_store = FAISSStore("./indexes/faiss_index.pkl")
            self.faiss_store.load()
        return self.faiss_store

    def get_graph_rag(self) -> GraphRAG:
        """Get the GraphRAG instance."""
        if not self._initialized:
            self.initialize()
        elif self.graph_rag is None:
            # If initialized but graph is None, create a new instance without full initialization
            logger.warning(
                "GraphRAG was None despite initialization, creating new instance"
            )
            self.graph_rag = GraphRAG("./indexes/graph_rag.pkl")
            self.graph_rag.load()
        return self.graph_rag

    def clear_all(self):
        """Clear all RAG components."""
        if self.bm25_index:
            self.bm25_index.clear()
        if self.faiss_store:
            self.faiss_store.clear()
        if self.graph_rag:
            self.graph_rag.clear()
        self._initialized = False
        logger.info("All RAG components cleared")


# Create a global instance
rag_singleton = RAGSingleton()
