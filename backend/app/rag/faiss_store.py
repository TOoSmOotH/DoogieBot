import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSStore:
    """
    FAISS vector store for similarity search.
    """

    def __init__(
        self,
        index_path: str = "faiss_index.pkl",
        dimension: int = 1536,  # Default for OpenAI embeddings
    ):
        """
        Initialize the FAISS vector store.

        Args:
            index_path: Path to save/load the index
            dimension: Dimension of the embedding vectors
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.doc_ids = []
        self.doc_contents = []
        self.doc_metadata = []

    def _create_index(self) -> None:
        """
        Create a new FAISS index using CPU.
        """
        # Force CPU usage
        faiss.get_num_gpus = lambda: 0
        self.index = faiss.IndexFlatL2(self.dimension)

    def add_embedding(
        self,
        doc_id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a document embedding to the index.

        Args:
            doc_id: Document ID
            embedding: Document embedding vector
            content: Document content
            metadata: Optional document metadata
        """
        if self.index is None:
            self._create_index()

        # Convert embedding to numpy array
        embedding_np = np.array([embedding], dtype=np.float32)

        # Add to index
        self.index.add(embedding_np)
        self.doc_ids.append(doc_id)
        self.doc_contents.append(content)
        self.doc_metadata.append(metadata or {})

    def add_embeddings(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add multiple document embeddings to the index.

        Args:
            documents: List of documents with 'id', 'embedding', 'content', and optional 'metadata' fields
        """
        if not documents:
            return

        if self.index is None:
            self._create_index()

        # Extract embeddings
        embeddings = []
        for doc in documents:
            doc_id = doc.get("id")
            embedding = doc.get("embedding")
            content = doc.get("content")
            metadata = doc.get("metadata", {})

            if doc_id and embedding and content:
                embeddings.append(embedding)
                self.doc_ids.append(doc_id)
                self.doc_contents.append(content)
                self.doc_metadata.append(metadata)

        if embeddings:
            # Convert embeddings to numpy array
            embeddings_np = np.array(embeddings, dtype=np.float32)

            # Add to index
            self.index.add(embeddings_np)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the index for documents similar to the query embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of dictionaries with document ID, score, content, and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty. No results returned.")
            return []

        # Convert query embedding to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)

        # Search index
        distances, indices = self.index.search(query_np, min(top_k, self.index.ntotal))

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no match
                results.append(
                    {
                        "id": self.doc_ids[idx],
                        "score": float(
                            1.0 / (1.0 + distances[0][i])
                        ),  # Convert distance to similarity score
                        "content": self.doc_contents[idx],
                        "metadata": self.doc_metadata[idx],
                    }
                )

        return results

    def save(self) -> bool:
        """
        Save the index to disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True
            )

            # Save index data
            with open(self.index_path, "wb") as f:
                pickle.dump(
                    {
                        "doc_ids": self.doc_ids,
                        "doc_contents": self.doc_contents,
                        "doc_metadata": self.doc_metadata,
                        "dimension": self.dimension,
                    },
                    f,
                )

            # Save FAISS index
            index_bin_path = self.index_path + ".bin"
            if self.index is not None:
                faiss.write_index(self.index, index_bin_path)

            logger.info(f"FAISS index saved to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            return False

    def load(self) -> bool:
        """
        Load the index from disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.index_path):
                logger.warning(f"Index file {self.index_path} does not exist")
                return False

            # Load index data
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.doc_ids = data["doc_ids"]
                self.doc_contents = data["doc_contents"]
                self.doc_metadata = data["doc_metadata"]
                self.dimension = data["dimension"]

            # Load FAISS index
            index_bin_path = self.index_path + ".bin"
            if os.path.exists(index_bin_path):
                self.index = faiss.read_index(index_bin_path)
                logger.info(
                    f"FAISS index loaded from {index_bin_path} with {self.index.ntotal} vectors"
                )
                return True
            else:
                logger.warning(
                    f"FAISS binary index file {index_bin_path} does not exist"
                )
                return False
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False

    def clear(self) -> None:
        """
        Clear the index.
        """
        self.index = None
        self.doc_ids = []
        self.doc_contents = []
        self.doc_metadata = []

        # Remove index files if they exist
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
            logger.info(f"FAISS index file {self.index_path} removed")

        index_bin_path = self.index_path + ".bin"
        if os.path.exists(index_bin_path):
            os.remove(index_bin_path)
            logger.info(f"FAISS binary index file {index_bin_path} removed")
