import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 index for keyword-based retrieval.
    """

    def __init__(self, index_path: str = "bm25_index.pkl"):
        """
        Initialize the BM25 index.

        Args:
            index_path: Path to save/load the index
        """
        self.index_path = index_path
        self.index = None
        self.corpus = []
        self.doc_ids = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization by splitting on whitespace and removing punctuation
        tokens = text.lower()
        # Remove common punctuation
        for char in ",.!?;:()[]{}\"'\\/":
            tokens = tokens.replace(char, " ")
        # Split on whitespace and filter out empty tokens
        return [token for token in tokens.split() if token]

    def add_document(self, doc_id: str, text: str) -> None:
        """
        Add a document to the index.

        Args:
            doc_id: Document ID
            text: Document text
        """
        tokenized_text = self._tokenize(text)
        self.corpus.append(text)
        self.doc_ids.append(doc_id)
        self.tokenized_corpus.append(tokenized_text)

        # Rebuild index
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
        else:
            logger.warning("No documents to index. BM25 index not created.")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add multiple documents to the index.

        Args:
            documents: List of documents with 'id' and 'content' fields
        """
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("content")
            if doc_id and text:
                tokenized_text = self._tokenize(text)
                self.corpus.append(text)
                self.doc_ids.append(doc_id)
                self.tokenized_corpus.append(tokenized_text)

        # Rebuild index only if we have documents
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
        else:
            logger.warning("No documents to index. BM25 index not created.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dictionaries with document ID, score, and content
        """
        if not self.index:
            logger.warning("Index is empty. No results returned.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append(
                    {
                        "id": self.doc_ids[idx],
                        "score": float(scores[idx]),
                        "content": self.corpus[idx],
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
                        "corpus": self.corpus,
                        "doc_ids": self.doc_ids,
                        "tokenized_corpus": self.tokenized_corpus,
                    },
                    f,
                )

            logger.info(f"BM25 index saved to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving BM25 index: {str(e)}")
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
                self.corpus = data["corpus"]
                self.doc_ids = data["doc_ids"]
                self.tokenized_corpus = data["tokenized_corpus"]

            # Rebuild index
            if self.tokenized_corpus:
                self.index = BM25Okapi(self.tokenized_corpus)
                logger.info(
                    f"BM25 index loaded from {self.index_path} with {len(self.corpus)} documents"
                )
                return True
            else:
                logger.warning("Loaded index is empty")
                return False
        except Exception as e:
            logger.error(f"Error loading BM25 index: {str(e)}")
            return False

    def clear(self) -> None:
        """
        Clear the index.
        """
        self.index = None
        self.corpus = []
        self.doc_ids = []
        self.tokenized_corpus = []

        # Remove index file if it exists
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
            logger.info(f"BM25 index file {self.index_path} removed")
