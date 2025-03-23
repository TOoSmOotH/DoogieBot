import re
from typing import Any, Dict, List, Optional


class DocumentChunker:
    """
    Chunker for splitting documents into smaller pieces for indexing.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker with chunk size and overlap.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with specified size and overlap.

        Args:
            text: The text to chunk
            metadata: Optional metadata to include with each chunk

        Returns:
            List of dictionaries containing chunk content and metadata
        """
        if not text:
            return []

        chunks = []

        # Split text into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)

        current_chunk = []
        current_size = 0

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If adding this paragraph would exceed chunk size and we already have content,
            # create a new chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                # Join the current chunk and add it to the list
                chunk_text = "\n\n".join(current_chunk)
                chunk_meta = self._create_chunk_metadata(metadata, i, len(chunks))

                chunks.append({"content": chunk_text, "metadata": chunk_meta})

                # Start a new chunk with overlap
                overlap_size = max(0, len(current_chunk) - 1)
                current_chunk = (
                    current_chunk[-overlap_size:] if overlap_size > 0 else []
                )
                current_size = sum(len(p) for p in current_chunk)

            # Add the paragraph to the current chunk
            current_chunk.append(para)
            current_size += para_size

        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_meta = self._create_chunk_metadata(
                metadata, len(paragraphs), len(chunks)
            )

            chunks.append({"content": chunk_text, "metadata": chunk_meta})

        return chunks

    def _create_chunk_metadata(
        self, metadata: Optional[Dict[str, Any]], position: int, chunk_index: int
    ) -> Dict[str, Any]:
        """
        Create metadata for a chunk.

        Args:
            metadata: Original document metadata
            position: Position in the original document
            chunk_index: Index of the chunk

        Returns:
            Dictionary with chunk metadata
        """
        chunk_meta = {"chunk_index": chunk_index, "position": position}

        # Include original metadata if provided
        if metadata:
            chunk_meta.update(metadata)

        return chunk_meta

    def chunk_document(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document based on its content and metadata.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            List of chunks with content and metadata
        """
        return self.chunk_text(content, metadata)
