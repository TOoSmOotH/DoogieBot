from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RAGComponentToggle(BaseModel):
    """
    Schema for toggling a RAG component.
    """

    component: str
    enabled: bool


class RAGBuildOptions(BaseModel):
    """
    Schema for RAG build options.
    """

    rebuild: bool = False
    use_bm25: bool = True
    use_faiss: bool = True
    use_graph: bool = True
    batch_size: int = 1000


class RAGRetrieveOptions(BaseModel):
    """
    Schema for RAG retrieve options.
    """

    query: str
    query_embedding: Optional[List[float]] = None
    top_k: int = 5
    use_bm25: Optional[bool] = None
    use_faiss: Optional[bool] = None
    use_graph: Optional[bool] = None
    rerank: bool = True
    fast_mode: bool = True
