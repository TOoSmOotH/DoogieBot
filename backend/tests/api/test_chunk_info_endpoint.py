import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from main import app
from app.models.document import DocumentChunk, Document
from app.utils.deps import get_db, get_current_user # Import get_current_user
from app.models.user import User, UserRole # Import User

# Create test client
client = TestClient(app)

# Define mock user
mock_test_user = User(
    id="test_user_id",
    email="test@example.com",
    role=UserRole.USER, # Use UserRole enum
    status="active"
)

# Mock database dependency fixture
@pytest.fixture
def mock_db():
    """Create a mock database session"""
    mock_session = MagicMock(spec=Session)
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value.first.return_value = None
    mock_query.filter.return_value.all.return_value = []
    return mock_session

# Override get_db and get_current_user for each function using fixtures
@pytest.fixture(autouse=True)
def apply_overrides(mock_db):
     """Applies dependency overrides for db and user for test functions."""
     app.dependency_overrides[get_db] = lambda: mock_db
     app.dependency_overrides[get_current_user] = lambda: mock_test_user # Use defined mock user
     yield
     app.dependency_overrides = {} # Clear overrides after test


@patch("app.api.routes.rag.retrieval.DocumentService.get_document")
@patch("app.api.routes.rag.retrieval.DocumentService.get_chunk")
def test_get_chunk_info_success(mock_get_chunk, mock_get_document, mock_db): # mock_db is injected by fixture
    """Test successful retrieval of chunk information"""
    # Mock get_chunk
    test_chunk = DocumentChunk(
        id="test_chunk_id",
        document_id="test_doc_id",
        content="Test chunk content",
        chunk_index=1,
        created_at=datetime.now(timezone.utc)
    )
    mock_get_chunk.return_value = test_chunk

    # Mock get_document
    test_document = Document(
        id="test_doc_id",
        title="Test Document",
        type="text",
        filename="test_file.txt",
        uploaded_by="test_user_id",
        created_at=datetime.now(timezone.utc)
    )
    mock_get_document.return_value = test_document

    response = client.get("/api/v1/rag/chunks/test_chunk_id")

    assert response.status_code == 200
    data = response.json()
    assert data["chunk_id"] == "test_chunk_id"
    assert data["document_title"] == "Test Document"
    mock_get_chunk.assert_called_once_with(mock_db, "test_chunk_id")
    mock_get_document.assert_called_once_with(mock_db, "test_doc_id")


@patch("app.api.routes.rag.retrieval.DocumentService.get_chunk")
def test_get_chunk_info_not_found(mock_get_chunk, mock_db): # mock_db is injected by fixture
    """Test retrieval of a non-existent chunk"""
    mock_get_chunk.return_value = None

    response = client.get("/api/v1/rag/chunks/nonexistent_chunk_id")

    assert response.status_code == 404
    assert "Chunk not found" in response.json()["detail"]
    mock_get_chunk.assert_called_once_with(mock_db, "nonexistent_chunk_id")


@patch("app.api.routes.rag.retrieval.DocumentService.get_document")
@patch("app.api.routes.rag.retrieval.DocumentService.get_chunk")
def test_get_chunk_info_document_not_found(mock_get_chunk, mock_get_document, mock_db): # mock_db is injected by fixture
    """Test when chunk exists but document doesn't"""
    # Mock get_chunk to return a valid chunk
    mock_chunk = DocumentChunk(
        id="test_chunk_id",
        document_id="missing_doc_id",
        content="Test chunk content",
        chunk_index=1,
        created_at=datetime.now(timezone.utc)
    )
    mock_get_chunk.return_value = mock_chunk

    # Mock get_document to return None
    mock_get_document.return_value = None

    response = client.get("/api/v1/rag/chunks/test_chunk_id")

    # Endpoint should return 200 OK with placeholders
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_id"] == "test_chunk_id"
    assert data["document_id"] == "missing_doc_id"
    assert data["document_title"] == "Unknown document"
    assert data["document_type"] is None
    assert data["document_filename"] is None

    mock_get_chunk.assert_called_once_with(mock_db, "test_chunk_id")
    mock_get_document.assert_called_once_with(mock_db, "missing_doc_id")
