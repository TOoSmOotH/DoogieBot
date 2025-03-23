import glob
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import git
import requests
from app.core.config import settings
from app.models.document import Document, DocumentChunk, DocumentType
from fastapi import UploadFile
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

# Set up logging
logger = logging.getLogger(__name__)


class DocumentService:
    @staticmethod
    async def upload_document(
        db: Session, file: UploadFile, user_id: str, title: Optional[str] = None
    ) -> Document:
        """
        Upload a document and save it to the filesystem.
        """
        # Generate unique ID for the document
        doc_id = str(uuid.uuid4())

        # Determine document type from file extension
        file_ext = file.filename.split(".")[-1].lower()
        doc_type = None

        if file_ext == "pdf":
            doc_type = DocumentType.PDF
        elif file_ext == "docx":
            doc_type = DocumentType.DOCX
        elif file_ext == "md":
            doc_type = DocumentType.MARKDOWN
        elif file_ext == "rst":
            doc_type = DocumentType.RST
        elif file_ext == "txt":
            doc_type = DocumentType.TEXT
        elif file_ext == "json":
            doc_type = DocumentType.JSON
        elif file_ext == "jsonl":
            doc_type = DocumentType.JSONL
        elif file_ext == "yaml":
            doc_type = DocumentType.YAML
        elif file_ext == "yml":
            doc_type = DocumentType.YML
        else:
            # Default to text
            doc_type = DocumentType.TEXT

        # Create uploads directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True)

        # Save file to disk
        file_path = f"{settings.UPLOAD_DIR}/{doc_id}.{file_ext}"

        # Read file content
        contents = await file.read()

        # Write to disk
        with open(file_path, "wb") as f:
            f.write(contents)

        # Create document record
        document = Document(
            id=doc_id,
            filename=file.filename,
            title=title or file.filename,
            type=doc_type,
            content=file_path,  # Store the file path
            uploaded_by=user_id,
            meta_data={"original_filename": file.filename, "size": len(contents)},
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    @staticmethod
    def create_manual_document(
        db: Session,
        title: str,
        content: str,
        user_id: str,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Create a manual document entry.
        """
        doc_id = str(uuid.uuid4())

        document = Document(
            id=doc_id,
            filename=None,
            title=title,
            type=DocumentType.MANUAL,
            content=content,  # Store the content directly
            uploaded_by=user_id,
            meta_data=meta_data or {},
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    @staticmethod
    def get_document(db: Session, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        """
        return db.query(Document).filter(Document.id == doc_id).first()

    @staticmethod
    def count_documents(
        db: Session, doc_type: Optional[str] = None, user_id: Optional[str] = None
    ) -> int:
        """
        Count documents, optionally filtered by type and user.
        """
        query = db.query(Document)

        if doc_type:
            query = query.filter(Document.type == doc_type)

        if user_id:
            query = query.filter(Document.uploaded_by == user_id)

        return query.count()

    @staticmethod
    def get_documents(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        doc_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Get all documents, optionally filtered by type and user.
        """
        query = db.query(Document)

        if doc_type:
            query = query.filter(Document.type == doc_type)

        if user_id:
            query = query.filter(Document.uploaded_by == user_id)

        return (
            query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
        )

    @staticmethod
    def update_document(
        db: Session,
        doc_id: str,
        title: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Document]:
        """
        Update a document's metadata.
        """
        document = DocumentService.get_document(db, doc_id)
        if not document:
            return None

        if title:
            document.title = title

        if meta_data:
            # Merge with existing metadata
            current_meta = document.meta_data or {}
            current_meta.update(meta_data)
            document.meta_data = current_meta

        db.commit()
        db.refresh(document)
        return document

    @staticmethod
    def update_document_content(
        db: Session, doc_id: str, title: str, content: str
    ) -> Optional[Document]:
        """
        Update a manual document's content.
        """
        document = DocumentService.get_document(db, doc_id)
        if not document:
            return None

        # Update title and content
        document.title = title
        document.content = content

        # Update timestamp
        document.updated_at = func.now()

        # Delete existing chunks since content has changed
        DocumentService.delete_chunks(db, doc_id)

        db.commit()
        db.refresh(document)
        return document

    @staticmethod
    def delete_document(db: Session, doc_id: str) -> bool:
        """
        Delete a document and its file if it exists.
        """
        document = DocumentService.get_document(db, doc_id)
        if not document:
            return False

        try:
            # Delete associated chunks first
            DocumentService.delete_chunks(db, doc_id)

            # Delete file if it's a file-based document
            if (
                document.type != DocumentType.MANUAL
                and document.content
                and os.path.exists(document.content)
            ):
                os.remove(document.content)

            # Delete document from database
            db.delete(document)
            db.commit()
        except Exception as e:
            db.rollback()
            raise e

        return True

    @staticmethod
    def add_chunk(
        db: Session,
        doc_id: str,
        content: str,
        chunk_index: int,
        meta_data: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> DocumentChunk:
        """
        Add a chunk to a document.
        """
        chunk_id = str(uuid.uuid4())

        # Convert embedding to JSON if provided
        embedding_json = json.dumps(embedding) if embedding else None

        chunk = DocumentChunk(
            id=chunk_id,
            document_id=doc_id,
            content=content,
            chunk_index=chunk_index,
            meta_data=meta_data or {},
            embedding=embedding_json,
        )

        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        return chunk

    @staticmethod
    def get_chunks(
        db: Session, doc_id: str, skip: int = 0, limit: int = 100
    ) -> List[DocumentChunk]:
        """
        Get all chunks for a document.
        """
        return (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.chunk_index)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_chunk(db: Session, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a chunk by ID.
        """
        return db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()

    @staticmethod
    def delete_chunks(db: Session, doc_id: str) -> int:
        """
        Delete all chunks for a document. Returns the number of chunks deleted.
        """
        result = (
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
        )
        db.commit()
        return result


@staticmethod
def delete_all_chunks(db: Session) -> int:
    """
    Delete all document chunks from the database. Returns the number of chunks deleted.
    """
    try:
        result = db.query(DocumentChunk).delete()
        db.commit()
        return result
    except Exception as e:
        db.rollback()
        raise e

    @staticmethod
    async def import_github_repository(
        db: Session,
        repo_url: str,
        branch: str,
        file_extensions: List[str],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Import documents from a GitHub repository by cloning it to a temporary directory.

        Args:
            db: Database session
            repo_url: GitHub repository URL (e.g., https://github.com/username/repo)
            branch: Branch name (default: main)
            file_extensions: List of file extensions to import (e.g., ["rst", "txt"])
            user_id: ID of the user importing the documents
            max_files: Maximum number of files to import

        Returns:
            Dictionary with import results
        """
        from app.core.config import settings

        # Parse GitHub repository URL
        # Expected format: https://github.com/username/repo
        try:
            # Check if git is available
            try:
                git.Git().version()
            except git.GitCommandError:
                return {
                    "success": False,
                    "message": "Git is not available. Please make sure Git is installed and try again.",
                }
            except ImportError:
                return {
                    "success": False,
                    "message": "Git executable not found. Please make sure Git is installed and in the PATH.",
                }

            # Remove trailing slash if present
            if repo_url.endswith("/"):
                repo_url = repo_url[:-1]

            parts = repo_url.split("/")
            if "github.com" not in repo_url or len(parts) < 5:
                return {
                    "success": False,
                    "message": "Invalid GitHub repository URL. Expected format: https://github.com/username/repo",
                }

            owner = parts[-2]
            repo_name = parts[-1]

            logger.info(
                f"Attempting to import from GitHub repository: {owner}/{repo_name}, branch: {branch}"
            )

            # Create a temporary directory for the repository
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")

            try:
                # Clone the repository
                logger.info(f"Cloning repository {repo_url} to {temp_dir}")
                repo = git.Repo.clone_from(repo_url, temp_dir)

                # Checkout the specified branch if provided
                if branch:
                    logger.info(f"Checking out branch: {branch}")
                    try:
                        repo.git.checkout(branch)
                    except git.GitCommandError as e:
                        logger.error(f"Error checking out branch {branch}: {str(e)}")
                        return {
                            "success": False,
                            "message": f"Branch '{branch}' not found in repository. Please check the branch name and try again.",
                        }

                # Find all files with the specified extensions
                all_files = []
                for ext in file_extensions:
                    # Use glob to find files with the specified extension
                    pattern = os.path.join(temp_dir, f"**/*.{ext}")
                    files = glob.glob(pattern, recursive=True)
                    all_files.extend(files)

                logger.info(
                    f"Found {len(all_files)} files with extensions: {', '.join(file_extensions)}"
                )

                if not all_files:
                    return {
                        "success": False,
                        "message": f"No files with extensions {', '.join(file_extensions)} found in the repository",
                    }

                # Log the number of files found
                logger.info(f"Processing {len(all_files)} files")

                # Import each file
                imported_count = 0
                document_ids = []

                for file_path in all_files:
                    try:
                        # Get relative path for display
                        rel_path = os.path.relpath(file_path, temp_dir)
                        logger.info(f"Processing file: {rel_path}")

                        # Get file extension
                        _, file_ext = os.path.splitext(file_path)
                        file_ext = file_ext[
                            1:
                        ].lower()  # Remove the dot and convert to lowercase

                        # Determine document type
                        doc_type = None
                        if file_ext == "pdf":
                            doc_type = DocumentType.PDF
                            # For binary files, copy to uploads directory
                            upload_dir = Path(settings.UPLOAD_DIR)
                            upload_dir.mkdir(exist_ok=True)

                            dest_path = (
                                f"{settings.UPLOAD_DIR}/{uuid.uuid4()}.{file_ext}"
                            )
                            shutil.copy2(file_path, dest_path)
                            file_content = dest_path  # Store the file path
                        elif file_ext == "docx":
                            doc_type = DocumentType.DOCX
                            # For binary files, copy to uploads directory
                            upload_dir = Path(settings.UPLOAD_DIR)
                            upload_dir.mkdir(exist_ok=True)

                            dest_path = (
                                f"{settings.UPLOAD_DIR}/{uuid.uuid4()}.{file_ext}"
                            )
                            shutil.copy2(file_path, dest_path)
                            file_content = dest_path  # Store the file path
                        elif file_ext == "md":
                            doc_type = DocumentType.MARKDOWN
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        elif file_ext == "rst":
                            doc_type = DocumentType.RST
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        elif file_ext == "txt":
                            doc_type = DocumentType.TEXT
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        elif file_ext == "json":
                            doc_type = DocumentType.JSON
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        elif file_ext == "jsonl":
                            doc_type = DocumentType.JSONL
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        elif file_ext == "yaml":
                            doc_type = DocumentType.YAML
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        elif file_ext == "yml":
                            doc_type = DocumentType.YML
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                        else:
                            # Default to text
                            doc_type = DocumentType.TEXT
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()

                        # Create document record
                        doc_id = str(uuid.uuid4())
                        filename = os.path.basename(file_path)

                        document = Document(
                            id=doc_id,
                            filename=filename,
                            title=f"{filename} (from {owner}/{repo_name})",
                            type=doc_type,
                            content=file_content,
                            uploaded_by=user_id,
                            meta_data={
                                "source": "github",
                                "repository": f"{owner}/{repo_name}",
                                "branch": branch or "default",
                                "path": rel_path,
                            },
                        )

                        db.add(document)
                        db.commit()
                        db.refresh(document)

                        imported_count += 1
                        document_ids.append(doc_id)
                        logger.info(f"Imported document: {filename}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        continue

                return {
                    "success": True,
                    "message": f"Successfully imported {imported_count} documents from GitHub repository",
                    "imported_count": imported_count,
                    "document_ids": document_ids,
                }

            except git.GitCommandError as e:
                logger.error(f"Git error: {str(e)}")
                if "not found" in str(e).lower():
                    return {
                        "success": False,
                        "message": f"Repository not found: {owner}/{repo_name}. Make sure the repository exists and is public.",
                    }
                else:
                    return {"success": False, "message": f"Git error: {str(e)}"}
            finally:
                # Clean up the temporary directory
                try:
                    logger.info(f"Cleaning up temporary directory: {temp_dir}")
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary directory: {str(e)}")

        except Exception as e:
            logger.error(f"Error importing GitHub repository: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to import GitHub repository: {str(e)}",
            }
