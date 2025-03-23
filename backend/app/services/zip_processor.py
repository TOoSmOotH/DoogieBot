import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.models.document import Document, DocumentType
from app.rag.document_processor import DocumentProcessor
from app.services.document import DocumentService
from fastapi import UploadFile
from sqlalchemy.orm import Session

# Set up logging
logger = logging.getLogger(__name__)


async def process_zip_file(
    db: Session,
    file_content: bytes,
    filename: str,
    user_id: str,
    process: bool = True,
    generate_embeddings: bool = True,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Tuple[List[Document], List[str], int]:
    """Process a zip file containing multiple documents."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for zip extraction: {temp_dir}")

    # Generate a unique ID for the zip file
    zip_id = str(uuid.uuid4())

    # Save the file content to disk
    zip_path = f"{temp_dir}/{zip_id}.zip"

    with open(zip_path, "wb") as f:
        f.write(file_content)

    # Lists to store results
    documents = []
    errors = []
    total_files = 0

    try:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get list of files in the zip
            file_list = [f for f in zip_ref.namelist() if not f.endswith("/")]
            total_files = len(file_list)
            logger.info(f"Found {total_files} files in zip archive")

            # Extract all files
            zip_ref.extractall(temp_dir)

            # Process each file
            for i, file_name in enumerate(file_list):
                try:
                    # Log progress
                    logger.info(f"Processing file {i+1}/{len(file_list)}: {file_name}")

                    # Skip directories and hidden files
                    if file_name.endswith("/") or os.path.basename(
                        file_name
                    ).startswith("."):
                        logger.info(f"Skipping directory or hidden file: {file_name}")
                        continue

                    # Get file extension
                    file_ext = os.path.splitext(file_name)[1].lower().lstrip(".")

                    # Skip unsupported file types
                    supported_extensions = [
                        "pdf",
                        "docx",
                        "md",
                        "rst",
                        "txt",
                        "json",
                        "jsonl",
                        "yaml",
                        "yml",
                    ]
                    if file_ext not in supported_extensions:
                        logger.info(
                            f"Skipping unsupported file type: {file_name} (extension: {file_ext})"
                        )
                        errors.append(f"Skipped unsupported file type: {file_name}")
                        continue

                    # Get the extracted file path
                    extracted_path = os.path.join(temp_dir, file_name)

                    # Create a document record
                    doc_id = str(uuid.uuid4())

                    # Determine document type
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
                    elif file_ext == "yaml" or file_ext == "yml":
                        doc_type = DocumentType.YAML
                    else:
                        # Default to text
                        doc_type = DocumentType.TEXT

                    # Create uploads directory if it doesn't exist
                    upload_dir = Path(settings.UPLOAD_DIR)
                    upload_dir.mkdir(exist_ok=True)

                    # Copy file to uploads directory
                    dest_path = f"{settings.UPLOAD_DIR}/{doc_id}.{file_ext}"
                    shutil.copy2(extracted_path, dest_path)

                    # Get file size
                    file_size = os.path.getsize(dest_path)

                    # Create document record
                    document = Document(
                        id=doc_id,
                        filename=os.path.basename(file_name),
                        title=os.path.basename(file_name),
                        type=doc_type,
                        content=dest_path,  # Store the file path
                        uploaded_by=user_id,
                        meta_data={
                            "original_filename": file_name,
                            "size": file_size,
                            "source": f"zip:{filename}",
                        },
                    )

                    db.add(document)
                    db.commit()
                    db.refresh(document)

                    documents.append(document)

                    # Process document if requested
                    if process:
                        logger.info(f"Processing document content for {file_name}")
                        processor = DocumentProcessor(
                            db,
                            generate_embeddings=generate_embeddings,
                            embedding_provider=embedding_provider,
                            embedding_model=embedding_model,
                        )

                        try:
                            success, message, chunks = await processor.process_document(
                                document
                            )

                            if success:
                                logger.info(
                                    f"Successfully processed {file_name} with {chunks} chunks"
                                )
                            else:
                                logger.warning(
                                    f"Failed to process {file_name}: {message}"
                                )
                                errors.append(
                                    f"Error processing {file_name}: {message}"
                                )
                        except Exception as proc_error:
                            logger.error(
                                f"Exception during document processing for {file_name}: {str(proc_error)}"
                            )
                            errors.append(
                                f"Exception processing {file_name}: {str(proc_error)}"
                            )

                except Exception as e:
                    logger.error(
                        f"Error processing file {file_name} from zip: {str(e)}"
                    )
                    errors.append(f"Error processing {file_name}: {str(e)}")

    except zipfile.BadZipFile:
        errors.append("Invalid zip file format")
    except Exception as e:
        logger.error(f"Error processing zip file: {str(e)}")
        errors.append(f"Error processing zip file: {str(e)}")
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

    return documents, errors, total_files
