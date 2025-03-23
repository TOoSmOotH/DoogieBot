import { Document, PaginatedResponse, PaginationParams, ProcessingStatus } from '@/types';
import { del, get, getPaginated, post, put } from './api';

// Get all documents with pagination
export const getDocuments = async (
  params?: PaginationParams
): Promise<{
  documents?: PaginatedResponse<Document>;
  error?: string;
}> => {
  console.log('Fetching documents with params:', params);
  const response = await getPaginated<Document>('/documents', params);

  console.log('Documents response:', response);
  if (response.error) {
    console.error('Error fetching documents:', response.error);
    return { error: response.error };
  }

  return { documents: response.data };
};

// Get a single document by ID
export const getDocument = async (
  documentId: string
): Promise<{ document?: Document; error?: string }> => {
  const response = await get<Document>(`/documents/${documentId}`);

  if (response.error) {
    return { error: response.error };
  }

  return { document: response.data };
};

// Upload a document
export const uploadDocument = async (
  file: File,
  title?: string,
  process: boolean = false
): Promise<{ document?: Document; error?: string }> => {
  const formData = new FormData();
  formData.append('file', file);

  if (title) {
    formData.append('title', title);
  } else {
    formData.append('title', file.name);
  }

  // Add process flag if needed
  if (process) {
    formData.append('process', 'true');
  }

  console.log('Uploading document:', { title: title || file.name, process });
  const response = await post<Document>('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  console.log('Upload response:', response);
  if (response.error) {
    console.error('Error uploading document:', response.error);
    return { error: response.error };
  }

  return { document: response.data };
};

// Create a manual document
export const createManualDocument = async (
  title: string,
  content: string,
  metaData?: any
): Promise<{ document?: Document; error?: string }> => {
  console.log('Creating manual document:', { title, contentLength: content.length });
  const response = await post<Document>('/documents/manual', {
    title,
    content,
    meta_data: metaData,
  });

  console.log('Manual document response:', response);
  if (response.error) {
    console.error('Error creating manual document:', response.error);
    return { error: response.error };
  }

  return { document: response.data };
};

// Update a document
export const updateDocument = async (
  documentId: string,
  title?: string,
  metaData?: any
): Promise<{ document?: Document; error?: string }> => {
  const response = await put<Document>(`/documents/${documentId}`, {
    title,
    meta_data: metaData,
  });

  if (response.error) {
    return { error: response.error };
  }

  return { document: response.data };
};

// Update a manual document's content
export const updateDocumentContent = async (
  documentId: string,
  title: string,
  content: string
): Promise<{ document?: Document; error?: string }> => {
  console.log('Updating manual document content:', {
    documentId,
    title,
    contentLength: content.length,
  });
  const response = await put<Document>(`/documents/${documentId}/content`, {
    title,
    content,
  });

  console.log('Update document content response:', response);
  if (response.error) {
    console.error('Error updating document content:', response.error);
    return { error: response.error };
  }

  return { document: response.data };
};

// Delete a document
export const deleteDocument = async (
  documentId: string
): Promise<{ success?: boolean; error?: string }> => {
  console.log('Deleting document:', documentId);
  const response = await del<boolean>(`/documents/${documentId}`);

  console.log('Delete response:', response);
  if (response.error) {
    console.error('Error deleting document:', response.error);
    return { error: response.error };
  }

  return { success: response.data };
};

// Process a document
export const processDocument = async (
  documentId: string,
  config?: any
): Promise<{ status?: ProcessingStatus; error?: string }> => {
  console.log('Processing document:', documentId, config);
  const response = await post<ProcessingStatus>(`/documents/${documentId}/process`, config);

  console.log('Process response:', response);
  if (response.error) {
    console.error('Error processing document:', response.error);
    return { error: response.error };
  }

  return { status: response.data };
};

// Batch process documents
export const batchProcessDocuments = async (
  documentIds: string[],
  config?: any
): Promise<{ result?: any; error?: string }> => {
  const response = await post('/documents/batch-process', {
    document_ids: documentIds,
    ...config,
  });

  if (response.error) {
    return { error: response.error };
  }

  return { result: response.data };
};

// Get RAG statistics
export const getRAGStats = async (): Promise<{ stats?: any; error?: string }> => {
  const response = await get('/rag/stats');

  if (response.error) {
    return { error: response.error };
  }

  return { stats: response.data };
};

// Rebuild RAG indexes
export const rebuildRAG = async (options?: any): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/rag/rebuild', options);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Rebuild GraphRAG
export const rebuildGraphRAG = async (): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/rag/rebuild-graph');

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Delete all documents and reset RAG
export const deleteAllDocumentsAndResetRAG = async (): Promise<{
  success?: boolean;
  error?: string;
}> => {
  console.log('Deleting all documents and resetting RAG...');

  // First delete all documents
  const deleteResponse = await del('/documents/all');

  if (deleteResponse.error) {
    console.error('Error deleting all documents:', deleteResponse.error);
    return { error: deleteResponse.error };
  }

  // Then delete all chunks
  const chunksResponse = await post('/rag/delete-all-chunks');

  if (chunksResponse.error) {
    console.error('Error deleting all chunks:', chunksResponse.error);
    return { error: chunksResponse.error };
  }

  // Reset the RAG system by deleting index files
  const resetResponse = await post('/rag/reset');

  if (resetResponse.error) {
    console.error('Error resetting RAG system:', resetResponse.error);
    return { error: resetResponse.error };
  }

  return { success: true };
};

// Add manual information to RAG
export const addManualInfo = async (
  title: string,
  content: string
): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/documents/manual', { title, content });

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Upload a zip file containing multiple documents
export const uploadZipFile = async (
  file: File,
  generateEmbeddings: boolean = true
): Promise<{ success?: boolean; error?: string; message?: string }> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('process', 'true'); // Always process documents from zip
  formData.append('generate_embeddings', generateEmbeddings ? 'true' : 'false');

  console.log('Uploading zip file:', { filename: file.name, size: file.size, generateEmbeddings });
  const response = await post<any>('/documents/upload-zip', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  console.log('Zip upload response:', response);
  if (response.error) {
    console.error('Error uploading zip file:', response.error);
    return { error: response.error };
  }

  return {
    success: true,
    message: response.data?.message || 'Zip file uploaded successfully',
  };
};

// Import documents from GitHub repository
export const importGitHubRepository = async (
  repoUrl: string,
  branch: string = 'main',
  fileTypes: string = 'rst,txt',
  backgroundProcessing: boolean = false,
  refresh: boolean = false
): Promise<{ success?: boolean; error?: string; message?: string; imported_count?: number }> => {
  console.log('Importing GitHub repository:', {
    repoUrl,
    branch,
    fileTypes,
    backgroundProcessing,
    refresh,
  });

  // If refresh is true, first delete any existing documents from this repository
  if (refresh) {
    try {
      // Get all documents
      const { documents } = await getDocuments();

      if (documents && documents.items) {
        // Find documents from this repository
        const repoDocuments = documents.items.filter(
          (doc) =>
            doc.meta_data &&
            doc.meta_data.source === 'github' &&
            doc.meta_data.repository &&
            doc.meta_data.repository.includes(repoUrl.split('/').slice(-2).join('/'))
        );

        // Delete each document
        for (const doc of repoDocuments) {
          await deleteDocument(doc.id);
        }

        console.log(`Deleted ${repoDocuments.length} existing documents from repository`);
      }
    } catch (err) {
      console.error('Error deleting existing repository documents:', err);
      // Continue with import even if deletion fails
    }
  }

  // Import the repository
  const response = await post<any>('/documents/github', {
    repo_url: repoUrl,
    branch,
    file_types: fileTypes,
    background_processing: backgroundProcessing,
  });

  console.log('GitHub import response:', response);
  if (response.error) {
    console.error('Error importing GitHub repository:', response.error);
    return { error: response.error };
  }

  if (response.data?.status === 'success') {
    return {
      success: true,
      message: response.data.message,
      imported_count: response.data.imported_count,
    };
  }

  return { success: true };
};
