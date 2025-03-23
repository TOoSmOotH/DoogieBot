import { RAGStats, RAGBuildResult, RAGBuildOptions } from '@/types';
import { get, post } from './api';

/**
 * Get RAG system status
 */
export const getRAGStatus = async (): Promise<{ stats?: RAGStats; error?: string }> => {
  const response = await get<RAGStats>('/rag/status');

  if (response.error) {
    return { error: response.error };
  }

  return { stats: response.data };
};

/**
 * Build or rebuild RAG indexes
 */
export const buildIndexes = async (
  options: RAGBuildOptions
): Promise<{ result?: RAGBuildResult; error?: string }> => {
  const response = await post<RAGBuildResult>('/rag/build-indexes', options);

  if (response.error) {
    return { error: response.error };
  }

  return { result: response.data };
};

/**
 * Toggle RAG component status
 */
export const toggleRAGComponent = async (
  component: 'bm25' | 'faiss' | 'graph',
  enabled: boolean
): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/rag/toggle-component', {
    component,
    enabled,
  });

  if (response.error) {
    return { error: response.error };
  }
  return { success: true };
};

/**
 * Delete all document chunks from the database
 */
export const deleteAllChunks = async (): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/rag/delete-all-chunks');

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

/**
/**
 * Rebuild GraphRAG specifically
 */
export const rebuildGraphRAG = async (): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/rag/graph/rebuild');

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

/**
 * Process all documents for RAG
 */
export const processAllDocuments = async (
  config?: any
): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/rag/process-all', config);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

/**
 * Retrieve documents using the hybrid approach
 */
export const retrieveDocuments = async (
  query: string,
  queryEmbedding?: number[],
  options?: {
    topK?: number;
    useBM25?: boolean;
    useFAISS?: boolean;
    useGraph?: boolean;
    rerank?: boolean;
  }
): Promise<{ results?: any[]; error?: string }> => {
  const retrieveOptions = {
    query,
    query_embedding: queryEmbedding,
    top_k: options?.topK || 5,
    use_bm25: options?.useBM25 !== undefined ? options.useBM25 : null,
    use_faiss: options?.useFAISS !== undefined ? options.useFAISS : null,
    use_graph: options?.useGraph !== undefined ? options.useGraph : null,
    rerank: options?.rerank !== undefined ? options.rerank : true,
    fast_mode: true,
  };

  const response = await post<any[]>('/rag/retrieve', retrieveOptions);

  if (response.error) {
    return { error: response.error };
  }

  return { results: response.data || [] };
};

/**
 * Get information about a document chunk, including its document ID and title
 */
// Define the ChunkInfo type
type ChunkInfo = {
  chunk_id: string;
  chunk_index: number;
  document_id: string;
  document_title: string;
  document_type: string;
  document_filename: string;
  created_at: string;
};

export const getChunkInfo = async (
  chunkId: string
): Promise<{
  info?: ChunkInfo;
  error?: string;
}> => {
  try {
    console.log(`Fetching chunk info for ${chunkId}`);
    const response = await get<ChunkInfo>(`/rag/chunks/${chunkId}`);

    if (response.error) {
      // Handle specific error cases
      if (response.error.includes('404') || response.error.includes('not found')) {
        console.warn(`Chunk ${chunkId} not found in database`);
        return {
          error: `Chunk not found in database. This may happen if documents have been deleted or reprocessed since this chat was created.`,
        };
      }

      console.error(`Error fetching chunk info: ${response.error}`);
      return { error: response.error };
    }

    console.log(`Got chunk info:`, response.data);
    return { info: response.data };
  } catch (error) {
    console.error(`Exception fetching chunk info: ${error}`);
    return {
      error: `Failed to retrieve chunk information. This may happen if documents have been deleted or reprocessed since this chat was created.`,
    };
  }
};
