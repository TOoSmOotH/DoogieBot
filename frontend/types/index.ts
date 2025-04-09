// User Types
export interface User {
  id: string;
  email: string;
  role: 'admin' | 'user';
  status: 'active' | 'pending' | 'inactive';
  theme_preference: string;
  created_at: string;
  updated_at: string;
  last_login?: string;
}

export interface UserCreate {
  email: string;
  password: string;
}

export interface UserUpdate {
  email?: string;
  password?: string;
  status?: string;
  role?: string;
}

// Authentication Types
export interface LoginOptions {
  rememberMe: boolean;
}

export interface Token {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface TokenPayload {
  sub: string;
  exp: number;
  refresh?: boolean;
}

// Chat Types
export interface Chat {
  id: string;
  title: string;
  user_id: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
  tags?: string[];
}

export interface Tag {
  id: string;
  name: string;
  color: string;
}

export interface Message {
  id: number;
  chat_id: number;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  created_at: string;
  tokens?: number;
  tokens_per_second?: number;
  model?: string;
  provider?: string;
  feedback?: string;
  feedback_text?: string;
  reviewed?: boolean;
  context_documents?: string[];
  document_ids?: string[]; // Add this field for document references
  related_question_content?: string; // Add field for related question content
  tool_calls?: any[]; // Tool calls for assistant messages
  tool_call_id?: string; // ID linking a tool result back to a tool call
  name?: string; // Name of the function called (for tool role messages)
  parentMessages?: Message[]; // Messages in the same chat for reference (used for tool call display)
}

export interface Feedback {
  feedback: string;
  feedback_text?: string;
}

// Document Types
export interface Document {
  id: string;
  title: string;
  filename: string;
  type: string;
  uploaded_by: string;
  created_at: string;
  updated_at: string;
  meta_data?: any;
  content?: string; // Content might still be present for manual docs or detail view
  chunk_count?: number; // Added chunk count
  // chunks?: DocumentChunk[]; // Removed, fetched separately
}

export interface DocumentChunk { // Keep this for potential future use or other contexts
  id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  meta_data?: any;
  created_at: string;
  embedding?: number[];
}

// New type for listing chunk IDs
export interface DocumentChunkId {
  id: string;
  chunk_index: number;
}

// New type for fetching chunk details
export interface DocumentChunkDetail {
  id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  meta_data?: any;
  created_at: string;
}


export interface ProcessingStatus {
  status: string;
  message: string;
  document_id?: string;
  total_chunks?: number;
}

export interface ChunkingConfig {
  chunk_size: number;
  chunk_overlap: number;
}

export interface EmbeddingConfig {
  id: string;
  provider: string;
  model: string;
  api_key?: string;
  base_url?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  config?: any;
}

export interface RerankingConfig {
  id: string;
  provider: string;
  model: string;
  api_key?: string;
  base_url?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  config?: any;
}

export interface ProcessingConfig {
  chunking: ChunkingConfig;
  embedding?: EmbeddingConfig;
  rebuild_index: boolean;
}

// LLM Types
export interface LLMProvider {
  id: string;
  name: string;
  description: string;
  requires_api_key: boolean;
  requires_base_url: boolean;
}

export interface LLMModel {
  id: string;
  name: string;
  provider_id: string;
  description: string;
  max_tokens: number;
  is_chat_model: boolean;
  is_embedding_model: boolean;
}

export interface LLMConfig {
  id: string;
  chat_provider: string;
  embedding_provider: string;
  reranking_provider?: string;
  model: string;
  embedding_model: string;
  reranking_model?: string;
  system_prompt: string;
  api_key?: string;
  base_url?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  reranked_top_n?: number | null; // Add the new field here
  config?: any;
}

// RAG Types
export interface RAGStats {
  document_count: number;
  chunk_count: number;
  last_updated: string;
  bm25_status: RAGComponentStatus;
  faiss_status: RAGComponentStatus;
  graph_status: RAGComponentStatus;
}

export interface RAGComponentStatus {
  enabled: boolean;
  document_count: number;
  last_indexed: string | null;
  status: 'idle' | 'indexing' | 'error';
  error_message?: string;
  node_count?: number;
  edge_count?: number;
  implementation?: string;
}

export interface RAGBuildResult {
  status: string;
  message: string;
  bm25: RAGBuildComponentResult;
  faiss: RAGBuildComponentResult;
  graph: RAGBuildComponentResult;
}

export interface RAGBuildComponentResult {
  status: string;
  documents?: number;
  nodes?: number;
  edges?: number;
  message?: string;
}

export interface RAGBuildOptions {
  rebuild: boolean;
  use_bm25: boolean;
  use_faiss: boolean;
  use_graph: boolean;
}

// Theme Types
export type Theme = 'dark' | 'light';

// API Response Types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  errorObject?: any;
}

// Pagination Types
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export interface PaginationParams {
  page?: number;
  size?: number;
  search?: string;
  type?: string;
  doc_type?: string;
}
