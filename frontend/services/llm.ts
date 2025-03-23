import { LLMConfig, LLMModel, LLMProvider } from '@/types';
import { get, post, put, del } from './api';

// Get all available LLM providers
export const getLLMProviders = async (): Promise<{ providers?: any; error?: string }> => {
  const response = await get<any>('/llm/providers');

  if (response.error) {
    return { error: response.error };
  }

  return { providers: response.data };
};

// Get available models for a provider
export const getProviderModels = async (
  providerId: string,
  apiKey?: string,
  baseUrl?: string
): Promise<{ chatModels?: string[]; embeddingModels?: string[]; error?: string }> => {
  // Build query parameters
  const params = new URLSearchParams();
  if (apiKey) params.append('api_key', apiKey);
  if (baseUrl) params.append('base_url', baseUrl);

  const queryString = params.toString() ? `?${params.toString()}` : '';
  const response = await get<{ chat_models: string[]; embedding_models: string[] }>(
    `/llm/providers/${providerId}/models${queryString}`
  );

  if (response.error) {
    return { error: response.error };
  }

  return {
    chatModels: response.data?.chat_models || [],
    embeddingModels: response.data?.embedding_models || [],
  };
};

// Get all LLM configurations
export const getAllLLMConfigs = async (): Promise<{ configs?: LLMConfig[]; error?: string }> => {
  const response = await get<LLMConfig[]>('/llm/admin/config');

  if (response.error) {
    return { error: response.error };
  }

  return { configs: response.data };
};

// Get active LLM configuration
export const getActiveLLMConfig = async (): Promise<{ config?: LLMConfig; error?: string }> => {
  const response = await get<LLMConfig>('/llm/admin/config/active');

  if (response.error) {
    return { error: response.error };
  }

  return { config: response.data };
};

// Create new LLM configuration
export const createLLMConfig = async (config: {
  provider: string;
  model: string;
  embedding_model: string;
  system_prompt: string;
  api_key?: string;
  base_url?: string;
}): Promise<{ config?: LLMConfig; error?: string }> => {
  const response = await post<LLMConfig>('/llm/admin/config', config);

  if (response.error) {
    return { error: response.error };
  }

  return { config: response.data };
};

// Update LLM configuration
export const updateLLMConfig = async (
  configId: string,
  config: {
    provider?: string;
    model?: string;
    embedding_model?: string;
    system_prompt?: string;
    api_key?: string;
    base_url?: string;
    is_active?: boolean;
  }
): Promise<{ config?: LLMConfig; error?: string }> => {
  const response = await put<LLMConfig>(`/llm/admin/config/${configId}`, config);

  if (response.error) {
    return { error: response.error };
  }

  return { config: response.data };
};

// Activate LLM configuration
export const activateLLMConfig = async (
  configId: string
): Promise<{ config?: LLMConfig; error?: string }> => {
  const response = await post<LLMConfig>(`/llm/admin/config/${configId}/activate`, {});

  if (response.error) {
    return { error: response.error };
  }

  return { config: response.data };
};

// Delete LLM configuration
export const deleteLLMConfig = async (
  configId: string
): Promise<{ success?: boolean; error?: string }> => {
  const response = await del(`/llm/admin/config/${configId}`);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};
