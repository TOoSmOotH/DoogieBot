import { get, post, put, del } from './api';
import { ApiResponse } from '@/types'; // Corrected: Import ApiResponse from its source
import { AppError } from '@/utils/errorHandling';

// --- TypeScript Interfaces (mirroring Pydantic schemas) ---

export interface MCPServerConfigBase {
  name: string;
  command: 'docker'; // Enforce 'docker' as per backend validation
  args: string[];
  env?: Record<string, string>;
  enabled: boolean;
}

export interface MCPServerConfigCreate extends MCPServerConfigBase {}

export interface MCPServerConfigUpdate {
  name?: string;
  args?: string[];
  env?: Record<string, string>;
  enabled?: boolean;
}

export interface MCPServerConfigResponse extends MCPServerConfigBase {
  id: string;
  user_id: string;
  created_at: string; // Representing datetime as string
  updated_at: string; // Representing datetime as string
}

export interface MCPServerStatus {
  id: string;
  name: string;
  enabled: boolean;
  status: 'running' | 'stopped' | 'error'; // Literal types for status
  container_id?: string;
  error_message?: string;
}

export interface MCPConfigJSON {
  mcpServers: Record<string, Record<string, any>>; // Using 'any' for flexibility as per backend schema
}

// --- Helper function to handle API response ---
const handleResponse = <T>(response: ApiResponse<T>, errorMessage: string): T => {
  if (response.error) {
    // Re-throw the specific error object if available, otherwise a generic one
    throw response.errorObject || new Error(response.error || errorMessage);
  }
  if (response.data === undefined) {
    // Handle cases where data might be unexpectedly undefined even without an error
    // For DELETE (204 No Content), undefined data is expected and okay.
    // For other methods, this might indicate an issue.
    // We return undefined here, callers might need to check.
    // Or throw new Error('API returned undefined data unexpectedly.');
  }
  // Type assertion needed because T could be void for DELETE
  return response.data as T;
};


// --- API Service Functions ---

const MCP_BASE_URL = '/mcp/configs'; // Base path for MCP config endpoints

/**
 * Fetches all MCP server configurations for the current user.
 */
export const getMcpConfigs = async (): Promise<MCPServerConfigResponse[]> => {
  const response = await get<MCPServerConfigResponse[]>(MCP_BASE_URL);
  return handleResponse(response, 'Failed to fetch MCP configurations');
};

/**
 * Fetches a specific MCP server configuration by its ID.
 * @param configId - The ID of the configuration to fetch.
 */
export const getMcpConfig = async (configId: string): Promise<MCPServerConfigResponse> => {
  const response = await get<MCPServerConfigResponse>(`${MCP_BASE_URL}/${configId}`);
  return handleResponse(response, `Failed to fetch MCP configuration ${configId}`);
};

/**
 * Creates a new MCP server configuration. (Admin only)
 * @param configData - The data for the new configuration.
 */
export const createMcpConfig = async (configData: MCPServerConfigCreate): Promise<MCPServerConfigResponse> => {
  const response = await post<MCPServerConfigResponse>(MCP_BASE_URL, configData);
  return handleResponse(response, 'Failed to create MCP configuration');
};

/**
 * Updates an existing MCP server configuration. (Admin only)
 * @param configId - The ID of the configuration to update.
 * @param configData - The update data.
 */
export const updateMcpConfig = async (configId: string, configData: MCPServerConfigUpdate): Promise<MCPServerConfigResponse> => {
  const response = await put<MCPServerConfigResponse>(`${MCP_BASE_URL}/${configId}`, configData);
  return handleResponse(response, `Failed to update MCP configuration ${configId}`);
};

/**
 * Deletes an MCP server configuration. (Admin only)
 * @param configId - The ID of the configuration to delete.
 */
export const deleteMcpConfig = async (configId: string): Promise<void> => {
  // DELETE requests often return 204 No Content, so data might be undefined
  const response = await del<void>(`${MCP_BASE_URL}/${configId}`);
  // Use handleResponse, but expect undefined data for success
  handleResponse(response, `Failed to delete MCP configuration ${configId}`);
};

/**
 * Fetches the status of a specific MCP server.
 * @param configId - The ID of the configuration whose status to fetch.
 */
export const getMcpConfigStatus = async (configId: string): Promise<MCPServerStatus> => {
  const response = await get<MCPServerStatus>(`${MCP_BASE_URL}/${configId}/status`);
  return handleResponse(response, `Failed to fetch status for MCP configuration ${configId}`);
};

/**
 * Starts a specific MCP server. (Admin only)
 * @param configId - The ID of the configuration to start.
 */
export const startMcpServer = async (configId: string): Promise<MCPServerStatus> => {
  const response = await post<MCPServerStatus>(`${MCP_BASE_URL}/${configId}/start`);
  return handleResponse(response, `Failed to start MCP server ${configId}`);
};

/**
 * Stops a specific MCP server. (Admin only)
 * @param configId - The ID of the configuration to stop.
 */
export const stopMcpServer = async (configId: string): Promise<MCPServerStatus> => {
  const response = await post<MCPServerStatus>(`${MCP_BASE_URL}/${configId}/stop`);
  return handleResponse(response, `Failed to stop MCP server ${configId}`);
};

/**
 * Restarts a specific MCP server. (Admin only)
 * @param configId - The ID of the configuration to restart.
 */
export const restartMcpServer = async (configId: string): Promise<MCPServerStatus> => {
  const response = await post<MCPServerStatus>(`${MCP_BASE_URL}/${configId}/restart`);
  return handleResponse(response, `Failed to restart MCP server ${configId}`);
};

/**
 * Fetches the MCP configuration in JSON format (e.g., for Claude Desktop).
 */
export const getMcpConfigJson = async (): Promise<MCPConfigJSON> => {
  const response = await get<MCPConfigJSON>(`${MCP_BASE_URL}/json`);
  return handleResponse(response, 'Failed to fetch MCP configuration JSON');
};