import { Chat, Feedback, Message, PaginatedResponse, PaginationParams, Tag } from '@/types';
import { del, get, getPaginated, post, put, getApiUrl, getToken } from './api';

// Use getUserTags() instead to get tags from the backend
// Empty array to avoid errors in case API isn't yet available
export const PREDEFINED_TAGS: Tag[] = [];

// Tag Management API Functions

/**
 * Get all tags for the current user (simple list, no pagination)
 */
export const getUserTags = async (): Promise<{
  tags?: Tag[];
  error?: string;
}> => {
  console.log('Fetching user tags');

  // Check authentication
  const token = getToken();
  if (!token) {
    console.error('No authentication token found');
    return { error: 'Authentication failed. Please log in again.', tags: [] };
  }

  try {
    // Use the standard API function to ensure proper URL handling
    const response = await get<Tag[]>('/tags');
    
    console.log('getUserTags response:', response);
    
    if (response.error) {
      console.error('Error from API:', response.error);
      return { error: response.error, tags: [] };
    }
    
    return { tags: response.data };
  } catch (error) {
    console.error('Error in getUserTags:', error);
    // Return empty array with error for graceful degradation
    return { tags: [], error: 'Failed to fetch tags. Please try again.' };
  }
};

/**
 * Search and filter tags with pagination
 */
export interface TagSearchParams {
  search?: string;
  color?: string;
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedTags {
  items: Tag[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export const searchTags = async (params: TagSearchParams = {}): Promise<{
  data?: PaginatedTags;
  error?: string;
}> => {
  try {
    console.log('Searching tags with params:', params);
    
    // Build the query parameters
    const queryParams: Record<string, string> = {};
    
    if (params.search) queryParams.search = params.search;
    if (params.color) queryParams.color = params.color;
    if (params.page) queryParams.page = params.page.toString();
    if (params.pageSize) queryParams.page_size = params.pageSize.toString();
    if (params.sortBy) queryParams.sort_by = params.sortBy;
    if (params.sortOrder) queryParams.sort_order = params.sortOrder;
    
    // Use the standard API function for proper URL handling
    const response = await get<PaginatedTags>('/tags/search', queryParams);
    
    console.log('searchTags response:', response);
    
    if (response.error) {
      console.error('Error from API:', response.error);
      return { error: response.error };
    }
    
    return { data: response.data };
  } catch (error) {
    console.error('Error in searchTags:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return { error: `Failed to search tags: ${errorMessage}` };
  }
};

/**
 * Create a new tag
 */
export const createTag = async (name: string, color: string): Promise<{
  tag?: Tag;
  error?: string;
}> => {
  try {
    console.log('Creating tag with name:', name, 'and color:', color);

    // Use the standard API function for proper URL handling
    const response = await post<Tag>('/tags', { name, color });
    
    console.log('createTag response:', response);
    
    if (response.error) {
      console.error('Error from API:', response.error);
      return { error: response.error };
    }
    
    return { tag: response.data };
  } catch (error) {
    console.error('Error in createTag:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return { error: `Failed to create tag: ${errorMessage}` };
  }
};

/**
 * Update an existing tag
 */
export const updateTag = async (
  tagId: string,
  data: { name?: string; color?: string }
): Promise<{
  tag?: Tag;
  error?: string;
}> => {
  try {
    console.log('Updating tag with id:', tagId, 'and data:', data);

    // Use the standard API function for proper URL handling
    const response = await put<Tag>(`/tags/${tagId}`, data);
    
    console.log('updateTag response:', response);
    
    if (response.error) {
      console.error('Error from API:', response.error);
      return { error: response.error };
    }
    
    return { tag: response.data };
  } catch (error) {
    console.error('Error in updateTag:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return { error: `Failed to update tag: ${errorMessage}` };
  }
};

/**
 * Delete a tag
 */
export const deleteTag = async (tagId: string): Promise<{
  success?: boolean;
  error?: string;
}> => {
  try {
    console.log('Deleting tag with id:', tagId);

    // Use the standard API function for proper URL handling
    const response = await del(`/tags/${tagId}`);
    
    console.log('deleteTag response:', response);
    
    if (response.error) {
      console.error('Error from API:', response.error);
      return { error: response.error };
    }
    
    return { success: true };
  } catch (error) {
    console.error('Error in deleteTag:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return { error: `Failed to delete tag: ${errorMessage}` };
  }
};

// Get all chats
export const getChats = async (): Promise<{
  chats?: Chat[];
  error?: string;
}> => {
  try {
    const response = await get<Chat[]>('/chats');
    console.log('getChats response:', response);

    if (response.error) {
      return { error: response.error };
    }

    return { chats: response.data };
  } catch (error) {
    console.error('Error in getChats:', error);
    return { error: 'Failed to fetch chats. Please try again.' };
  }
};

// Get a single chat by ID
export const getChat = async (chatId: string): Promise<{ chat?: Chat; error?: string }> => {
  try {
    console.log('Getting chat with ID:', chatId);
    const response = await get<Chat>(`/chats/${chatId}`);
    console.log('getChat response:', response);

    if (response.error) {
      return { error: response.error };
    }

    // Ensure chat has a tags property
    if (response.data && !response.data.tags) {
      response.data.tags = [];
    }

    return { chat: response.data };
  } catch (error) {
    console.error('Error in getChat:', error);
    return { error: 'Failed to fetch chat. Please try again.' };
  }
};

// Create a new chat
export const createChat = async (title: string): Promise<{ chat?: Chat; error?: string }> => {
  try {
    console.log('Creating chat with title:', title);
    
    // Use the standard API function for proper URL handling
    const response = await post<Chat>('/chats', { title });
    
    console.log('createChat response:', response);
    
    if (response.error) {
      console.error('Error from API:', response.error);
      return { error: response.error };
    }
    
    return { chat: response.data };
  } catch (error) {
    console.error('Error in createChat:', error);
    // More detailed error message for debugging
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return { error: `Failed to create chat: ${errorMessage}` };
  }
};

// Update a chat
export const updateChat = async (
  chatId: string, 
  data: { title?: string; tags?: string[] }
): Promise<{ success?: boolean; error?: string }> => {
  const response = await put(`/chats/${chatId}`, data);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Update chat tags
export const updateChatTags = async (
  chatId: string,
  tags: string[]
): Promise<{ success?: boolean; error?: string }> => {
  console.log('Using service updateChatTags to update tags for chat:', chatId, 'tags:', tags);
  try {
    // Validation
    if (!chatId) return { error: 'Missing chat ID' };
    if (!Array.isArray(tags)) {
      console.error('Tags is not an array:', tags);
      return { error: 'Invalid tags format' };
    }
    
    // Makes sure we have a clean array of strings
    const validTags = tags.filter(tag => typeof tag === 'string' && tag.trim().length > 0);
    
    // First try to use the dedicated tags endpoint (preferred method)
    const response = await put(`/tags/chats/${chatId}/tags`, { tags: validTags });
    console.log('updateChatTags response from tags endpoint:', response);
    
    if (!response.error) {
      console.log('Tags updated successfully using dedicated endpoint');
      return { success: true };
    }
    
    // If the dedicated endpoint fails, fall back to the chat update endpoint
    console.log('Dedicated endpoint failed. Trying fallback to /chats endpoint');
    const fallbackResponse = await put(`/chats/${chatId}`, { tags: validTags });
    console.log('updateChatTags fallback response:', fallbackResponse);
    
    if (fallbackResponse.error) {
      console.error('Error updating chat tags:', fallbackResponse.error);
      return { error: fallbackResponse.error };
    }
    
    console.log('Tags updated successfully using fallback endpoint');
    return { success: true };
  } catch (error) {
    console.error('Exception in updateChatTags:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return { error: `Failed to update chat tags: ${errorMessage}` };
  }
};

// Delete a chat
export const deleteChat = async (chatId: string): Promise<{ success?: boolean; error?: string }> => {
  const response = await del(`/chats/${chatId}`);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Send a message to a chat (just stores the message, doesn't get LLM response)
export const sendMessage = async (chatId: string, content: string): Promise<{ message?: Message; error?: string }> => {
  const response = await post<Message>(`/chats/${chatId}/messages`, { role: 'user', content });

  if (response.error) {
    return { error: response.error };
  }

  return { message: response.data };
};

// Update a message
export const updateMessage = async (
  chatId: string,
  messageId: string,
  content: string
): Promise<{ message?: Message; error?: string }> => {
  const response = await put<Message>(`/chats/${chatId}/messages/${messageId}`, { content });

  if (response.error) {
    return { error: response.error };
  }

  return { message: response.data };
};

// Send a message to the LLM and get a response
export const sendMessageToLLM = async (chatId: string, content: string): Promise<{ message?: Message; error?: string }> => {
  const response = await post<Message>(`/chats/${chatId}/llm`, { role: 'user', content });

  if (response.error) {
    return { error: response.error };
  }

  return { message: response.data };
};

// Stream a message from the LLM
export const streamMessage = async (
  chatId: string,
  content: string,
  onChunk: (chunk: string) => void,
  onComplete: (message: Message) => void,
  onError: (error: string) => void
): Promise<void> => {
  try {
    // Check if the message contains keywords likely to trigger tool calls
    // --- Removed problematic check for mightTriggerToolCall ---
    // All messages will now go through the streaming endpoint below,
    // which correctly handles the backend logic for potential tool calls.

    // Note: We don't need to add the user message here as it's already added in the backend
    // and we're adding it to the UI in the chat component
    
    // Get token for authentication
    const token = getToken();
    if (!token) {
      throw new Error('No authentication token found');
    }

    // Get the correct URL for EventSource with auth token in query param
    const streamUrl = getApiUrl(
      `/chats/${chatId}/stream?content=${encodeURIComponent(content)}&token=${encodeURIComponent(token)}`,
      false
    );
    
    console.log('Setting up EventSource at:', streamUrl);
    
    // Set up event source for SSE
    const eventSource = new EventSource(streamUrl);
    
    let fullContent = '';
    let metadata: {
      tokens?: number;
      tokens_per_second?: number;
      model?: string;
      provider?: string;
    } | null = null;
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received event data snippet:', event.data.substring(0, 50) + '...');
        
        // Update full content
        fullContent = data.content;
        
        // Send the content to the callback - the frontend will handle displaying it
        onChunk(data.content);
        
        // Store metadata if available
        if (data.metadata) {
          metadata = data.metadata;
        }
        
        // If this is the final chunk, complete the process
        if (data.done) {
          console.log('Stream complete. Closing EventSource.');
          eventSource.close();
          
          // Construct the complete message
          const message: Message = {
            id: Date.now(), // Temporary ID until we fetch the actual message
            chat_id: parseInt(chatId, 10), // Convert string ID to number
            role: 'assistant',
            content: fullContent,
            created_at: new Date().toISOString(),
            tokens: metadata?.tokens,
            tokens_per_second: metadata?.tokens_per_second,
            model: metadata?.model,
            provider: metadata?.provider
          };
          
          onComplete(message);
        }
      } catch (e) {
        console.error('Error processing event data:', e);
        onError('Error processing streaming response');
        eventSource.close();
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      onError('Connection error during streaming');
      eventSource.close();
    };
  } catch (error) {
    console.error('Error in streamMessage:', error);
    onError((error as Error).message);
  }
};

// Submit feedback for a message
export const submitFeedback = async (
  chatId: string,
  messageId: string,
  feedback: string,
  feedbackText?: string
): Promise<{ message?: Message; error?: string }> => {
  const response = await post<Message>(`/chats/${chatId}/messages/${messageId}/feedback`, {
    feedback,
    feedback_text: feedbackText,
  });

  if (response.error) {
    return { error: response.error };
  }

  return { message: response.data };
};

// Get paginated messages with feedback (admin only)
export const getAdminFeedbackMessages = async (
  params: {
    feedbackType?: string;
    reviewed?: boolean;
    page?: number;
    pageSize?: number;
  } = {}
): Promise<{
  data?: PaginatedResponse<Message>; // Use the generic PaginatedResponse type
  error?: string;
}> => {
  const { feedbackType, reviewed, page = 1, pageSize = 20 } = params;
  const skip = (page - 1) * pageSize;
  const limit = pageSize;

  const queryParams: Record<string, string> = {
    skip: skip.toString(),
    limit: limit.toString(),
  };

  if (feedbackType) {
    queryParams.feedback_type = feedbackType;
  }
  if (reviewed !== undefined) {
    queryParams.reviewed = reviewed.toString();
  }

  // Use the standard 'get' function with query parameters
  const response = await get<PaginatedResponse<Message>>('/chats/admin/feedback', queryParams);

  if (response.error) {
    return { error: response.error };
  }

  // Ensure the response matches the PaginatedResponse structure if needed,
  // or adjust the backend to return exactly this structure.
  // Assuming the backend returns { items, total, page, size, pages }
  return { data: response.data };
};


// Get all chats with negative feedback (admin only) - Legacy function for compatibility
export const getFlaggedChats = async (params?: PaginationParams): Promise<{
  chats?: PaginatedResponse<Chat>;
  error?: string;
}> => {
  const response = await getPaginated<Chat>('/chats/admin/chats/flagged', params);

  if (response.error) {
    return { error: response.error };
  }

  return { chats: response.data };
};

// Mark message as reviewed (admin only)
export const markMessageAsReviewed = async (messageId: string): Promise<{ message?: Message; error?: string }> => {
  const response = await put<Message>(`/chats/admin/messages/${messageId}`, { reviewed: true });

  if (response.error) {
    return { error: response.error };
  }

  return { message: response.data };
};
