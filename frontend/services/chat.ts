import { Chat, Feedback, Message, PaginatedResponse, PaginationParams } from '@/types';
import { del, get, getPaginated, post, put, getApiUrl } from './api';

// Get all chats
export const getChats = async (): Promise<{
  chats?: Chat[];
  error?: string;
}> => {
  const response = await get<Chat[]>('/chats');

  if (response.error) {
    return { error: response.error };
  }

  return { chats: response.data };
};

// Get a single chat by ID
export const getChat = async (chatId: string): Promise<{ chat?: Chat; error?: string }> => {
  const response = await get<Chat>(`/chats/${chatId}`);

  if (response.error) {
    return { error: response.error };
  }

  return { chat: response.data };
};

// Create a new chat
export const createChat = async (title: string): Promise<{ chat?: Chat; error?: string }> => {
  const response = await post<Chat>('/chats', { title });

  if (response.error) {
    return { error: response.error };
  }

  return { chat: response.data };
};

// Update a chat
export const updateChat = async (
  chatId: string,
  title: string
): Promise<{ success?: boolean; error?: string }> => {
  const response = await put(`/chats/${chatId}`, { title });

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Delete a chat
export const deleteChat = async (
  chatId: string
): Promise<{ success?: boolean; error?: string }> => {
  const response = await del(`/chats/${chatId}`);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Send a message to a chat (just stores the message, doesn't get LLM response)
export const sendMessage = async (
  chatId: string,
  content: string
): Promise<{ message?: Message; error?: string }> => {
  const response = await post<Message>(`/chats/${chatId}/messages`, { role: 'user', content });

  if (response.error) {
    return { error: response.error };
  }

  return { message: response.data };
};

// Send a message to the LLM and get a response
export const sendMessageToLLM = async (
  chatId: string,
  content: string
): Promise<{ message?: Message; error?: string }> => {
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
    // Note: We don't need to add the user message here as it's already added in the backend
    // and we're adding it to the UI in the chat component

    // Get token for authentication
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('No authentication token found');
    }

    // Get the correct URL for EventSource with auth token in query param
    const streamUrl = getApiUrl(
      `/chats/${chatId}/stream?content=${encodeURIComponent(content)}&token=${encodeURIComponent(token)}`,
      false
    );

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
            provider: metadata?.provider,
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

// Get all messages with feedback (admin only)
export const getFeedbackMessages = async (
  feedbackType?: string,
  reviewed?: boolean
): Promise<{
  messages?: Message[];
  error?: string;
}> => {
  let url = '/chats/admin/feedback';
  const params = new URLSearchParams();

  if (feedbackType) {
    params.append('feedback_type', feedbackType);
  }

  if (reviewed !== undefined) {
    params.append('reviewed', reviewed.toString());
  }

  const queryString = params.toString();
  if (queryString) {
    url += `?${queryString}`;
  }

  const response = await get<Message[]>(url);

  if (response.error) {
    return { error: response.error };
  }

  return { messages: response.data };
};

// Get all chats with negative feedback (admin only) - Legacy function for compatibility
export const getFlaggedChats = async (
  params?: PaginationParams
): Promise<{
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
export const markMessageAsReviewed = async (
  messageId: string
): Promise<{ message?: Message; error?: string }> => {
  const response = await put<Message>(`/chats/admin/messages/${messageId}`, { reviewed: true });

  if (response.error) {
    return { error: response.error };
  }

  return { message: response.data };
};
