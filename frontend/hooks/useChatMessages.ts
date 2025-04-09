import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/router';
import { Chat, Message } from '@/types';
import { submitFeedback, updateMessage, createChat, updateChat, getChat } from '@/services/chat';
import { getApiUrl, getToken, post } from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import { useNotification } from '@/contexts/NotificationContext';
import { announce } from '@/utils/accessibilityUtils';
import { FeedbackType } from '@/components/chat/FeedbackButton'; // Assuming FeedbackType is defined here

export interface UseChatMessagesReturn {
  isStreaming: boolean;
  isWaitingForResponse: boolean; // Expose new state
  error: string | null; // Error specific to messaging
  messagesEndRef: React.RefObject<HTMLDivElement>;
  handleSendMessage: (messageContent: string, contextDocuments?: string[]) => Promise<void>;
  handleFeedback: (messageId: string, feedback: FeedbackType, feedbackText?: string) => Promise<void>;
  handleUpdateMessage: (messageId: string, newContent: string) => Promise<boolean>;
  closeEventSource: () => void; // Expose close function if needed externally
  refreshChat: () => Promise<void>; // Expose refresh function
  toolErrors: {[key: string]: any}; // Expose tool errors
}

export const useChatMessages = (
  currentChat: Chat | null,
  setCurrentChat: React.Dispatch<React.SetStateAction<Chat | null>>,
  // Pass setChats to update list when title changes on first message
  setChats: React.Dispatch<React.SetStateAction<Chat[]>>,
  // Pass loadChats to refresh list after streaming completes or new chat created
  loadChats: () => Promise<void>
): UseChatMessagesReturn => {
  const { isAuthenticated } = useAuth();
  const { showNotification } = useNotification();
  const router = useRouter();

  const [isStreaming, setIsStreaming] = useState(false);
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false); // New state for initial wait
  const [error, setError] = useState<string | null>(null);
  const [needsRefresh, setNeedsRefresh] = useState(false); // Add state for tracking refresh needs
  const [toolErrors, setToolErrors] = useState<{[key: string]: any}>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  
  const closeEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      console.log('Closing existing EventSource connection');
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  // Clean up EventSource on unmount or when chat changes significantly
  useEffect(() => {
    return () => {
      closeEventSource();
    };
  }, [closeEventSource, currentChat?.id]); // Close if chat ID changes

  // Scroll to bottom when messages change or during streaming
  useEffect(() => {
    const scrollTimeout = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({
        behavior: isStreaming ? 'auto' : 'smooth',
        block: 'end'
      });
    }, 100); // Increased timeout slightly

    return () => clearTimeout(scrollTimeout);
  }, [currentChat?.messages, isStreaming]);

  // Add a useEffect to refresh the chat data when needed
  useEffect(() => {
    // If we have a chat ID but no messages, or if we need to refresh after a tool call
    if (currentChat?.id && (!currentChat?.messages || currentChat.messages.length === 0 || needsRefresh)) {
      loadChats(); // Use the passed-in function to refresh/load chats
      setNeedsRefresh(false);
    }
  }, [currentChat?.id, currentChat?.messages, needsRefresh]);


  const handleFeedback = async (messageId: string, feedback: FeedbackType, feedbackText?: string) => {
    if (!currentChat) return;
    setError(null);
    try {
      const { message: updatedMessage, error: feedbackError } = await submitFeedback(
        String(currentChat.id),
        messageId,
        feedback,
        feedbackText
      );

      if (feedbackError) {
        throw new Error(feedbackError);
      }

      // Update the message in the UI
      setCurrentChat((prev: Chat | null) => {
        if (!prev) return null;
        return {
          ...prev,
          messages: prev.messages?.map((msg: Message) =>
            String(msg.id) === messageId
              ? { ...msg, feedback, feedback_text: feedbackText }
              : msg
          )
        };
      });

      showNotification('Feedback submitted successfully', 'success');
      announce({ message: 'Feedback submitted successfully', politeness: 'polite' });
    } catch (err) {
      console.error('Error submitting feedback:', err);
      const errorMsg = err instanceof Error ? err.message : 'Failed to submit feedback';
      setError(errorMsg);
      showNotification(errorMsg, 'error');
    }
  };

  const handleUpdateMessage = async (messageId: string, newContent: string): Promise<boolean> => {
    if (!currentChat) return false;
    setError(null);
    try {
      const { message: updatedMessage, error: updateError } = await updateMessage(
        currentChat.id,
        messageId,
        newContent
      );

      if (updateError) {
        throw new Error(updateError);
      }

      // Update the message in the UI
      if (updatedMessage) {
        setCurrentChat((prev: Chat | null) => {
          if (!prev) return null;
          return {
            ...prev,
            messages: prev.messages?.map((msg: Message) =>
              String(msg.id) === messageId ? { ...updatedMessage } : msg // Use the full updated message from backend
            )
          };
        });
        showNotification('Message updated successfully', 'success');
        return true;
      }
      return false;
    } catch (err) {
      console.error('Error updating message:', err);
      const errorMsg = err instanceof Error ? err.message : 'Failed to update message';
      setError(errorMsg);
      showNotification(errorMsg, 'error');
      return false;
    }
  };

  // Set up EventSource for streaming
  const setupEventSource = (chatId: string, content: string, contextDocuments?: string[]) => {
    closeEventSource(); // Ensure previous connection is closed

    const token = getToken();
    if (!token) {
      throw new Error('No authentication token found');
    }

    let streamUrl = `/chats/${chatId}/stream?content=${encodeURIComponent(content)}`;
    if (contextDocuments && contextDocuments.length > 0) {
        const contextParam = contextDocuments.join(',');
        streamUrl += `&context_documents=${encodeURIComponent(contextParam)}`;
    }
    streamUrl += `&token=${encodeURIComponent(token)}&_=${Date.now()}`; // Cache buster

    const fullUrl = getApiUrl(streamUrl, false); // useBaseUrl = false for EventSource
    console.log('Setting up EventSource connection to:', fullUrl);
    console.log('Using token for EventSource:', token.substring(0, 10) + '...');

    const eventSource = new EventSource(fullUrl);
    eventSourceRef.current = eventSource;
    return eventSource;
  };

  // Handle incoming EventSource messages
  const handleEventMessage = (event: MessageEvent) => {
    try {
      // console.log('Received event data:', event.data.substring(0, 100) + '...'); // Log less verbosely
      const data = JSON.parse(event.data);

      // Clear waiting state on first received chunk
      if (isWaitingForResponse) {
        setIsWaitingForResponse(false);
      }

      if (data.error) {
        setError(data.content || 'An error occurred during streaming');
        closeEventSource();
        setIsStreaming(false);
        setIsWaitingForResponse(false);
        return;
      }

      // Update the last assistant message content
      setCurrentChat((prev: Chat | null) => {
        if (!prev || !prev.messages) return prev;

        const updatedMessages = [...prev.messages];
        const lastAssistantIndex = updatedMessages.findLastIndex((msg: Message) => msg.role === 'assistant');

        if (lastAssistantIndex !== -1) {
          // Check if the message content is already updated to avoid duplicates
          if (updatedMessages[lastAssistantIndex].content === data.content) {
            return prev; // No change needed
          }
          
          // Update the existing placeholder or last assistant message
          updatedMessages[lastAssistantIndex] = {
            ...updatedMessages[lastAssistantIndex],
            content: data.content, // Backend sends full content
            tokens: data.tokens,
            tokens_per_second: data.tokens_per_second,
            model: data.model,
            provider: data.provider,
            // Add tool_calls if they exist in the data
            tool_calls: data.tool_calls || updatedMessages[lastAssistantIndex].tool_calls,
            // Update document_ids if they arrive during streaming
            document_ids: data.document_ids || updatedMessages[lastAssistantIndex].document_ids,
            // Update id if it wasn't set before (e.g., placeholder) or if backend provides it
            id: data.id || updatedMessages[lastAssistantIndex].id,
            // Update created_at if backend provides it
            created_at: data.created_at || updatedMessages[lastAssistantIndex].created_at,
          };
        } else {
          // Should not happen if placeholder was added, but handle defensively
          console.error("Streaming update: Couldn't find last assistant message to update.");
        }

        return { ...prev, messages: updatedMessages };
      });

      // Add handling for tool result messages
      if (data.role === 'tool') {
        setCurrentChat((prev: Chat | null) => {
          if (!prev || !prev.messages) return prev;
          
          // Find the assistant message that contains the tool call
          const assistantMessage = prev.messages.find((msg: Message) =>
            msg.role === 'assistant' &&
            msg.tool_calls &&
            msg.tool_calls.some((tc: any) => tc.id === data.tool_call_id)
          );
          
          // Add the tool result message to the messages array
          const toolResultMessage: Message = {
            id: typeof data.id === 'string' ? parseInt(data.id, 10) : data.id,
            chat_id: typeof prev.id === 'string' ? parseInt(prev.id, 10) : prev.id,
            role: 'tool',
            content: data.content,
            tool_call_id: data.tool_call_id,
            created_at: data.created_at || new Date().toISOString(),
          };
          
          const updatedMessages = [...prev.messages, toolResultMessage];
          
          // Update all messages to include parentMessages reference
          // We need to avoid circular references, so we'll only include necessary fields
          const messagesWithParents = updatedMessages.map((msg: Message) => {
            // Create a simplified version of the messages for the parentMessages property
            // to avoid circular references and excessive memory usage
            const parentMsgs = updatedMessages.map((parentMsg: Message) => ({
              id: parentMsg.id,
              role: parentMsg.role,
              content: parentMsg.content,
              tool_call_id: parentMsg.tool_call_id,
              tool_calls: parentMsg.tool_calls
            }));
            
            return {
              ...msg,
              parentMessages: parentMsgs
            };
          });
          
          // Set needsRefresh to true to trigger a refresh after tool call
          setNeedsRefresh(true);
          
          return { ...prev, messages: messagesWithParents as Message[] };
        });
      }

      if (data.done) {
        console.log('Received final chunk.');
        closeEventSource();
        setIsStreaming(false);
        setIsWaitingForResponse(false); // Ensure waiting state is reset when done
        // Refresh the chat list to ensure title/timestamp updates are reflected
        loadChats();
        // Refresh the current chat to get final message IDs and potentially other updates
        if (currentChat?.id) {
          const chatIdToRefresh = currentChat.id; // Capture ID before potential state changes
          getChat(chatIdToRefresh).then(result => {
            const fetchedChat = result.chat; // Assign to a constant first
            if (fetchedChat) { // Check the constant
              console.log('Refreshed current chat data from backend:', fetchedChat.id);
              setCurrentChat((prevChat: Chat | null): Chat | null => { // Explicitly define return type
                // If there's no previous chat, or the ID doesn't match the fetched chat,
                // use the newly fetched chat directly. fetchedChat is guaranteed non-null here.
                if (!prevChat || prevChat.id !== fetchedChat.id) { // Use the constant
                  return fetchedChat; // Use the constant
                }

                // --- If we are here, prevChat exists and IDs match. Merge messages. ---

                // Create a map of previous messages for efficient lookup
                const prevMessagesMap = new Map(prevChat.messages?.map((msg: Message) => [msg.id, msg]));

                // Ensure fetched messages is an array
                const fetchedMessages = fetchedChat.messages || []; // Use the constant

                // Map over fetched messages and merge with previous ones if necessary
                const finalMessages = fetchedMessages.map((fetchedMsg: Message) => {
                  const prevMsg = prevMessagesMap.get(fetchedMsg.id);

                  // If a previous message exists and it's an assistant message, merge token data
                  if (prevMsg && prevMsg.role === 'assistant') {
                    return {
                      ...fetchedMsg, // Start with the fetched message data
                      // Keep token info from prev state ONLY if missing in fetched state
                      tokens: fetchedMsg.tokens ?? prevMsg.tokens,
                      tokens_per_second: fetchedMsg.tokens_per_second ?? prevMsg.tokens_per_second,
                    };
                  }

                  // Otherwise (no previous message, or not an assistant message), use the fetched message as is
                  return fetchedMsg;
                });

                // Construct the final state: Start with prevChat, overlay fetched data (like updated_at),
                // and use the merged messages array. This ensures we return a valid Chat object.
                const updatedChat: Chat = {
                    ...prevChat,      // Base is the previous state
                    ...fetchedChat,   // Use the constant to overlay fields (e.g., updated_at)
                    messages: finalMessages, // Use the carefully merged messages
                };
                return updatedChat; // Return the correctly typed Chat object
              });
            } else {
              console.error('Failed to refresh chat after streaming:', result.error);
              showNotification(`Failed to refresh chat details: ${result.error}`, 'error');
            }
          });
        }
        showNotification('Response completed successfully', 'success');
        announce({ message: 'Response completed successfully', politeness: 'polite' });
      }
    } catch (e) {
      console.error('Error processing event data:', e);
      console.error('Raw event data:', event.data);
      setError('Error processing streaming response');
      closeEventSource();
      setIsStreaming(false);
      setIsWaitingForResponse(false); // Reset waiting state on error
    }
  };

  // Handle EventSource errors
  const handleEventError = (error: Event) => {
    console.error('EventSource error:', error);
    const eventSource = eventSourceRef.current;
    let errorMessage = 'Connection error during streaming.';
    if (eventSource && eventSource.readyState === EventSource.CLOSED) {
      errorMessage = 'Connection closed unexpectedly. Please try again.';
    }
    setError(errorMessage);
    showNotification(errorMessage, 'error');
    closeEventSource();
    setIsStreaming(false);
    setIsWaitingForResponse(false); // Reset waiting state on error
  };


  const handleSendMessage = async (messageContent: string, contextDocuments?: string[]) => {
    if (isStreaming || !isAuthenticated) return;
    setError(null);

    let chatId = currentChat?.id;
    let chatToUpdate = currentChat;

    // 1. Create new chat if necessary
    if (!chatToUpdate) {
      try {
        // Start with default title, will be updated later
        const { chat: newChat, error: createError } = await createChat("New Conversation");
        if (!newChat) {
          const errorMsg = createError || 'Failed to create chat';
          setError(errorMsg);
          showNotification(errorMsg, 'error');
          return;
        }
        chatToUpdate = newChat;
        chatId = newChat.id;
        setCurrentChat(chatToUpdate); // Set the newly created chat as current
        router.push(`/chat?id=${chatId}`, undefined, { shallow: true });
        await loadChats(); // Refresh list to show the new chat
        // Wait a moment for state/router updates
        await new Promise(resolve => setTimeout(resolve, 50));
      } catch (err) {
        console.error('Error creating new chat during send:', err);
        setError('Failed to initiate chat.');
        showNotification('Failed to initiate chat.', 'error');
        return;
      }
    }

    if (!chatId || !chatToUpdate) {
        setError('Chat session not available.');
        return;
    }

    // 2. Add user message optimistically
    const userMessage: Message = {
      id: -Date.now(), // Temporary ID (negative number)
      chat_id: typeof chatId === 'string' ? parseInt(chatId, 10) : chatId,
      role: 'user' as const,
      content: messageContent,
      created_at: new Date().toISOString(),
      context_documents: contextDocuments
    };

    // 3. Add assistant placeholder message optimistically
    const assistantMessage: Message = {
      id: -(Date.now() + 1), // Temporary ID (negative number, ensure different from user)
      chat_id: typeof chatId === 'string' ? parseInt(chatId, 10) : chatId,
      role: 'assistant' as const,
      content: '', // Placeholder
      created_at: new Date().toISOString(),
    };

    setCurrentChat((prev: Chat | null) => ({
      ...(prev || chatToUpdate!), // Use chatToUpdate if prev is null
      messages: [...(prev?.messages || chatToUpdate!.messages || []), userMessage, assistantMessage],
    }));

    // 4. Set streaming state and scroll
    setIsStreaming(true);
    setError(null);
    setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' }), 10);

    try {
      // 5. Update title if it's the first message
      if (chatToUpdate.title === "New Conversation") {
        const newTitle = messageContent.length > 30
          ? `${messageContent.substring(0, 30)}...`
          : messageContent;
        console.log('Updating chat title based on first message:', newTitle);
        const updateResult = await updateChat(chatId, { title: newTitle });
        if (updateResult.success) {
          setCurrentChat((prev: Chat | null) => prev ? { ...prev, title: newTitle } : null);
          setChats((prevChats: Chat[]) => // Update title in the main list as well
            prevChats.map((c: Chat) => c.id === chatId ? { ...c, title: newTitle } : c)
          );
          console.log('Successfully updated chat title in backend');
        } else {
          console.error('Failed to update chat title in backend:', updateResult.error);
          // Continue anyway, title update failure isn't critical for the stream
        }
      }

      // Set waiting state before starting stream
      setIsWaitingForResponse(true);

      // Check if the message is likely to trigger a tool call
      const isFetchToolCall = messageContent.toLowerCase().includes('fetch') &&
        (messageContent.toLowerCase().includes('http://') || messageContent.toLowerCase().includes('https://'));
      
      // Detect other potential tool calls - expanded to catch more cases
      const isToolCall = isFetchToolCall ||
        messageContent.toLowerCase().includes('use tool') ||
        messageContent.toLowerCase().includes('search for') ||
        messageContent.toLowerCase().includes('find information') ||
        messageContent.toLowerCase().includes('mcp') ||
        messageContent.toLowerCase().includes('tool call');
      
      console.log('Message content:', messageContent);
      console.log('Is tool call detected:', isToolCall);

      if (isToolCall) {
        console.log('Detected potential tool call, disabling streaming');
        
        // For tool calls, use the API utilities instead of direct fetch
        console.log('Using API utilities for potential tool call');
        
        // Use post from api.ts which automatically handles authentication
        const apiResponse = await post<any>(`/chats/${chatId}/messages`, {
          role: 'user', // Add the role field
          message: messageContent,
          stream: false, // Disable streaming for tool calls
          context_documents: contextDocuments
        });
        
        // Check if the response has an error property
        if (apiResponse.error) { // Line 500
          console.error('API error response:', apiResponse.error); // Line 501

          let errorMessage = 'Failed to send message';
          let isAuthError = false;
          // Attempt to get status from the response object if available (depends on api.ts implementation)
          const status = (apiResponse as any).status || null;

          // Safely check the error message, prioritizing specific fields
          let errorDetail = (apiResponse as any).detail || (apiResponse as any).message || apiResponse.error;

          if (typeof errorDetail === 'string') {
            errorMessage = errorDetail;
            // Check if this is an authentication error by status or content
            const lowerCaseError = errorDetail.toLowerCase();
            if (status === 401 ||
                lowerCaseError.includes('not authenticated') ||
                lowerCaseError.includes('unauthorized')) {
              isAuthError = true;
            }
          } else if (typeof errorDetail === 'object' && errorDetail !== null) {
            // If it's an object, try to get a message property or stringify
            errorMessage = (errorDetail as any).message || JSON.stringify(errorDetail);
            // Check for auth error primarily by status if the detail is an object
            if (status === 401) {
              isAuthError = true;
            }
            // Optional: Could also check stringified object content if needed
            // const lowerCaseError = JSON.stringify(errorDetail).toLowerCase();
            // if (!isAuthError && (lowerCaseError.includes('not authenticated') || lowerCaseError.includes('unauthorized'))) {
            //   isAuthError = true;
            // }
          } else {
            // Fallback if errorDetail is not a string or object
            errorMessage = status ? `API Error: ${status}` : 'Unknown API Error';
            // Ensure we check status 401 even in fallback
            if (status === 401) {
              isAuthError = true;
            }
          }

          if (isAuthError) {
            console.error('Authentication error:', errorMessage);
            
            // Check if the user is actually logged in before redirecting
            const isLoggedIn = await checkAuthStatus();
            
            if (!isLoggedIn) {
              // Only redirect if the user is not logged in
              if (typeof window !== 'undefined') {
                window.location.href = '/login?error=session_expired';
                return; // Return early to prevent further processing
              }
            } else {
              // User is logged in but got an auth error for this specific request
              setError('Authentication error. Please try again.');
              setIsWaitingForResponse(false);
              return; // Return early as auth error handled locally
            }
          }
          
          // If it wasn't an auth error that caused a redirect/return, throw the processed error
          // Ensure the error thrown is an Error object
          // Convert non-string errors to string before creating Error object
          const errorString = typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage);
          throw new Error(errorString);
        }

        // Handle the non-streaming response
        const data = apiResponse.data;
        // Update the user message with the actual ID from the server
        setCurrentChat((prev: Chat | null) => {
          if (!prev || !prev.messages) return prev;
          
          // Find the temporary user message and update its ID
          const updatedMessages = prev.messages.map(msg =>
            msg.id === userMessage.id ? { ...msg, id: data.user_message_id } : msg
          );
          
          // Find the temporary assistant message and replace it with the actual response
          const assistantIndex = updatedMessages.findIndex(msg => msg.id === assistantMessage.id);
          if (assistantIndex !== -1) {
            updatedMessages[assistantIndex] = {
              id: data.id,
              chat_id: typeof chatId === 'string' ? parseInt(chatId, 10) : chatId,
              role: 'assistant',
              content: data.content,
              tokens: data.tokens,
              tokens_per_second: data.tokens_per_second,
              model: data.model,
              provider: data.provider,
              tool_calls: data.tool_calls,
              document_ids: data.document_ids,
              created_at: data.created_at,
            };
          }
          
          // Add tool result messages if any
          if (data.tool_results && data.tool_results.length > 0) {
            data.tool_results.forEach((toolResult: any) => {
              updatedMessages.push({
                id: typeof toolResult.id === 'string' ? parseInt(toolResult.id, 10) : toolResult.id,
                chat_id: typeof chatId === 'string' ? parseInt(chatId, 10) : chatId,
                role: 'tool',
                content: toolResult.content,
                tool_call_id: toolResult.tool_call_id,
                created_at: toolResult.created_at || new Date().toISOString(),
              });
            });
          }
          
          return {
            ...prev,
            messages: updatedMessages,
          };
        });
        
        setIsStreaming(false);
        setIsWaitingForResponse(false);
        
        // Refresh the chat to ensure we have the latest state
        refreshChat();
        
        // Refresh the chat list to ensure title/timestamp updates are reflected
        loadChats();
        
        announce({ message: 'Response with tool call completed', politeness: 'polite' });
      } else {
        // For regular messages, use streaming as before
        const eventSource = setupEventSource(chatId, messageContent, contextDocuments);
        eventSource.onmessage = handleEventMessage;
        eventSource.onerror = (error) => {
          handleEventError(error);
          
          // Check if this might be an authentication error
          // EventSource doesn't provide detailed error information, so we need to check auth status
          checkAuthStatus().then(isAuthenticated => {
            if (!isAuthenticated) {
              // Authentication error - only redirect if actually not authenticated
              if (typeof window !== 'undefined') {
                window.location.href = '/login?error=session_expired';
              }
            } else {
              // User is authenticated but still got an error
              // This might be a temporary issue, so just show an error message
              setError('Connection error. Please try again.');
              setIsWaitingForResponse(false);
            }
          });
        };

        announce({ message: 'Message sent, waiting for response', politeness: 'polite' });
      }
    } catch (err) {
      console.error('Error sending message:', err);
      
      // Parse the error
      let errorMessage = 'An error occurred';
      let errorDetails = null;
      
      if (err instanceof Error) {
        errorMessage = err.message;
        errorDetails = err;
      } else if (typeof err === 'object' && err !== null) {
        // Use type assertion to tell TypeScript that err might have a message property
        errorMessage = (err as { message?: string }).message || JSON.stringify(err);
        errorDetails = err;
      } else if (typeof err === 'string') {
        errorMessage = err;
      }
      
      setError(`Failed to send message: ${errorMessage}`);
      showNotification(`Failed to send message: ${errorMessage}`, 'error');
      
      // Check if this was a tool call
      const isToolCall = currentChat?.messages && currentChat.messages.some(msg =>
        msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0
      );
      
      // If this was a tool call, set the error for the specific tool
      if (isToolCall && currentChat?.messages) {
        const lastAssistantMessage = [...currentChat.messages]
          .reverse()
          .find(msg => msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0);
        
        if (lastAssistantMessage && lastAssistantMessage.tool_calls) {
          // Set error for each tool call in the message
          lastAssistantMessage.tool_calls.forEach(toolCall => {
            setToolErrors((prev: {[key: string]: any}) => ({
              ...prev,
              [toolCall.id]: errorDetails || errorMessage
            }));
          });
        }
      }
      
      // Rollback optimistic UI updates? Remove placeholder messages?
      setCurrentChat((prev: Chat | null) => {
          if (!prev) return null;
          // Remove the last two messages (user + assistant placeholder)
          const messages = prev.messages?.slice(0, -2) || [];
          return {...prev, messages };
      });
      setIsStreaming(false);
      setIsWaitingForResponse(false); // Reset waiting state on error
      closeEventSource();
    }
  };

// Function to check authentication status
const checkAuthStatus = async (): Promise<boolean> => {
  try {
    // First, check if we have a token
    const token = getToken();
    if (!token) {
      console.error('No token found in storage');
      return false;
    }
    
    console.log('Checking auth status with token:', token.substring(0, 10) + '...');
    
    // Then, check if the token is valid by making a request to the auth check endpoint
    const response = await fetch('/api/v1/auth/check', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    const isAuthenticated = response.ok;
    console.log('Auth check response:', response.status, 'Authenticated:', isAuthenticated);
    
    return isAuthenticated;
  } catch (error) {
    console.error('Error checking authentication status:', error);
    return false;
  }
};

// Add a function to refresh the chat data
const refreshChat = async () => {
  if (!currentChat?.id) return;
  
  try {
    console.log('Refreshing chat data for chat ID:', currentChat.id);
    const { chat, error: fetchError } = await getChat(currentChat.id);
    
    if (fetchError) {
      throw new Error(fetchError);
    }
    
    if (chat) {
      setCurrentChat(chat);
      console.log('Chat data refreshed successfully');
    }
  } catch (error) {
    console.error('Error refreshing chat:', error);
  }
};
return {
  isStreaming,
  isWaitingForResponse, // Add new state to return object
  error,
  messagesEndRef,
  handleSendMessage,
  handleFeedback,
  handleUpdateMessage,
  closeEventSource,
  refreshChat, // Expose the refresh function
  toolErrors // Expose tool errors
};
};