import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useRouter } from 'next/router';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import Layout from '@/components/layout/Layout';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent } from '@/components/ui/Card';
import { useAuth } from '@/contexts/AuthContext';
import { Chat, Message, Document } from '@/types';
import {
  getChats,
  createChat,
  getChat,
  streamMessage,
  deleteChat,
  submitFeedback,
} from '@/services/chat';
import { getDocument } from '@/services/document';
import { getChunkInfo } from '@/services/rag';
import { getApiUrl } from '@/services/api';
import { parseThinkTags } from '@/utils/thinkTagParser';

// Document References component
// Cache for document information to avoid redundant API calls
const chunkInfoCache: {
  [key: string]: {
    documentId: string;
    documentTitle: string;
    loading: boolean;
    error: boolean;
  };
} = {};

const DocumentReferences = ({ documentIds }: { documentIds: string[] }) => {
  const [chunkInfo, setChunkInfo] = useState<{
    [key: string]: {
      documentId: string;
      documentTitle: string;
      loading: boolean;
      error: boolean;
    };
  }>({});

  useEffect(() => {
    // Initialize state with cached values or defaults
    const initialState: {
      [key: string]: {
        documentId: string;
        documentTitle: string;
        loading: boolean;
        error: boolean;
      };
    } = {};

    documentIds.forEach((chunkId) => {
      if (chunkInfoCache[chunkId]) {
        // Use cached value if available
        initialState[chunkId] = chunkInfoCache[chunkId];
      } else {
        // Initialize with loading state
        initialState[chunkId] = {
          documentId: '',
          documentTitle: '',
          loading: true,
          error: false,
        };

        // Fetch chunk information
        getChunkInfo(chunkId)
          .then(({ info, error }) => {
            // Log the response for debugging
            console.log(`Chunk info for ${chunkId}:`, { info, error });

            if (info) {
              // Update cache and state with document info
              const chunkData = {
                documentId: info.document_id,
                documentTitle: info.document_title,
                loading: false,
                error: false,
              };
              console.log(`Setting chunk data for ${chunkId}:`, chunkData);
              chunkInfoCache[chunkId] = chunkData;
              setChunkInfo((prev) => ({ ...prev, [chunkId]: chunkData }));
            } else {
              // Handle error
              console.error(`Error getting chunk info for ${chunkId}:`, error);
              const chunkData = {
                documentId: '',
                documentTitle: '',
                loading: false,
                error: true,
              };
              chunkInfoCache[chunkId] = chunkData;
              setChunkInfo((prev) => ({ ...prev, [chunkId]: chunkData }));
            }
          })
          .catch((err) => {
            // Handle fetch error
            console.error(`Exception getting chunk info for ${chunkId}:`, err);
            const chunkData = {
              documentId: '',
              documentTitle: '',
              loading: false,
              error: true,
            };
            chunkInfoCache[chunkId] = chunkData;
            setChunkInfo((prev) => ({ ...prev, [chunkId]: chunkData }));
          });
      }
    });

    setChunkInfo(initialState);
  }, [documentIds]);

  return (
    <div>
      <ul className="list-disc list-inside mb-2">
        {documentIds.map((chunkId, i) => {
          // Format the chunk ID for display
          const shortId = chunkId.length > 12 ? `${chunkId.substring(0, 8)}...` : chunkId;
          const info = chunkInfo[chunkId] || {
            documentId: '',
            documentTitle: '',
            loading: true,
            error: false,
          };

          return (
            <li key={chunkId} className="text-gray-600 dark:text-gray-400 truncate mb-1">
              <span className="font-medium">Document {i + 1}:</span>{' '}
              {info.loading ? (
                <span className="text-gray-400">Loading...</span>
              ) : info.error || !info.documentTitle ? (
                // Fallback to just showing the chunk ID with a search link
                <>
                  <span>{shortId}</span>
                  <a
                    href={`/admin/documents?search=${chunkId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-2 text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 text-xs"
                    title="Search for this document in admin"
                  >
                    Search in admin
                  </a>
                </>
              ) : (
                // Show document title and link when available
                <>
                  <span className="font-medium">{info.documentTitle}</span>
                  <a
                    href={`/admin/documents?search=${info.documentId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-2 text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 text-xs"
                    title="View document in admin"
                  >
                    View document
                  </a>
                  <span className="text-gray-400 text-xs ml-1">({shortId})</span>
                </>
              )}
            </li>
          );
        })}
      </ul>
      <p className="text-xs text-gray-500 dark:text-gray-500 italic">
        Note: These IDs refer to document chunks used by the RAG system. The system will attempt to
        retrieve the original document titles, but some chunks may no longer exist in the database.
      </p>
    </div>
  );
};

// Feedback types
type FeedbackType = 'positive' | 'negative';

const FeedbackButton = ({
  messageId,
  type,
  label,
  onClick,
  icon,
}: {
  messageId: string;
  type: FeedbackType;
  label: string;
  onClick: () => void;
  icon: React.ReactNode;
}) => (
  <button
    onClick={onClick}
    className={`p-1 ${
      type === 'positive'
        ? 'text-green-500 hover:bg-green-100 dark:hover:bg-green-900/30'
        : 'text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30'
    } rounded`}
    title={label}
  >
    {icon}
  </button>
);

const MessageContent = ({ content, message }: { content: string; message: Message }) => {
  const [collapsedThinkTags, setCollapsedThinkTags] = useState<{ [key: number]: boolean }>({});
  // Parse think tags using memoization to avoid unnecessary re-parsing
  const parts = useMemo(() => parseThinkTags(content), [content]);

  // Toggle think tag collapse state
  const toggleThinkTag = (index: number) => {
    setCollapsedThinkTags((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  // Custom components for ReactMarkdown
  const components = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter style={vscDarkPlus} language={match[1]} PreTag="div" {...props}>
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <div className="relative group">
      <div>
        <div className="message-parts mb-4">
          {parts.map((part, index) =>
            part.isThink ? (
              <div
                key={index}
                className={`think-tag ${part.isComplete ? 'complete' : 'incomplete'} ${collapsedThinkTags[index] ? 'collapsed' : 'expanded'}`}
              >
                {part.isComplete && (
                  <div
                    className="think-tag-header cursor-pointer flex items-center text-xs text-gray-500 mb-1"
                    onClick={() => toggleThinkTag(index)}
                  >
                    <span className="mr-1">{collapsedThinkTags[index] ? '▶' : '▼'}</span>
                    <span>Thinking</span>
                  </div>
                )}

                {(!part.isComplete || !collapsedThinkTags[index]) && (
                  <div className="think-tag-content bg-gray-50 dark:bg-gray-800/50 p-2 rounded border border-gray-200 dark:border-gray-700">
                    <div
                      className={
                        message.role === 'user'
                          ? 'user-message-content'
                          : 'assistant-message-content'
                      }
                    >
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                        components={components}
                        className={
                          message.role === 'user'
                            ? 'user-message-markdown'
                            : 'assistant-message-markdown'
                        }
                      >
                        {part.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div
                className={
                  message.role === 'user' ? 'user-message-content' : 'assistant-message-content'
                }
              >
                <ReactMarkdown
                  key={index}
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={components}
                  className={
                    message.role === 'user' ? 'user-message-markdown' : 'assistant-message-markdown'
                  }
                >
                  {part.content}
                </ReactMarkdown>
              </div>
            )
          )}
        </div>
        {/* Document references are now shown in the info panel */}
      </div>
    </div>
  );
};

export default function ChatPage() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const router = useRouter();
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  // Load chats function
  const loadChats = async () => {
    if (!isAuthenticated) return;

    try {
      const { chats, error } = await getChats();
      if (chats) {
        setChats(chats);
      } else if (error) {
        setError(`Failed to load chats: ${error}`);
      } else {
        // If no chats data and no error, set empty array
        setChats([]);
      }
    } catch (err) {
      console.error('Error loading chats:', err);
      setError('An unexpected error occurred while loading chats.');
      // Ensure chats is initialized to an empty array on error
      setChats([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Load chats on mount and when auth changes
  useEffect(() => {
    loadChats();
  }, [isAuthenticated]);

  // Load chat if ID is in URL
  useEffect(() => {
    const chatId = router.query.id ? String(router.query.id) : null;

    const loadChat = async () => {
      if (!chatId || !isAuthenticated) return;

      try {
        const { chat, error } = await getChat(chatId);
        if (chat) {
          setCurrentChat(chat);
        } else if (error) {
          setError(`Failed to load chat: ${error}`);
        }
      } catch (err) {
        console.error('Error loading chat:', err);
        setError('An unexpected error occurred while loading the chat.');
      }
    };

    loadChat();
  }, [router.query.id, isAuthenticated, router]);

  // Scroll to bottom when messages change or during streaming
  useEffect(() => {
    // Use a small timeout to ensure the DOM has updated before scrolling
    const scrollTimeout = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({
        behavior: isStreaming ? 'auto' : 'smooth',
        block: 'end',
      });
    }, 10);

    return () => clearTimeout(scrollTimeout);
  }, [currentChat?.messages, isStreaming]);

  const handleNewChat = () => {
    // Clear all relevant state
    setCurrentChat(null);
    setMessage('');
    setError(null);
    setIsStreaming(false);
    // Force a re-render of the empty chat state
    router.push('/chat', undefined, { shallow: true });
  };

  const handleSelectChat = (chat: Chat) => {
    router.push(`/chat?id=${chat.id}`);
  };

  const handleDeleteChat = async (chatId: string) => {
    console.log('Deleting chat:', chatId);
    setIsLoading(true);
    setError(null);

    try {
      // Optimistically remove the chat from the UI
      setChats((prevChats) => prevChats.filter((chat) => chat.id !== chatId));

      const { success, error } = await deleteChat(chatId);
      console.log('Delete response:', { success, error });

      if (success) {
        console.log('Delete successful, updating UI...');
        if (currentChat?.id === chatId) {
          console.log('Current chat was deleted, redirecting...');
          setCurrentChat(null);
          router.push('/chat');
        }
        // Reload the full list to ensure consistency
        console.log('Reloading chat list...');
        await loadChats();
      } else {
        console.error('Delete failed with error:', error);
        setError(`Failed to delete chat: ${error}`);
        // Restore the chat list since deletion failed
        await loadChats();
      }
    } catch (err) {
      console.error('Error deleting chat:', err);
      setError('An unexpected error occurred while deleting the chat.');
      // Restore the chat list since deletion failed
      await loadChats();
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (
    messageId: string,
    feedback: FeedbackType,
    feedbackText?: string
  ) => {
    try {
      const { message: updatedMessage, error: feedbackError } = await submitFeedback(
        String(currentChat?.id),
        messageId,
        feedback,
        feedbackText
      );

      if (feedbackError) {
        throw new Error(feedbackError);
      }

      // Update the message in the UI
      setCurrentChat((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          messages: prev.messages?.map((msg) =>
            String(msg.id) === messageId ? { ...msg, feedback, feedback_text: feedbackText } : msg
          ),
        };
      });
    } catch (err) {
      console.error('Error submitting feedback:', err);
      setError('Failed to submit feedback');
    }
  };

  // Keep track of active EventSource to prevent multiple connections
  const eventSourceRef = useRef<EventSource | null>(null);

  // Function to close any existing EventSource
  const closeEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      console.log('Closing existing EventSource connection');
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  // Clean up EventSource on unmount
  useEffect(() => {
    return () => {
      closeEventSource();
    };
  }, [closeEventSource]);

  // Set up EventSource for streaming
  const setupEventSource = (chatId: string, content: string) => {
    // Close any existing EventSource
    closeEventSource();

    // Get token for authentication
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('No authentication token found');
    }

    // Create URL with cache-busting parameter
    const timestamp = Date.now();
    const streamUrl = getApiUrl(
      `/chats/${chatId}/stream?content=${encodeURIComponent(content)}&token=${encodeURIComponent(token)}&_=${timestamp}`,
      false
    );

    console.log('Setting up EventSource connection to:', streamUrl);

    // Create EventSource
    const eventSource = new EventSource(streamUrl);
    eventSourceRef.current = eventSource;

    return eventSource;
  };

  // Handle incoming EventSource messages
  const handleEventMessage = (event: MessageEvent) => {
    try {
      // Log only basic info to reduce console noise
      console.log('Received event data:', event.data.substring(0, 50) + '...');

      const data = JSON.parse(event.data);

      // Check for error condition
      if (data.error) {
        setError(data.content || 'An error occurred during streaming');
        closeEventSource();
        setIsStreaming(false);
        return;
      }

      // Force a refresh of the current chat to ensure we have the latest state
      setCurrentChat((prev) => {
        if (!prev) return prev;
        // Find the last assistant message (searching from the end of the array)
        const messages = prev.messages || [];
        let lastAssistantIndex = -1;

        // Search from the end of the array to find the last assistant message
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i].role === 'assistant') {
            lastAssistantIndex = i;
            break;
          }
        }

        if (lastAssistantIndex === -1) return prev;

        // Create a new messages array with the updated content
        const updatedMessages = [...messages];

        // Update the assistant message with the new content from the server
        // The backend sends the full content in each chunk, not just the delta
        updatedMessages[lastAssistantIndex] = {
          ...updatedMessages[lastAssistantIndex],
          content: data.content,
          tokens: data.tokens,
          tokens_per_second: data.tokens_per_second,
          model: data.model,
          provider: data.provider,
        };

        // Return a new chat object with the updated messages
        return {
          ...prev,
          messages: updatedMessages,
        };
      });

      // If this is the final chunk, complete the process
      if (data.done) {
        console.log('Received final chunk, closing EventSource');
        closeEventSource();
        setIsStreaming(false);

        // Refresh the chat list to ensure the chat appears with the correct title
        loadChats();

        // Refresh the current chat to ensure we have the latest messages from the database
        if (currentChat) {
          console.log('Refreshing current chat from database');
          const refreshChat = async () => {
            try {
              const { chat: refreshedChat } = await getChat(currentChat.id);
              if (refreshedChat) {
                setCurrentChat(refreshedChat);
              }
            } catch (err) {
              console.error('Error refreshing chat:', err);
            }
          };
          refreshChat();
        }
      }
    } catch (e) {
      console.error('Error processing event data:', e);
      console.error('Raw event data:', event.data);
      setError('Error processing streaming response');
      closeEventSource();
      setIsStreaming(false);
    }
  };

  // Handle EventSource errors
  const handleEventError = (error: Event) => {
    console.error('EventSource error:', error);

    // Get the current state of the EventSource
    const eventSource = eventSourceRef.current;
    if (!eventSource) return;

    console.error('EventSource readyState:', eventSource.readyState);

    let errorMessage = 'Connection error during streaming.';

    // Simple error handling without complex retry logic
    if (eventSource.readyState === EventSource.CLOSED) {
      errorMessage = 'Connection closed unexpectedly. Please try again.';
    }

    setError(errorMessage);
    closeEventSource();
    setIsStreaming(false);
  };

  const handleSendMessage = async () => {
    const trimmedMessage = message.trim();
    if (!trimmedMessage || isStreaming) return;

    // Create new chat if none exists
    let chat = currentChat;
    if (!chat) {
      const title = trimmedMessage.slice(0, 30) + (trimmedMessage.length > 30 ? '...' : '');
      const { chat: newChat, error: createError } = await createChat(title);
      if (!newChat) {
        setError(`Failed to create chat: ${createError}`);
        return;
      }
      chat = newChat;
      setCurrentChat(chat);

      // Ensure the URL is updated with the new chat ID
      router.push(`/chat?id=${chat.id}`, undefined, { shallow: true });

      // Wait a moment for the router to update
      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    // Add user message to UI immediately
    const userMessage: Message = {
      id: Date.now(),
      chat_id: typeof chat.id === 'string' ? parseInt(chat.id, 10) : chat.id,
      role: 'user' as const,
      content: trimmedMessage,
      created_at: new Date().toISOString(),
    };

    setCurrentChat((prev) => ({
      ...(prev || chat),
      messages: (prev || chat).messages ? [...(prev || chat).messages, userMessage] : [userMessage],
    }));

    // Clear input and set states
    setMessage('');
    setIsStreaming(true);
    setError(null);

    // Create placeholder for assistant response
    const assistantMessage: Message = {
      id: Date.now() + 1,
      chat_id: typeof chat.id === 'string' ? parseInt(chat.id, 10) : chat.id,
      role: 'assistant' as const,
      content: '',
      created_at: new Date().toISOString(),
    };

    setCurrentChat((prev) => ({
      ...(prev || chat),
      messages: (prev || chat).messages
        ? [...(prev || chat).messages, assistantMessage]
        : [assistantMessage],
    }));

    // Force scroll to bottom immediately after adding the assistant message placeholder
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({
        behavior: 'auto',
        block: 'end',
      });
    }, 10);

    try {
      // Refresh chat list to show new chat with correct title
      await loadChats();

      // Ensure we have the latest chat state
      if (!currentChat || currentChat.id !== chat.id) {
        const { chat: refreshedChat } = await getChat(chat.id);
        if (refreshedChat) {
          // Keep our newly added messages
          refreshedChat.messages = refreshedChat.messages || [];
          if (refreshedChat.messages.length === 0) {
            refreshedChat.messages = [userMessage, assistantMessage];
          }
          setCurrentChat(refreshedChat);
        }
      }

      // Set up EventSource
      const eventSource = setupEventSource(chat.id, trimmedMessage);

      // Set up event handlers
      eventSource.onmessage = handleEventMessage;
      eventSource.onerror = handleEventError;
    } catch (err) {
      console.error('Error sending message:', err);
      setError('An unexpected error occurred while sending your message.');
      setIsStreaming(false);
    }
  };

  if (authLoading) {
    return (
      <Layout title="Chat - Doogie Chat Bot">
        <div className="flex justify-center items-center h-64">
          <p>Loading...</p>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Chat - Doogie Chat Bot">
      <div className="flex h-[calc(100vh-12rem)]">
        {/* Chat Sidebar */}
        <div className="w-64 bg-white dark:bg-gray-800 rounded-lg shadow-md mr-4 overflow-y-auto flex-shrink-0">
          <div className="p-4">
            <Button onClick={handleNewChat} className="w-full mb-4">
              New Chat
            </Button>
            <div className="space-y-2">
              {chats && chats.length > 0
                ? chats.map((chat) => (
                    <div
                      key={chat.id}
                      className={`flex items-center justify-between p-2 rounded-md ${
                        currentChat?.id === chat.id
                          ? 'bg-primary-100 dark:bg-primary-900 text-primary-900 dark:text-primary-100'
                          : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                      }`}
                    >
                      <div
                        onClick={() => handleSelectChat(chat)}
                        className="flex-grow cursor-pointer truncate"
                      >
                        <p className="truncate text-sm">{chat.title}</p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm('Are you sure you want to delete this chat?')) {
                            handleDeleteChat(chat.id);
                          }
                        }}
                        className="ml-2 p-1 text-gray-500 hover:text-red-500 dark:text-gray-400 dark:hover:text-red-400"
                        title="Delete chat"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-4 w-4"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          />
                        </svg>
                      </button>
                    </div>
                  ))
                : null}
              {isLoading && <p className="text-sm text-gray-500">Loading chats...</p>}
              {!isLoading && (!chats || chats.length === 0) && (
                <p className="text-sm text-gray-500">No chats yet. Start a new conversation!</p>
              )}
            </div>
          </div>
        </div>

        {/* Chat Main Area */}
        <div className="flex-1 flex flex-col bg-white dark:bg-gray-800 rounded-lg shadow-md">
          <div className="flex-1 flex flex-col min-h-0">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 scroll-smooth min-h-0">
              {currentChat && currentChat.messages && currentChat.messages.length > 0 ? (
                currentChat.messages.map((msg, index) => (
                  <div
                    key={msg.id || index}
                    className={`chat-message mb-12 relative ${
                      msg.role === 'user' ? 'user-message' : 'assistant-message'
                    }`}
                  >
                    <div className="font-semibold mb-2">
                      {msg.role === 'user' ? 'You' : 'Doogie'}
                    </div>
                    <MessageContent content={msg.content} message={msg} />
                    {/* Feedback UI - positioned outside the chat bubble border */}
                    {msg.role === 'assistant' && !msg.feedback && (
                      <div className="absolute right-0 -bottom-6 flex flex-row gap-1 bg-white dark:bg-gray-800 px-2 py-1 rounded shadow-sm">
                        <button
                          onClick={() => handleFeedback(String(msg.id), 'positive')}
                          className="p-1 text-gray-400 hover:text-green-500 transition-colors duration-200"
                          title="Helpful"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                          >
                            <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
                          </svg>
                        </button>
                        <button
                          onClick={() => handleFeedback(String(msg.id), 'negative')}
                          className="p-1 text-gray-400 hover:text-red-500 transition-colors duration-200"
                          title="Not helpful"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                          >
                            <path d="M18 9.5a1.5 1.5 0 11-3 0v-6a1.5 1.5 0 013 0v6zM14 9.667v-5.43a2 2 0 00-1.105-1.79l-.05-.025A4 4 0 0011.055 2H5.64a2 2 0 00-1.962 1.608l-1.2 6A2 2 0 004.44 12H8v4a2 2 0 002 2 1 1 0 001-1v-.667a4 4 0 01.8-2.4l1.4-1.866a4 4 0 00.8-2.4z" />
                          </svg>
                        </button>
                        {/* Info button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            const infoPanel = document.getElementById(`info-panel-${msg.id}`);
                            if (infoPanel) {
                              infoPanel.classList.toggle('hidden');
                            }
                          }}
                          className="p-1 text-gray-400 hover:text-blue-500 transition-colors duration-200"
                          title="View message info"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                          >
                            <path
                              fillRule="evenodd"
                              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9a1 1 0 00-1-1z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </button>
                      </div>
                    )}
                    {/* Feedback status */}
                    {msg.role === 'assistant' && msg.feedback && (
                      <div className="absolute right-0 -bottom-6 flex items-center gap-1 bg-white dark:bg-gray-800 px-2 py-1 rounded shadow-sm">
                        <span
                          className={`text-sm ${msg.feedback === 'positive' ? 'text-green-500' : 'text-red-500'}`}
                        >
                          {msg.feedback === 'positive' ? '👍' : '👎'}
                        </span>
                        {msg.feedback_text && (
                          <span className="text-xs text-gray-400" title={msg.feedback_text}>
                            💬
                          </span>
                        )}
                        {/* Info button for messages with feedback */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            const infoPanel = document.getElementById(`info-panel-${msg.id}`);
                            if (infoPanel) {
                              infoPanel.classList.toggle('hidden');
                            }
                          }}
                          className="p-1 text-gray-400 hover:text-blue-500 transition-colors duration-200"
                          title="View message info"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                          >
                            <path
                              fillRule="evenodd"
                              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9a1 1 0 00-1-1z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </button>
                      </div>
                    )}

                    {/* Info Panel - Hidden by default */}
                    {msg.role === 'assistant' && (
                      <div
                        id={`info-panel-${msg.id}`}
                        className="hidden mt-2 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-md border border-gray-200 dark:border-gray-700 text-xs"
                      >
                        <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">
                          Message Information
                        </h4>

                        {/* Performance Stats */}
                        <div className="mb-3">
                          <h5 className="font-medium text-gray-600 dark:text-gray-400 mb-1">
                            Performance
                          </h5>
                          <div className="grid grid-cols-2 gap-2">
                            {msg.tokens && (
                              <div>
                                <span className="text-gray-500 dark:text-gray-500">Tokens:</span>{' '}
                                {msg.tokens}
                              </div>
                            )}
                            {msg.tokens_per_second && (
                              <div>
                                <span className="text-gray-500 dark:text-gray-500">
                                  Tokens/sec:
                                </span>{' '}
                                {msg.tokens_per_second.toFixed(2)}
                              </div>
                            )}
                            {msg.model && (
                              <div>
                                <span className="text-gray-500 dark:text-gray-500">Model:</span>{' '}
                                {msg.model}
                              </div>
                            )}
                            {msg.provider && (
                              <div>
                                <span className="text-gray-500 dark:text-gray-500">Provider:</span>{' '}
                                {msg.provider}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Document References */}
                        {msg.context_documents && msg.context_documents.length > 0 && (
                          <div>
                            <h5 className="font-medium text-gray-600 dark:text-gray-400 mb-1">
                              Referenced Documents
                            </h5>
                            <DocumentReferences documentIds={msg.context_documents} />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 dark:text-gray-400 my-8">
                  Start typing your message to begin a new conversation!
                </div>
              )}
              {isStreaming && (
                <div className="flex justify-center my-2">
                  <div className="animate-pulse text-gray-500 dark:text-gray-400">
                    Doogie is thinking...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area - Always visible */}
            <div className="p-4 border-t border-gray-200 dark:border-gray-700">
              {error && (
                <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-400 rounded-md">
                  {error}
                </div>
              )}
              <div className="flex">
                <Input
                  type="text"
                  placeholder="Type your message..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  className="flex-1 mr-2"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!message.trim() || isStreaming}
                  isLoading={isStreaming}
                >
                  Send
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
