import React, { useState, useMemo, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import ToolCallDisplay from './ToolCallDisplay';
// Removed duplicate Button import

// Define a simple custom style to replace dracula
const customDraculaStyle = {
  'code[class*="language-"]': {
    color: '#f8f8f2',
    background: 'none',
    textShadow: '0 1px rgba(0, 0, 0, 0.3)',
    fontFamily: "Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    lineHeight: '1.5',
    tabSize: 4,
    hyphens: 'none',
  },
  'pre[class*="language-"]': {
    color: '#f8f8f2',
    background: '#282a36',
    textShadow: '0 1px rgba(0, 0, 0, 0.3)',
    fontFamily: "Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    lineHeight: '1.5',
    tabSize: 4,
    hyphens: 'none',
    padding: '1em',
    margin: '.5em 0',
    overflow: 'auto',
    borderRadius: '0.3em',
  },
  ':not(pre) > code[class*="language-"]': {
    background: '#282a36',
    padding: '.1em',
    borderRadius: '.3em',
    whiteSpace: 'normal',
  },
  comment: { color: '#6272a4' },
  prolog: { color: '#6272a4' },
  doctype: { color: '#6272a4' },
  cdata: { color: '#6272a4' },
  punctuation: { color: '#f8f8f2' },
  property: { color: '#ff79c6' },
  tag: { color: '#ff79c6' },
  constant: { color: '#ff79c6' },
  symbol: { color: '#ff79c6' },
  deleted: { color: '#ff79c6' },
  boolean: { color: '#bd93f9' },
  number: { color: '#bd93f9' },
  selector: { color: '#50fa7b' },
  'attr-name': { color: '#50fa7b' },
  string: { color: '#f1fa8c' },
  char: { color: '#f1fa8c' },
  builtin: { color: '#50fa7b' },
  inserted: { color: '#50fa7b' },
  operator: { color: '#ff79c6' },
  entity: { color: '#f8f8f2', cursor: 'help' },
  url: { color: '#f8f8f2' },
  variable: { color: '#f8f8f2' },
  atrule: { color: '#ff79c6' },
  'attr-value': { color: '#f1fa8c' },
  function: { color: '#50fa7b' },
  'class-name': { color: '#8be9fd' },
  keyword: { color: '#ff79c6' },
  regex: { color: '#f1fa8c' },
  important: { color: '#f1fa8c', fontWeight: 'bold' },
  bold: { fontWeight: 'bold' },
  italic: { fontStyle: 'italic' },
};
import { Message } from '@/types';
import { parseThinkTags } from '@/utils/thinkTagParser';
import { useCurrentChat } from '@/hooks/useCurrentChat';
import { FeedbackType } from '@/components/chat/FeedbackButton'; // Import FeedbackType
import Tooltip from '@/components/ui/CustomTooltip';
import { useNotification } from '@/contexts/NotificationContext';
import DocumentReferences from '@/components/chat/DocumentReferences';
import MarkdownEditor from '@/components/chat/MarkdownEditor';
import { Button } from '@/components/ui/Button';

interface MessageContentProps {
  content: string;
  message: Message;
  onUpdateMessage?: (messageId: string, newContent: string) => Promise<boolean>;
  onFeedback?: (messageId: string, feedbackType: FeedbackType, comment?: string) => Promise<void>; // Add onFeedback prop
  isWaitingForResponse?: boolean; // Add prop for loading state
}

const ImprovedMessageContent: React.FC<MessageContentProps> = ({
  content,
  message,
  onUpdateMessage,
  onFeedback, // Destructure onFeedback
  isWaitingForResponse // Destructure new prop
}) => {
  const [collapsedThinkTags, setCollapsedThinkTags] = useState<{[key: number]: boolean}>({});
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showActionsOnMobile, setShowActionsOnMobile] = useState(false);
  const [isInfoExpanded, setIsInfoExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const { showNotification } = useNotification();
  const [showNegativeFeedbackInput, setShowNegativeFeedbackInput] = useState(false);
  const [negativeFeedbackComment, setNegativeFeedbackComment] = useState('');
  const messageRef = useRef<HTMLDivElement>(null);
  const [toolErrors, setToolErrors] = useState<{[key: string]: any}>({});
  const [retryingTools, setRetryingTools] = useState<{[key: string]: boolean}>({});
  const currentChatHook = useCurrentChat(
    null,
    () => {} // Dummy function for setChats parameter
  );

  // Handle retrying a tool call
  const handleRetryToolCall = async (toolCall: any) => {
    if (!currentChatHook || !currentChatHook.currentChat || !toolCall.id) return;
    
    try {
      setRetryingTools(prev => ({ ...prev, [toolCall.id]: true }));
      
      // Call the API to retry the tool call
      const response = await fetch(`/api/v1/chats/${currentChatHook.currentChat.id}/retry-tool`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tool_call_id: toolCall.id,
          function_name: toolCall.function?.name || toolCall.name,
          arguments: toolCall.function?.arguments || toolCall.arguments,
        }),
      });
      
      if (!response.ok) {
        // Try to parse the error as JSON
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          // If it's not JSON, get the text
          const errorText = await response.text();
          errorData = { message: errorText || 'Failed to retry tool call' };
        }
        
        // Add status code to the error
        errorData.status = response.status;
        
        throw errorData;
      }
      
      // Refresh the chat to get the updated tool result
      if (currentChatHook.loadSpecificChat && currentChatHook.currentChat.id) {
        await currentChatHook.loadSpecificChat(currentChatHook.currentChat.id);
      }
      
      showNotification('Tool call retried successfully', 'success');
    } catch (error) {
      console.error('Error retrying tool call:', error);
      showNotification('Failed to retry tool call', 'error');
    } finally {
      setRetryingTools(prev => {
        const newRetrying = { ...prev };
        delete newRetrying[toolCall.id];
        return newRetrying;
      });
    }
  };

  // Parse think tags using memoization to avoid unnecessary re-parsing
  const parts = useMemo(() => parseThinkTags(content), [content]);

  // Toggle think tag collapse state
  const toggleThinkTag = (index: number) => {
    setCollapsedThinkTags(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };
  
  // Toggle action buttons on mobile
  const toggleActionsOnMobile = () => {
    setShowActionsOnMobile(!showActionsOnMobile);
  };
  
  // Handle clicking outside to close mobile actions
  useEffect(() => {
    if (!showActionsOnMobile) return;
    
    const handleClickOutside = (event: MouseEvent) => {
      if (messageRef.current && !messageRef.current.contains(event.target as Node)) {
        setShowActionsOnMobile(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showActionsOnMobile]);

  
    // Copy content to clipboard with fallback for insecure contexts
    const copyToClipboard = async (text: string, index: number) => {
      let success = false;
      try {
        // Try modern clipboard API first (requires secure context)
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
          success = true;
        } else {
          // Fallback for insecure contexts or older browsers
          const textArea = document.createElement('textarea');
          textArea.value = text;
          textArea.style.position = 'fixed'; // Prevent scrolling to bottom
          textArea.style.left = '-9999px';
          document.body.appendChild(textArea);
          textArea.focus();
          textArea.select();
          try {
            success = document.execCommand('copy');
          } catch (err) {
            console.error('Fallback copy failed:', err);
            success = false;
          }
          document.body.removeChild(textArea);
        }
  
        if (success) {
          setCopiedIndex(index);
          showNotification('Copied to clipboard!', 'success');
          // Reset copy state after 2 seconds
          setTimeout(() => {
            setCopiedIndex(null);
          }, 2000);
        } else {
          throw new Error('Copy command was not successful.');
        }
        
        return success;
      } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showNotification('Failed to copy to clipboard', 'error');
        return false;
      }
    }; // End of copyToClipboard
  
    // Copy the entire message content
    const copyMessageContent = (e: React.MouseEvent) => { // Restore function definition and parameter
      e.stopPropagation(); // Prevent triggering the message click
      copyToClipboard(content, -1);
    }; // End of copyMessageContent
  
  // Format the creation date for tooltip display
  const formattedDate = message.created_at 
    ? new Date(message.created_at).toLocaleString()
    : 'Unknown time';

  // Handle saving edited content
  const handleSaveContent = async (newContent: string) => {
    if (!onUpdateMessage) {
      setIsEditing(false);
      return;
    }

    setIsSaving(true);
    
    try {
      const success = await onUpdateMessage(String(message.id), newContent);
      
      if (success) {
        showNotification('Message updated successfully', 'success');
        setIsEditing(false);
      } else {
        showNotification('Failed to update message', 'error');
      }
    } catch (error) {
      console.error('Error updating message:', error);
      showNotification('An error occurred while updating the message', 'error');
    } finally {
      setIsSaving(false);
    }
  };

  // Handle toggling the negative feedback input
  const handleNegativeFeedbackClick = () => {
    setShowNegativeFeedbackInput(!showNegativeFeedbackInput);
    setNegativeFeedbackComment(''); // Clear comment on toggle
  };

  // Handle submitting the negative feedback comment
  const submitNegativeFeedback = async () => {
    if (onFeedback) {
      try {
        await onFeedback(String(message.id), 'negative', negativeFeedbackComment);
        showNotification('Feedback submitted', 'success');
      } catch (error) {
        showNotification('Failed to submit feedback', 'error');
      }
    }
    setShowNegativeFeedbackInput(false);
    setNegativeFeedbackComment('');
  };

  // Info tooltip content
  const infoTooltipContent = (
    <div className="text-xs p-2 max-w-xs">
      <div className="mb-1"><span className="font-medium">Model:</span> {message.model || 'Not specified'}</div>
      <div className="mb-1"><span className="font-medium">Provider:</span> {message.provider || 'Not specified'}</div>
      <div className="mb-1"><span className="font-medium">Tokens:</span> {message.tokens !== undefined ? message.tokens : 'Not tracked'}</div>
      <div className="mb-1"><span className="font-medium">Speed:</span> {message.tokens_per_second !== undefined ? `${message.tokens_per_second} tokens/s` : 'Not tracked'}</div>
      <div className="mb-1"><span className="font-medium">Time:</span> {formattedDate}</div>
      
      {/* Display document references if available, or indicate no RAG data was used */}
      <div className="mt-2 border-t border-gray-200 dark:border-gray-700 pt-2">
        <div className="font-medium mb-1">Sources:</div>
        {message.context_documents && message.context_documents.length > 0 ? (
          <DocumentReferences documentIds={message.context_documents} />
        ) : (
          <div className="text-gray-500 italic">No RAG data was used for this response</div>
        )}
      </div>
      
      {/* Add debugging info in development */}
      {process.env.NODE_ENV === 'development' && (
        <div className="mt-2 border-t border-gray-200 dark:border-gray-700 pt-2">
          <details>
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">Debug Info</summary>
            <pre className="mt-1 text-xs bg-gray-100 dark:bg-gray-800 p-1 rounded overflow-auto max-h-40">
              {JSON.stringify(message, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );

  // Custom components for ReactMarkdown
  const MarkdownComponents: Components = {
    code: (props: any) => {
      const { className, children, inline } = props;
      const match = /language-(\w+)/.exec(className || '');
      const codeContent = children ? String(children).replace(/\n$/, '') : '';
      
      // Check if it's a code block with language
      if (!inline && match) {
        return (
          <div className="relative group">
            <SyntaxHighlighter
              style={customDraculaStyle as any}
              language={match[1]}
              PreTag="pre"
            >
              {codeContent}
            </SyntaxHighlighter>
            <button
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering the message click
                copyToClipboard(codeContent, message.id + codeContent.length);
              }}
              className="absolute top-2 right-2 bg-gray-700 dark:bg-gray-600 text-white p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
              aria-label="Copy code to clipboard"
              title="Copy code to clipboard"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
              </svg>
            </button>
          </div>
        );
      } else {
        return (
          <code className={className}>
            {children}
          </code>
        );
      }
    }
  };

  // If in editing mode, show the Markdown Editor
  if (isEditing) {
    return (
      <div className="mt-2">
        <MarkdownEditor
          initialValue={content}
          onSave={handleSaveContent}
          onCancel={() => setIsEditing(false)}
        />
        {isSaving && (
          <div className="mt-2 text-sm text-gray-500 dark:text-gray-400 flex items-center">
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-primary-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Saving changes...
          </div>
        )}
      </div>
    );
  }

  // Show loading indicator if waiting for the first chunk of an assistant message
  if (isWaitingForResponse && message.role === 'assistant' && !content) {
    return (
      <div className="flex items-center text-gray-500 dark:text-gray-400 italic">
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-primary-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Thinking...
      </div>
    );
  }

  return (
    <div
      ref={messageRef}
      className="relative group/message w-full" 
      onClick={(e) => {
        // Only toggle on smaller screens and prevent toggling when clicking action buttons
        if (window.innerWidth < 768 && e.target === e.currentTarget) {
          toggleActionsOnMobile();
        }
      }}
    >
      <div className="w-full">
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
                    <div className={message.role === 'user' ? 'user-message-content' : 'assistant-message-content'}>
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                        components={MarkdownComponents}
                        className={message.role === 'user' ? 'user-message-markdown' : 'assistant-message-markdown'}
                      >
                        {part.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div key={index} className={message.role === 'user' ? 'user-message-content' : 'assistant-message-content'}>
                {/* Display tool calls if present */}
                {message.role === 'assistant' && message.tool_calls && message.tool_calls.length > 0 && (
                  <div className="tool-calls-container mb-4">
                    {message.tool_calls.map((toolCall, idx) => {
                      // Find the corresponding tool result message
                      const toolResult = currentChatHook?.currentChat?.messages?.find(
                        msg => msg.role === 'tool' && msg.tool_call_id === toolCall.id
                      );
                      
                      // Add console log to debug
                      console.log('Tool call:', toolCall);
                      console.log('Tool result:', toolResult);
                      
                      return (
                        <ToolCallDisplay
                          key={`tool-call-${idx}`}
                          message={message}
                          toolCall={toolCall}
                          toolResult={toolResult}
                          isLoading={isWaitingForResponse && !toolResult}
                          onRetry={handleRetryToolCall}
                          error={toolErrors[toolCall.id]}
                          isRetrying={retryingTools[toolCall.id]}
                        />
                      );
                    })}
                  </div>
                )}
                
                {/* Display regular content */}
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={MarkdownComponents}
                  className={message.role === 'user' ? 'user-message-markdown' : 'assistant-message-markdown'}
                >
                  {part.content}
                </ReactMarkdown>
              </div>
            )
          )}
        </div>

        {/* Message Action Buttons */}
        <div className="flex justify-end space-x-2 mt-2">
          {/* Edit Button - Show for both user and assistant messages if onUpdateMessage is provided */}
          {onUpdateMessage && (
            <button
              onClick={() => setIsEditing(true)}
              className="p-1.5 rounded-full shadow-sm bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
              aria-label="Edit message"
              title="Edit message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
              </svg>
            </button>
          )}

          {/* Thumbs up/down buttons - Only show for assistant messages */}
          {message.role === 'assistant' && (
            <div className="flex space-x-1">
              {/* Thumbs Up Button */}
              <button
                onClick={() => { if (onFeedback) onFeedback(String(message.id), 'positive'); }} // Call onFeedback prop
                className={`p-1.5 rounded-full shadow-sm border border-gray-200 dark:border-gray-700 hover:bg-green-100 dark:hover:bg-green-900 ${
                  message.feedback === 'positive'
                    ? 'bg-green-500 text-white' // Highlighted state
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300' // Default state
                }`}
                aria-label="Thumbs up"
                title="Thumbs up"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                </svg>
              </button>
              {/* Thumbs Down Button */}
              <button
                onClick={handleNegativeFeedbackClick} // Use the new handler
                className={`p-1.5 rounded-full shadow-sm border border-gray-200 dark:border-gray-700 hover:bg-red-100 dark:hover:bg-red-900 ${
                  message.feedback === 'negative'
                    ? 'bg-red-500 text-white' // Highlighted state
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300' // Default state
                }`}
                aria-label="Thumbs down"
                title="Thumbs down"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
                </svg>
              </button>
            </div>
          )}

          {/* Copy Button */}
          <button
            onClick={copyMessageContent}
            className={`p-1.5 rounded-full shadow-sm ${copiedIndex === -1 ? 'bg-green-500 text-white' : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700'}`}
            aria-label={copiedIndex === -1 ? "Copied!" : "Copy message to clipboard"}
            title={copiedIndex === -1 ? "Copied!" : "Copy message"}
          >
            {copiedIndex === -1 ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
              </svg>
            )}
          </button>
          
          {/* Info Button - Only show for non-user messages */}
          {message.role !== 'user' && (
            <button
              className={`p-1.5 rounded-full shadow-sm ${isInfoExpanded ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700'}`}
              aria-label="Show message information"
              title="Information"
              onClick={() => setIsInfoExpanded(!isInfoExpanded)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </button>
          )}
        </div>

        {/* Negative Feedback Input Area */}
        {showNegativeFeedbackInput && (
          <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-700 animate-fade-in">
            <label htmlFor={`feedback-comment-${message.id}`} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Why was this response unhelpful? (Optional)
            </label>
            <textarea
              id={`feedback-comment-${message.id}`}
              rows={3}
              className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-gray-200"
              value={negativeFeedbackComment}
              onChange={(e) => setNegativeFeedbackComment(e.target.value)}
              placeholder="Provide details..."
            />
            <div className="mt-2 flex justify-end space-x-2">
              <Button variant="outline" size="sm" onClick={handleNegativeFeedbackClick}>
                Cancel
              </Button>
              <Button variant="destructive" size="sm" onClick={submitNegativeFeedback}>
                Submit Feedback
              </Button>
            </div>
          </div>
        )}

        {/* Expandable Info Section */}
        {isInfoExpanded && message.role !== 'user' && (
          <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-800/50 rounded border border-gray-200 dark:border-gray-700 text-sm animate-fade-in">
            <div className="grid grid-cols-2 gap-2">
              <div><span className="font-medium">Model:</span> {message.model || 'Not specified'}</div>
              <div><span className="font-medium">Provider:</span> {message.provider || 'Not specified'}</div>
              <div><span className="font-medium">Tokens:</span> {message.tokens !== undefined ? message.tokens : 'Not tracked'}</div>
              <div><span className="font-medium">Speed:</span> {message.tokens_per_second !== undefined ? `${message.tokens_per_second} tokens/s` : 'Not tracked'}</div>
              <div className="col-span-2"><span className="font-medium">Time:</span> {formattedDate}</div>
            </div>
            
            {/* Sources section */}
            <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
              <div className="font-medium mb-1">Sources:</div>
              {message.context_documents && message.context_documents.length > 0 ? (
                <DocumentReferences documentIds={message.context_documents} />
              ) : (
                <div className="text-gray-500 italic">No RAG data was used for this response</div>
              )}
            </div>
            
            {/* Debug info in development mode */}
            {process.env.NODE_ENV === 'development' && (
              <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                <details>
                  <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">Debug Info</summary>
                  <pre className="mt-1 text-xs bg-gray-100 dark:bg-gray-800 p-1 rounded overflow-auto max-h-40">
                    {JSON.stringify(message, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImprovedMessageContent;