import React, { useState, useMemo, useRef, useEffect } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { Message } from '@/types';
import { parseThinkTags } from '@/utils/thinkTagParser';
import Tooltip from '@/components/ui/CustomTooltip';
import { useNotification } from '@/contexts/NotificationContext';
import DocumentReferences from '@/components/chat/DocumentReferences';
import { submitFeedback } from '@/services/chat'; // Import the feedback service

import { FeedbackType } from '@/components/chat/FeedbackButton'; // Assuming FeedbackType is defined here

interface MessageContentProps {
  content: string;
  message: Message;
  // REMOVED handleFeedback prop
}

// Define the expected structure for code blocks to help with TypeScript typing
interface CodeProps {
  children?: React.ReactNode;
  className?: string;
  inline?: boolean;
  [key: string]: any;
}

const MessageContent: React.FC<MessageContentProps> = ({ content, message }) => { // REMOVED handleFeedback from destructuring
  const [collapsedThinkTags, setCollapsedThinkTags] = useState<{[key: number]: boolean}>({});
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showActionsOnMobile, setShowActionsOnMobile] = useState(false);
  const [isInfoExpanded, setIsInfoExpanded] = useState(false);
  const { showNotification } = useNotification();
  const messageRef = useRef<HTMLDivElement>(null);
  // REMOVED local feedback state - will rely on message prop
  
  // Parse think tags using memoization to avoid unnecessary re-parsing
  const parts = useMemo(() => parseThinkTags(content), [content]);

  // REMOVED local handleFeedback function

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

  // Copy content to clipboard
  const copyToClipboard = async (text: string, index: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      showNotification('Copied to clipboard!', 'success');
      
      // Reset copy state after 2 seconds
      setTimeout(() => {
        setCopiedIndex(null);
      }, 2000);
      
      return true;
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      showNotification('Failed to copy to clipboard', 'error');
      return false;
    }
  };

  // Copy the entire message content
  const copyMessageContent = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering the message click
    copyToClipboard(content, -1);
  };

  // Format the creation date for tooltip display
  const formattedDate = message.created_at 
    ? new Date(message.created_at).toLocaleString()
    : 'Unknown time';

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
    code: (props: CodeProps) => {
      const { className, children, inline, ...rest } = props;
      const match = /language-(\w+)/.exec(className || '');
      const codeContent = children ? String(children).replace(/\n$/, '') : '';
      
      // Check if it's a code block with language
      if (!inline && match) {
        return (
          <div className="relative group">
            <SyntaxHighlighter
              style={dracula as any}
              language={match[1]}
              PreTag="pre"
              {...rest}
            >
              {codeContent}
            </SyntaxHighlighter>
            <button
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering the message click
                copyToClipboard(codeContent, 999 + Math.random());
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
          <code className={className} {...rest}>
            {children}
          </code>
        );
      }
    }
  };

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

        {/* Message Action Buttons - Now at the bottom of the message */}
        <div className="flex justify-end space-x-2 mt-2">
          {/* Thumbs up/down buttons - Only show for assistant messages */}
          {message.role === 'assistant' && (
            <div className="flex space-x-1">
              <button
                className={`p-1.5 rounded-full shadow-sm border border-gray-200 dark:border-gray-700 hover:bg-green-100 dark:hover:bg-green-900 ${
                  message.feedback === 'up' // Use message.feedback prop
                    ? 'bg-green-500 text-white' // Highlighted state (hover is still applied from base)
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300' // Default state (hover is applied from base)
                }`}
                aria-label="Thumbs up"
                title="Thumbs up"
                // REMOVED onClick handler as handleFeedback is not in scope here
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                </svg>
              </button>
              <button
                className={`p-1.5 rounded-full shadow-sm border border-gray-200 dark:border-gray-700 hover:bg-red-100 dark:hover:bg-red-900 ${
                  message.feedback === 'down' // Use message.feedback prop
                    ? 'bg-red-500 text-white' // Highlighted state
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300' // Default state
                }`}
                aria-label="Thumbs down"
                title="Thumbs down"
                // REMOVED onClick handler as handleFeedback is not in scope here
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

export default MessageContent;