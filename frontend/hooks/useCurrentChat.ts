import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/router';
import { Chat } from '@/types';
import { getChat, updateChat } from '@/services/chat';
import { exportChat, ExportFormat } from '@/utils/exportUtils';
import { useAuth } from '@/contexts/AuthContext';
import { useNotification } from '@/contexts/NotificationContext';
import { announce } from '@/utils/accessibilityUtils';

export interface UseCurrentChatReturn {
  currentChat: Chat | null;
  setCurrentChat: React.Dispatch<React.SetStateAction<Chat | null>>;
  isLoading: boolean; // Loading state specific to the current chat
  error: string | null; // Error state specific to the current chat
  isEditingTitle: boolean;
  editedTitle: string;
  setEditedTitle: React.Dispatch<React.SetStateAction<string>>;
  originalTitle: string; // Keep track for dialog message
  editDialogOpen: boolean;
  setEditDialogOpen: React.Dispatch<React.SetStateAction<boolean>>;
  showExportMenu: boolean;
  setShowExportMenu: React.Dispatch<React.SetStateAction<boolean>>;
  exportMenuRef: React.RefObject<HTMLDivElement>;
  titleInputRef: React.RefObject<HTMLInputElement>;
  handleStartEditTitle: () => void;
  confirmEditTitle: () => void;
  handleSaveTitle: () => Promise<void>;
  handleUpdateTitle: (newTitle: string) => Promise<void>; // For external updates like from Layout
  handleExport: (format: ExportFormat) => void;
  loadSpecificChat: (chatId: string) => Promise<void>; // Expose loading function
}

export const useCurrentChat = (
    initialChat: Chat | null = null,
    // Pass setChats to update the list when title changes
    setChats: React.Dispatch<React.SetStateAction<Chat[]>>
): UseCurrentChatReturn => {
  const { isAuthenticated } = useAuth();
  const { showNotification } = useNotification();
  const router = useRouter();

  const [currentChat, setCurrentChat] = useState<Chat | null>(initialChat);
  const [isLoading, setIsLoading] = useState(false); // Initially false, true only when loading a specific chat
  const [error, setError] = useState<string | null>(null);

  // Title Editing State
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState('');
  const [originalTitle, setOriginalTitle] = useState('');
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const titleInputRef = useRef<HTMLInputElement>(null);

  // Export State
  const [showExportMenu, setShowExportMenu] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  const loadSpecificChat = useCallback(async (chatId: string) => {
    if (!isAuthenticated) return;
    setIsLoading(true);
    setError(null);
    try {
      console.log(`useCurrentChat: Loading chat ${chatId}`);
      const { chat, error: fetchError } = await getChat(chatId);
      if (chat) {
          // Preserve existing messages if the fetched chat doesn't have them
          // This handles potential race conditions during streaming/updates
          setCurrentChat(prevState => ({
              ...chat,
              messages: (chat.messages && chat.messages.length > 0)
                  ? chat.messages
                  : prevState?.messages || [],
          }));
          console.log(`useCurrentChat: Loaded chat ${chatId} successfully.`);
      } else if (fetchError) {
          setError(`Failed to load chat: ${fetchError}`);
          console.error(`useCurrentChat: Failed to load chat ${chatId}: ${fetchError}`);
          // Optionally clear currentChat or leave it as is? Clearing might be safer.
          // setCurrentChat(null);
      }
    } catch (err) {
      console.error(`useCurrentChat: Error loading chat ${chatId}:`, err);
      setError('An unexpected error occurred while loading the chat.');
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);


  // Load chat if ID is in URL
  useEffect(() => {
    const chatId = router.query.id ? String(router.query.id) : null;
    
    // Always load the chat when the ID changes, even if it's the same as currentChat.id
    // This ensures we get fresh data when navigating between chats
    if (chatId) {
        console.log(`useCurrentChat: URL chat ID changed to ${chatId}, loading chat`);
        loadSpecificChat(chatId);
    } else if (!chatId) {
        // If URL has no ID, clear the current chat
        console.log('useCurrentChat: No chat ID in URL, clearing current chat');
        setCurrentChat(null);
        setIsEditingTitle(false); // Reset editing state
        setError(null);
    }
  }, [router.query.id, isAuthenticated, loadSpecificChat]); // Removed currentChat?.id dependency

  // Listen for custom event to edit title from the Layout component
  useEffect(() => {
    const handleEditTitleEvent = (event: CustomEvent<{chatId: string}>) => {
      if (currentChat && currentChat.id === event.detail.chatId) {
        console.log('Received edit-chat-title event for chat ID:', event.detail.chatId);
        handleStartEditTitle();
      }
    };

    const handleEditCompletedEvent = async (event: CustomEvent<{chatId: string, newTitle: string}>) => {
      if (currentChat && currentChat.id === event.detail.chatId) {
        console.log('Received edit-chat-title-completed event for chat ID:', event.detail.chatId, 'with new title:', event.detail.newTitle);
        // Update title in backend and state
        await handleUpdateTitle(event.detail.newTitle);
      }
    };

    document.addEventListener('edit-chat-title', handleEditTitleEvent as unknown as EventListener);
    document.addEventListener('edit-chat-title-completed', handleEditCompletedEvent as unknown as EventListener);

    return () => {
      document.removeEventListener('edit-chat-title', handleEditTitleEvent as unknown as EventListener);
      document.removeEventListener('edit-chat-title-completed', handleEditCompletedEvent as unknown as EventListener);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentChat]); // Dependencies: currentChat


  // Start editing the chat title by opening the confirmation dialog
  const handleStartEditTitle = useCallback(() => {
    if (currentChat) {
      setOriginalTitle(currentChat.title);
      setEditedTitle(currentChat.title);
      setEditDialogOpen(true);
    }
  }, [currentChat]);

  // Start actual title editing after confirmation
  const confirmEditTitle = useCallback(() => {
    setEditDialogOpen(false);
    setIsEditingTitle(true);
    // Focus on the input after rendering
    setTimeout(() => {
      titleInputRef.current?.focus();
      titleInputRef.current?.select();
    }, 50);
  }, []);

  // Update title logic (used by handleSaveTitle and external events)
  const handleUpdateTitle = useCallback(async (newTitle: string) => {
    if (!currentChat || !newTitle.trim() || newTitle === currentChat.title) {
      return; // No change needed
    }

    setIsLoading(true); // Use the hook's loading state
    setError(null);

    try {
      console.log('Updating chat title for chat ID:', currentChat.id, 'with new title:', newTitle);
      const { success, error: updateError } = await updateChat(currentChat.id, { title: newTitle });

      if (success) {
        console.log('Chat title updated successfully in backend');
        // Update chat in local state
        setCurrentChat(prev => prev ? { ...prev, title: newTitle } : null);

        // Update chat in the main chats list
        setChats(prevChats =>
          prevChats.map(chat =>
            chat.id === currentChat.id ? { ...chat, title: newTitle } : chat
          )
        );

        showNotification('Chat title updated successfully', 'success');
        announce({ message: 'Chat title updated successfully', politeness: 'polite' });
      } else {
        console.error('Failed to update chat title in backend:', updateError);
        setError(`Failed to update title: ${updateError}`);
        showNotification(`Failed to update title: ${updateError}`, 'error');
        // Optionally reset editedTitle if save failed?
        // setEditedTitle(currentChat.title);
      }
    } catch (err) {
      console.error('Error updating chat title:', err);
      setError('An unexpected error occurred while updating the chat title.');
      showNotification('An unexpected error occurred while updating the chat title.', 'error');
    } finally {
      setIsLoading(false);
      // setIsEditingTitle(false); // Moved to handleSaveTitle caller
    }
  }, [currentChat, setChats, showNotification]);

  // Save the edited chat title (called from inline input usually)
  const handleSaveTitle = useCallback(async () => {
    if (!currentChat || !editedTitle.trim()) {
      setIsEditingTitle(false);
      setEditedTitle(currentChat?.title || ''); // Reset to original if invalid
      return;
    }

    if (editedTitle === currentChat.title) {
      setIsEditingTitle(false);
      return;
    }

    await handleUpdateTitle(editedTitle); // Use the common update logic
    setIsEditingTitle(false); // Ensure editing mode is turned off after attempt

  }, [currentChat, editedTitle, handleUpdateTitle]);


  // Handle export button click
  const handleExport = useCallback((format: ExportFormat) => {
    if (currentChat) {
      exportChat(currentChat, format);
      setShowExportMenu(false);
      showNotification(`Chat exported successfully as ${format}`, 'success');
      announce({ message: `Chat exported successfully as ${format}`, politeness: 'polite' });
    }
  }, [currentChat, showNotification]);

  // Handle clicking outside the export menu
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(event.target as Node)) {
        setShowExportMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);


  return {
    currentChat,
    setCurrentChat,
    isLoading,
    error,
    isEditingTitle,
    editedTitle,
    setEditedTitle,
    originalTitle,
    editDialogOpen,
    setEditDialogOpen,
    showExportMenu,
    setShowExportMenu,
    exportMenuRef,
    titleInputRef,
    handleStartEditTitle,
    confirmEditTitle,
    handleSaveTitle,
    handleUpdateTitle,
    handleExport,
    loadSpecificChat,
  };
};