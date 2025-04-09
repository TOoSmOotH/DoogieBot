import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';
import { Chat } from '@/types';
import { getChats, createChat, updateChatTags, deleteChat, getChat } from '@/services/chat';
import { useAuth } from '@/contexts/AuthContext';
import { useNotification } from '@/contexts/NotificationContext';
import { announce } from '@/utils/accessibilityUtils';

export interface UseChatListReturn {
  chats: Chat[];
  filteredChats: Chat[];
  searchTerm: string;
  selectedFilterTags: string[];
  isLoading: boolean;
  error: string | null;
  loadChats: () => Promise<void>;
  setSearchTerm: (term: string) => void;
  setSelectedFilterTags: (tags: string[]) => void;
  handleNewChat: () => Promise<void>;
  handleSelectChat: (chat: Chat) => void;
  handleUpdateTags: (chatId: string, tags: string[]) => Promise<void>;
  handleDeleteChatClick: (chatId: string, e?: React.MouseEvent) => void;
  setChats: React.Dispatch<React.SetStateAction<Chat[]>>; // Expose setChats for optimistic updates
  currentChatId: string | null; // Needed for delete logic
  setCurrentChat: React.Dispatch<React.SetStateAction<Chat | null>>; // Needed for delete logic
  setChatToDelete: React.Dispatch<React.SetStateAction<string | null>>; // Expose for dialog
  setDeleteDialogOpen: React.Dispatch<React.SetStateAction<boolean>>; // Expose for dialog
}

export const useChatList = (
  initialChats: Chat[] = [],
  currentChat: Chat | null,
  setCurrentChat: React.Dispatch<React.SetStateAction<Chat | null>>,
  setChatToDelete: React.Dispatch<React.SetStateAction<string | null>>,
  setDeleteDialogOpen: React.Dispatch<React.SetStateAction<boolean>>
): UseChatListReturn => {
  const { isAuthenticated } = useAuth();
  const { showNotification } = useNotification();
  const router = useRouter();
  const currentChatId = router.query.id ? String(router.query.id) : null;

  const [chats, setChats] = useState<Chat[]>(initialChats);
  const [filteredChats, setFilteredChats] = useState<Chat[]>(initialChats);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFilterTags, setSelectedFilterTags] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load chats function
  const loadChats = useCallback(async () => {
    if (!isAuthenticated) return;
    setIsLoading(true);
    setError(null);
    try {
      const { chats: fetchedChats, error: fetchError } = await getChats();
      if (fetchedChats) {
        setChats(fetchedChats);
        setFilteredChats(fetchedChats); // Initialize filtered chats as well
      } else if (fetchError) {
        setError(`Failed to load chats: ${fetchError}`);
        setChats([]); // Ensure empty array on error
        setFilteredChats([]);
      } else {
        setChats([]);
        setFilteredChats([]);
      }
    } catch (err) {
      console.error('Error loading chats:', err);
      setError('An unexpected error occurred while loading chats.');
      setChats([]);
      setFilteredChats([]);
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  // Load chats on mount and when auth changes
  useEffect(() => {
    loadChats();
  }, [isAuthenticated, loadChats]);

  // Filter chats when search term, tags, or chats change
  useEffect(() => {
    let filtered = [...chats];

    // Filter by search term if present
    if (searchTerm.trim()) {
      const lowerCaseSearchTerm = searchTerm.toLowerCase();
      filtered = filtered.filter(chat =>
        chat.title.toLowerCase().includes(lowerCaseSearchTerm)
      );
    }

    // Filter by selected tags if any
    if (selectedFilterTags.length > 0) {
      filtered = filtered.filter(chat =>
        chat.tags && selectedFilterTags.some(tag => chat.tags?.includes(tag))
      );
    }

    setFilteredChats(filtered);
  }, [searchTerm, selectedFilterTags, chats]);

  const handleNewChat = async () => {
    // Check if there's already a "New Conversation" chat that hasn't been used yet
    const existingNewChat = chats.find(chat =>
      chat.title === 'New Conversation' &&
      (!chat.messages || chat.messages.length === 0)
    );

    if (existingNewChat) {
      // Use the existing empty chat instead of creating a new one
      console.log('Using existing empty chat:', existingNewChat);
      setCurrentChat(null);
      router.push(`/chat?id=${existingNewChat.id}`, undefined, { shallow: true });
      
      // Announce for screen readers
      announce({
        message: 'New chat selected',
        politeness: 'polite'
      });
      
      return;
    }

    // Clear relevant state handled by other hooks/component
    setCurrentChat(null);
    setError(null);
    // setIsStreaming(false); // This will be handled by useChatMessages

    try {
      console.log('Creating new chat...');
      const { chat: newChat, error: createError } = await createChat('New Conversation');

      if (newChat) {
        console.log('New chat created successfully:', newChat);
        // Force a re-render with the new chat
        router.push(`/chat?id=${newChat.id}`, undefined, { shallow: true });
        // Refresh the chat list to include the new chat
        await loadChats();

        // Announce success for screen readers
        announce({
          message: 'New chat created successfully',
          politeness: 'polite'
        });
      } else {
        setError(createError || 'Failed to create new chat');
        showNotification(createError || 'Failed to create new chat', 'error');
        // Still clear current chat to show empty state
        router.push('/chat', undefined, { shallow: true });
      }
    } catch (err) {
      console.error('Error creating new chat:', err);
      setError('An unexpected error occurred while creating a new chat');
      // Still clear current chat to show empty state
      router.push('/chat', undefined, { shallow: true });
    }
  };

  const handleSelectChat = (chat: Chat) => {
    router.push(`/chat?id=${chat.id}`, undefined, { shallow: true });
    // Sidebar closing logic might move to useChatUI or stay in component
    // if (window.innerWidth < 768) {
    //   setIsSidebarOpen(false);
    // }

    // Announce for screen readers
    announce({
      message: `Selected chat: ${chat.title}`,
      politeness: 'polite'
    });
  };

  // Handle updating chat tags
  const handleUpdateTags = async (chatId: string, tags: string[]) => {
    console.log('Updating tags for chat:', chatId, 'New tags:', tags);
    setError(null);

    // Optimistically update the UI immediately
    const originalChats = chats;
    const originalCurrentChat = currentChat;

    setChats(prevChats =>
      prevChats.map(chat =>
        chat.id === chatId ? { ...chat, tags } : chat
      )
    );

    if (currentChat?.id === chatId) {
      setCurrentChat(prev => prev ? { ...prev, tags } : null);
    }

    try {
      const { success, error: updateError } = await updateChatTags(chatId, tags);

      if (success) {
        console.log('Tags updated successfully in the backend');
        // Optionally refresh the specific chat or the whole list for consistency
        await loadChats(); // Refresh the whole list for simplicity here

        // Ensure current chat state is updated if it was the one tagged
        if (currentChat?.id === chatId) {
             try {
               const { chat: refreshedChat } = await getChat(chatId);
               if (refreshedChat) {
                 setCurrentChat((prev) => {
                   if (!prev) return refreshedChat;
                   // Merge to preserve potentially unsaved message state
                   return { ...prev, ...refreshedChat, messages: prev.messages };
                 });
               }
             } catch (refreshError) {
               console.error('Error refreshing current chat after tag update:', refreshError);
             }
        }

        showNotification('Tags updated successfully', 'success');
      } else {
        console.error('Failed to update tags:', updateError);
        setError(`Failed to update tags: ${updateError}`);
        showNotification(`Failed to update tags: ${updateError}`, 'error');
        // Rollback optimistic update
        setChats(originalChats);
        if (originalCurrentChat?.id === chatId) {
            setCurrentChat(originalCurrentChat);
        }
      }
    } catch (err) {
      console.error('Error updating tags:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Failed to update tags: ${errorMessage}`);
      showNotification(`Failed to update tags: ${errorMessage}`, 'error');
      // Rollback optimistic update
      setChats(originalChats);
      if (originalCurrentChat?.id === chatId) {
          setCurrentChat(originalCurrentChat);
      }
    }
  };

  // Handle opening the delete confirmation dialog
  const handleDeleteChatClick = (chatId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setChatToDelete(chatId);
    setDeleteDialogOpen(true);
  };

  // Note: The actual handleDeleteChat function remains in the main component
  // because it needs access to setChatToDelete and setDeleteDialogOpen,
  // which are passed down to the ConfirmDialog. We expose the necessary setters.

  return {
    chats,
    filteredChats,
    searchTerm,
    selectedFilterTags,
    isLoading,
    error,
    loadChats,
    setSearchTerm,
    setSelectedFilterTags,
    handleNewChat,
    handleSelectChat,
    handleUpdateTags,
    handleDeleteChatClick,
    setChats, // Expose setChats
    currentChatId,
    setCurrentChat, // Pass down setCurrentChat
    setChatToDelete, // Expose setter
    setDeleteDialogOpen, // Expose setter
  };
};