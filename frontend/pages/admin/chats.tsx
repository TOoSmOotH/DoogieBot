import { useEffect, useState } from 'react';
import AdminLayout from '@/components/layout/AdminLayout';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { getFlaggedChats, deleteChat } from '@/services/chat';
import withAdmin from '@/utils/withAdmin';
import { Chat, Message } from '@/types';
import { PaginatedResponse } from '@/types';

const ChatReview = () => {
  const [chats, setChats] = useState<Chat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadChats();
  }, []);

  const loadChats = async () => {
    try {
      const { chats, error } = await getFlaggedChats({ page: 1, size: 100 });
      if (chats) {
        setChats(chats.items || []);
      } else if (error) {
        setError(error);
      }
    } catch (error) {
      console.error('Failed to load chats:', error);
      setError('Failed to load chats');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    try {
      const { success, error } = await deleteChat(chatId);
      if (success) {
        setChats(chats.filter((chat) => chat.id !== chatId));
      } else if (error) {
        setError(error);
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
      setError('Failed to delete chat');
    }
  };

  return (
    <AdminLayout title="Chat Review" description="Review flagged chats">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Flagged Chats Review</h1>
        </div>

        {error && (
          <div className="p-4 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-md">
            {error}
          </div>
        )}

        {loading ? (
          <div className="text-center">Loading chats...</div>
        ) : (
          <div className="grid gap-4">
            {chats.map((chat) => (
              <Card key={chat.id}>
                <div className="p-4 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        Chat ID: {chat.id}
                      </span>
                      <span className="mx-2">•</span>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {new Date(chat.created_at).toLocaleString()}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {chat.messages &&
                      chat.messages.map((message, index) => (
                        <div
                          key={message.id}
                          className="border-b border-gray-200 dark:border-gray-700 last:border-0 pb-2"
                        >
                          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            {message.role === 'user' ? 'User' : 'Assistant'}
                          </h4>
                          <p className="mt-1 text-gray-900 dark:text-white whitespace-pre-wrap">
                            {message.content}
                          </p>
                          {message.feedback && (
                            <div className="mt-2">
                              <span
                                className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                  message.feedback === 'positive'
                                    ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                                    : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                                }`}
                              >
                                {message.feedback === 'positive' ? 'Positive' : 'Negative'} Feedback
                              </span>
                              {message.feedback_text && (
                                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                                  "{message.feedback_text}"
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                  </div>

                  <div className="flex justify-end">
                    <Button variant="destructive" onClick={() => handleDeleteChat(chat.id)}>
                      Delete Chat
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
            {chats.length === 0 && !loading && (
              <div className="text-center text-gray-500 dark:text-gray-400">
                No flagged chats found.
              </div>
            )}
          </div>
        )}
      </div>
    </AdminLayout>
  );
};

export default withAdmin(ChatReview, 'Chat Review');
