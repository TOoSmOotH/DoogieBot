import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import AdminLayout from '@/components/layout/AdminLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { getUsers } from '@/services/user';
import { getRAGStats } from '@/services/document';
import { getActiveLLMConfig } from '@/services/llm';
import { getFlaggedChats } from '@/services/chat';
import withAdmin from '@/utils/withAdmin';

function AdminDashboard() {
  const router = useRouter();
  const [userCount, setUserCount] = useState<number | null>(null);
  const [pendingUserCount, setPendingUserCount] = useState<number | null>(null);
  const [documentCount, setDocumentCount] = useState<number | null>(null);
  const [flaggedChatCount, setFlaggedChatCount] = useState<number | null>(null);
  const [currentLLM, setCurrentLLM] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load dashboard data
  useEffect(() => {
    const loadDashboardData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Get user count
        const { users } = await getUsers({ page: 1, size: 1 });
        if (users) {
          setUserCount(users.total);
        }

        // Get pending user count
        const { users: pendingUsers } = await getUsers({ page: 1, size: 1 });
        if (pendingUsers) {
          setPendingUserCount(pendingUsers.total);
        }

        // Get RAG stats
        const { stats } = await getRAGStats();
        if (stats) {
          setDocumentCount(stats.document_count);
        }

        // Get flagged chat count
        const { chats } = await getFlaggedChats({ page: 1, size: 1 });
        if (chats) {
          setFlaggedChatCount(chats.total);
        }

        // Get current LLM config
        const { config } = await getActiveLLMConfig();
        if (config) {
          setCurrentLLM(config.provider);
        }
      } catch (err) {
        console.error('Error loading dashboard data:', err);
        setError('Failed to load dashboard data');
      } finally {
        setIsLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  return (
    <AdminLayout title="Admin Dashboard - Doogie Chat Bot">
      <h1 className="text-3xl font-bold mb-6">Admin Dashboard</h1>

      {error && (
        <div className="mb-6 p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-400 rounded-md">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* User Stats */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Total Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {isLoading ? '...' : userCount !== null ? userCount : 'N/A'}
            </div>
          </CardContent>
        </Card>

        {/* Pending Users */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Pending Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {isLoading ? '...' : pendingUserCount !== null ? pendingUserCount : 'N/A'}
            </div>
          </CardContent>
        </Card>

        {/* Document Count */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {isLoading ? '...' : documentCount !== null ? documentCount : 'N/A'}
            </div>
          </CardContent>
        </Card>

        {/* Flagged Chats */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Flagged Chats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {isLoading ? '...' : flaggedChatCount !== null ? flaggedChatCount : 'N/A'}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Current LLM Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>LLM Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Current Provider
                </p>
                <p className="text-lg">{isLoading ? '...' : currentLLM || 'Not configured'}</p>
              </div>
              <div className="pt-2">
                <button
                  onClick={() => router.push('/admin/llm')}
                  className="text-primary-600 dark:text-primary-400 hover:underline text-sm"
                >
                  Configure LLM Settings →
                </button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <button
                onClick={() => router.push('/admin/users')}
                className="w-full text-left p-3 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-5 h-5 mr-2"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M15 19.128a9.38 9.38 0 0 0 2.625.372 9.337 9.337 0 0 0 4.121-.952 4.125 4.125 0 0 0-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 0 1 8.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0 1 11.964-3.07M12 6.375a3.375 3.375 0 1 1-6.75 0 3.375 3.375 0 0 1 6.75 0Zm8.25 2.25a2.625 2.625 0 1 1-5.25 0 2.625 2.625 0 0 1 5.25 0Z"
                  />
                </svg>
                Manage Users
              </button>
              <button
                onClick={() => router.push('/admin/rag')}
                className="w-full text-left p-3 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-5 h-5 mr-2"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
                  />
                </svg>
                Manage Documents
              </button>
              <button
                onClick={() => router.push('/admin/chats')}
                className="w-full text-left p-3 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-5 h-5 mr-2"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z"
                  />
                </svg>
                Review Flagged Chats
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </AdminLayout>
  );
}

export default withAdmin(AdminDashboard, 'Admin Dashboard - Doogie Chat Bot');
