import { useEffect, useState } from 'react';
import AdminLayout from '../../components/layout/AdminLayout';
import { getUsers, getPendingUsers, updateUser, approveUser } from '../../services/user';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { User, PaginatedResponse } from '@/types';
import withAdmin from '@/utils/withAdmin';

interface UserListParams {
  page: number;
  size: number;
  filter: 'all' | 'pending';
}

const UserManagement = () => {
  const [params, setParams] = useState<UserListParams>({
    page: 1,
    size: 10,
    filter: 'all',
  });
  const [users, setUsers] = useState<PaginatedResponse<User> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadUsers();
  }, [params]);

  const loadUsers = async () => {
    try {
      setLoading(true);
      setError(null);

      const response =
        params.filter === 'all'
          ? await getUsers({ page: params.page, size: params.size })
          : await getPendingUsers({ page: params.page, size: params.size });

      if (response.error) {
        setError(response.error);
      } else if (response.users) {
        setUsers(response.users);
      }
    } catch (error) {
      setError('Failed to load users. Please try again.');
      console.error('Failed to load users:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    setParams((prev) => ({ ...prev, page: newPage }));
  };

  const handleFilterChange = (filter: 'all' | 'pending') => {
    setParams((prev) => ({ ...prev, filter, page: 1 }));
  };

  const toggleUserStatus = async (userId: string, status: string) => {
    try {
      const newStatus = status === 'active' ? 'inactive' : 'active';
      await updateUser(userId, { status: newStatus });
      loadUsers(); // Reload the user list
    } catch (error) {
      console.error('Failed to update user:', error);
      setError('Failed to update user status. Please try again.');
    }
  };

  const toggleAdminStatus = async (userId: string, role: string) => {
    try {
      const newRole = role === 'admin' ? 'user' : 'admin';
      await updateUser(userId, { role: newRole });
      loadUsers(); // Reload the user list
    } catch (error) {
      console.error('Failed to update user:', error);
      setError('Failed to update admin status. Please try again.');
    }
  };

  return (
    <AdminLayout title="User Management" description="Manage user accounts and permissions">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">User Management</h1>
          <div className="flex gap-2">
            <Button
              variant={params.filter === 'all' ? 'default' : 'outline'}
              onClick={() => handleFilterChange('all')}
            >
              All Users
            </Button>
            <Button
              variant={params.filter === 'pending' ? 'default' : 'outline'}
              onClick={() => handleFilterChange('pending')}
            >
              Pending Users
            </Button>
          </div>
        </div>

        {error && (
          <div className="p-4 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-md">
            {error}
          </div>
        )}

        {loading ? (
          <div className="text-center py-8">Loading users...</div>
        ) : (
          <div className="space-y-6">
            <div className="grid gap-4">
              {users?.items?.map((user: User) => (
                <Card key={user.id}>
                  <div className="flex items-center justify-between p-4">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {user.email}
                      </h3>
                      <div className="mt-1 flex gap-2">
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            user.status === 'active'
                              ? 'bg-green-100 text-green-800'
                              : user.status === 'pending'
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {user.status === 'active'
                            ? 'Active'
                            : user.status === 'pending'
                              ? 'Pending'
                              : 'Inactive'}
                        </span>
                        {user.role === 'admin' && (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            Admin
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => toggleUserStatus(user.id, user.status)}
                        variant={user.status === 'active' ? 'destructive' : 'default'}
                      >
                        {user.status === 'active' ? 'Deactivate' : 'Activate'}
                      </Button>
                      <Button
                        onClick={() => toggleAdminStatus(user.id, user.role)}
                        variant={user.role === 'admin' ? 'destructive' : 'secondary'}
                      >
                        {user.role === 'admin' ? 'Remove Admin' : 'Make Admin'}
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>

            {users && users.total > params.size && (
              <div className="flex items-center justify-center gap-2 py-4">
                <Button
                  variant="outline"
                  onClick={() => handlePageChange(params.page - 1)}
                  disabled={params.page === 1}
                >
                  Previous
                </Button>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Page {params.page} of {Math.ceil(users.total / params.size)}
                </span>
                <Button
                  variant="outline"
                  onClick={() => handlePageChange(params.page + 1)}
                  disabled={params.page >= Math.ceil(users.total / params.size)}
                >
                  Next
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </AdminLayout>
  );
};

export default withAdmin(UserManagement, 'User Management');
