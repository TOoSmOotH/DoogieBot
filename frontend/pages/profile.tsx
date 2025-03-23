import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import Layout from '@/components/layout/Layout';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from '@/components/ui/Card';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { updatePassword, deleteAccount, exportChatHistory } from '@/services/user';

// Define password change form schema
const passwordChangeSchema = z
  .object({
    currentPassword: z.string().min(1, 'Current password is required'),
    newPassword: z.string().min(8, 'New password must be at least 8 characters'),
    confirmPassword: z.string().min(8, 'Confirm password must be at least 8 characters'),
  })
  .refine((data) => data.newPassword === data.confirmPassword, {
    message: "Passwords don't match",
    path: ['confirmPassword'],
  });

type PasswordChangeFormValues = z.infer<typeof passwordChangeSchema>;

// Define account deletion form schema
const deleteAccountSchema = z.object({
  password: z.string().min(1, 'Password is required to confirm deletion'),
});

type DeleteAccountFormValues = z.infer<typeof deleteAccountSchema>;

export default function Profile() {
  const { user, isAuthenticated, isLoading: authLoading, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const router = useRouter();
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [isDeletingAccount, setIsDeletingAccount] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [passwordChangeSuccess, setPasswordChangeSuccess] = useState<string | null>(null);
  const [passwordChangeError, setPasswordChangeError] = useState<string | null>(null);
  const [deleteAccountError, setDeleteAccountError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  const {
    register: registerPasswordChange,
    handleSubmit: handleSubmitPasswordChange,
    formState: { errors: passwordChangeErrors },
    reset: resetPasswordChangeForm,
  } = useForm<PasswordChangeFormValues>({
    resolver: zodResolver(passwordChangeSchema),
    defaultValues: {
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
    },
  });

  const {
    register: registerDeleteAccount,
    handleSubmit: handleSubmitDeleteAccount,
    formState: { errors: deleteAccountErrors },
    reset: resetDeleteAccountForm,
  } = useForm<DeleteAccountFormValues>({
    resolver: zodResolver(deleteAccountSchema),
    defaultValues: {
      password: '',
    },
  });

  const handlePasswordChange = async (data: PasswordChangeFormValues) => {
    setIsChangingPassword(true);
    setPasswordChangeSuccess(null);
    setPasswordChangeError(null);

    try {
      const { success, error } = await updatePassword(data.currentPassword, data.newPassword);
      if (success) {
        setPasswordChangeSuccess('Password updated successfully');
        resetPasswordChangeForm();
      } else {
        setPasswordChangeError(error || 'Failed to update password');
      }
    } catch (err) {
      console.error('Error updating password:', err);
      setPasswordChangeError('An unexpected error occurred');
    } finally {
      setIsChangingPassword(false);
    }
  };

  const handleDeleteAccount = async (data: DeleteAccountFormValues) => {
    setIsDeletingAccount(true);
    setDeleteAccountError(null);

    try {
      const { success, error } = await deleteAccount(data.password);
      if (success) {
        logout();
        router.push('/');
      } else {
        setDeleteAccountError(error || 'Failed to delete account');
      }
    } catch (err) {
      console.error('Error deleting account:', err);
      setDeleteAccountError('An unexpected error occurred');
    } finally {
      setIsDeletingAccount(false);
    }
  };

  const handleExportHistory = async () => {
    setIsExporting(true);
    setExportError(null);

    try {
      const { url, error } = await exportChatHistory();
      if (url) {
        // Create a temporary link and trigger download
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_history.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      } else {
        setExportError(error || 'Failed to export chat history');
      }
    } catch (err) {
      console.error('Error exporting chat history:', err);
      setExportError('An unexpected error occurred');
    } finally {
      setIsExporting(false);
    }
  };

  if (authLoading) {
    return (
      <Layout title="Profile - Doogie Chat Bot">
        <div className="flex justify-center items-center h-64">
          <p>Loading...</p>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Profile - Doogie Chat Bot">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Your Profile</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* User Information */}
          <Card>
            <CardHeader>
              <CardTitle>Account Information</CardTitle>
              <CardDescription>Your account details</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Email</p>
                  <p className="text-lg">{user?.email}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Role</p>
                  <p className="text-lg">{user?.role === 'admin' ? 'Administrator' : 'User'}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Account Created
                  </p>
                  <p className="text-lg">
                    {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Preferences */}
          <Card>
            <CardHeader>
              <CardTitle>Preferences</CardTitle>
              <CardDescription>Customize your experience</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="font-medium">Theme</p>
                  <div className="flex items-center">
                    <span className="mr-2">{theme === 'dark' ? 'Dark' : 'Light'}</span>
                    <button
                      onClick={toggleTheme}
                      className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600"
                      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                    >
                      {theme === 'dark' ? (
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.5}
                          stroke="currentColor"
                          className="w-5 h-5"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M12 3v2.25m6.364.386-1.591 1.591M21 12h-2.25m-.386 6.364-1.591-1.591M12 18.75V21m-4.773-4.227-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z"
                          />
                        </svg>
                      ) : (
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.5}
                          stroke="currentColor"
                          className="w-5 h-5"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M21.752 15.002A9.72 9.72 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z"
                          />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button
                variant="outline"
                onClick={handleExportHistory}
                isLoading={isExporting}
                className="w-full"
              >
                Export Chat History
              </Button>
              {exportError && (
                <p className="mt-2 text-sm text-red-600 dark:text-red-400">{exportError}</p>
              )}
            </CardFooter>
          </Card>

          {/* Change Password */}
          <Card>
            <CardHeader>
              <CardTitle>Change Password</CardTitle>
              <CardDescription>Update your password</CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmitPasswordChange(handlePasswordChange)}>
              <CardContent className="space-y-4">
                {passwordChangeSuccess && (
                  <div className="p-3 bg-green-100 dark:bg-green-900/30 border border-green-400 dark:border-green-800 text-green-700 dark:text-green-400 rounded-md">
                    {passwordChangeSuccess}
                  </div>
                )}
                {passwordChangeError && (
                  <div className="p-3 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-400 rounded-md">
                    {passwordChangeError}
                  </div>
                )}
                <Input
                  label="Current Password"
                  type="password"
                  error={passwordChangeErrors.currentPassword?.message}
                  {...registerPasswordChange('currentPassword')}
                />
                <Input
                  label="New Password"
                  type="password"
                  error={passwordChangeErrors.newPassword?.message}
                  {...registerPasswordChange('newPassword')}
                />
                <Input
                  label="Confirm New Password"
                  type="password"
                  error={passwordChangeErrors.confirmPassword?.message}
                  {...registerPasswordChange('confirmPassword')}
                />
              </CardContent>
              <CardFooter>
                <Button type="submit" isLoading={isChangingPassword} className="w-full">
                  Update Password
                </Button>
              </CardFooter>
            </form>
          </Card>

          {/* Delete Account */}
          <Card>
            <CardHeader>
              <CardTitle>Delete Account</CardTitle>
              <CardDescription>Permanently delete your account and all data</CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmitDeleteAccount(handleDeleteAccount)}>
              <CardContent className="space-y-4">
                {deleteAccountError && (
                  <div className="p-3 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-400 rounded-md">
                    {deleteAccountError}
                  </div>
                )}
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  This action cannot be undone. All your data, including chat history, will be
                  permanently deleted.
                </p>
                <Input
                  label="Enter your password to confirm"
                  type="password"
                  error={deleteAccountErrors.password?.message}
                  {...registerDeleteAccount('password')}
                />
              </CardContent>
              <CardFooter>
                <Button
                  type="submit"
                  variant="destructive"
                  isLoading={isDeletingAccount}
                  className="w-full"
                >
                  Delete Account
                </Button>
              </CardFooter>
            </form>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
