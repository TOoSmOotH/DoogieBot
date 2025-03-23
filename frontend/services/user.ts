import { PaginatedResponse, PaginationParams, User, UserCreate, UserUpdate } from '@/types';
import { del, get, getPaginated, post, put } from './api';

// Get all users with pagination (admin only)
export const getUsers = async (
  params?: PaginationParams
): Promise<{
  users?: PaginatedResponse<User>;
  error?: string;
}> => {
  const response = await getPaginated<User>('/users', params);

  if (response.error) {
    return { error: response.error };
  }

  return { users: response.data };
};

// Get pending users (admin only)
export const getPendingUsers = async (
  params?: PaginationParams
): Promise<{
  users?: PaginatedResponse<User>;
  error?: string;
}> => {
  const response = await getPaginated<User>('/users/pending', params);

  if (response.error) {
    return { error: response.error };
  }

  return { users: response.data };
};

// Get a single user by ID (admin only)
export const getUser = async (userId: string): Promise<{ user?: User; error?: string }> => {
  const response = await get<User>(`/users/${userId}`);

  if (response.error) {
    return { error: response.error };
  }

  return { user: response.data };
};

// Create a new user (admin only)
export const createUser = async (
  userData: UserCreate
): Promise<{ user?: User; error?: string }> => {
  const response = await post<User>('/users', userData);

  if (response.error) {
    return { error: response.error };
  }

  return { user: response.data };
};

// Update a user (admin only or self)
export const updateUser = async (
  userId: string,
  userData: UserUpdate
): Promise<{ success?: boolean; error?: string }> => {
  const response = await put(`/users/${userId}`, userData);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Delete a user (admin only)
export const deleteUser = async (
  userId: string
): Promise<{ success?: boolean; error?: string }> => {
  const response = await del(`/users/${userId}`);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Approve a pending user (admin only)
export const approveUser = async (
  userId: string,
  isAdmin: boolean = false
): Promise<{ success?: boolean; error?: string }> => {
  const response = await put(`/users/${userId}/approve`, { role: isAdmin ? 'admin' : 'user' });

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Update current user's password
export const updatePassword = async (
  currentPassword: string,
  newPassword: string
): Promise<{ success?: boolean; error?: string }> => {
  const response = await put('/users/me/password', {
    current_password: currentPassword,
    new_password: newPassword,
  });

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Delete current user's account
export const deleteAccount = async (
  password: string
): Promise<{ success?: boolean; error?: string }> => {
  const response = await post('/users/me/delete', { password });

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
};

// Export chat history
export const exportChatHistory = async (): Promise<{ url?: string; error?: string }> => {
  const response = await get<{ download_url: string }>('/users/me/export');

  if (response.error) {
    return { error: response.error };
  }

  return { url: response.data?.download_url };
};
