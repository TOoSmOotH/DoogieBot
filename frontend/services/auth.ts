import { Token, User, UserCreate } from '@/types';
import { get, post, setToken, removeToken, getRefreshToken, API_BASE_URL } from './api';

// Transform backend user response to frontend user format
const transformUser = (backendUser: any): User => ({
  id: backendUser.id,
  email: backendUser.email,
  status: backendUser.status,
  role: backendUser.role,
  theme_preference: backendUser.theme_preference,
  created_at: backendUser.created_at,
  updated_at: backendUser.updated_at,
  last_login: backendUser.last_login,
});

// Login user
export const login = async (
  email: string,
  password: string
): Promise<{ user?: User; error?: string }> => {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);

  try {
    // Use the post function which now has special handling for auth endpoints
    console.log('Attempting login with:', email);

    const response = await post<Token>('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    console.log('Login response:', response);

    if (response.error) {
      return { error: response.error };
    }

    if (response.data) {
      setToken(response.data);
      const userResponse = await get<any>('/users/me');
      console.log('User response:', userResponse);

      if (userResponse.data) {
        return { user: transformUser(userResponse.data) };
      }
    }

    return { error: 'Failed to get user information' };
  } catch (error) {
    console.error('Login error:', error);
    return { error: 'An unexpected error occurred. Please try again.' };
  }
};

// Register new user
export const register = async (
  userData: UserCreate
): Promise<{ success?: boolean; error?: string }> => {
  try {
    const response = await post<User>('/auth/register', userData);

    if (response.error) {
      console.error('Registration error detail:', response.error);
      // Pass through the backend error message directly
      return { error: response.error };
    }

    return { success: true };
  } catch (error) {
    console.error('Unexpected registration error:', error);
    return { error: 'An unexpected error occurred during registration. Please try again.' };
  }
};

// Get current user
export const getCurrentUser = async (): Promise<{ user?: User; error?: string }> => {
  const response = await get<any>('/users/me');

  if (response.error) {
    return { error: response.error };
  }

  return { user: transformUser(response.data) };
};

// Refresh token
export const refreshToken = async (): Promise<{ success: boolean; error?: string }> => {
  const refreshToken = getRefreshToken();

  if (!refreshToken) {
    return { success: false, error: 'No refresh token available' };
  }

  try {
    const response = await post<Token>('/auth/refresh', { refresh_token: refreshToken });

    if (response.error) {
      return { success: false, error: response.error };
    }

    if (response.data) {
      setToken(response.data);
      return { success: true };
    }

    return { success: false, error: 'Failed to refresh token' };
  } catch (error) {
    console.error('Token refresh error:', error);
    return { success: false, error: 'An unexpected error occurred while refreshing token' };
  }
};

// Logout user
export const logout = (): void => {
  removeToken();
};
