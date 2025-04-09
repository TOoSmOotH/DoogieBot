import { Token, User, UserCreate, LoginOptions } from '@/types';
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
  last_login: backendUser.last_login
});

// Login user
export const login = async (email: string, password: string, options?: LoginOptions): Promise<{ user?: User; error?: string }> => {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);

  try {
    // Use the post function which now has special handling for auth endpoints
    console.log('Attempting login with:', email);
    console.log('Remember me:', options?.rememberMe || false);

    // Important: Don't add trailing slash for login since backend expects exact path
    const loginPath = '/auth/login'; // Don't normalize this path
    const url = `${API_BASE_URL}${loginPath}`; // Manually construct URL

    console.log('Login URL:', url);
    console.log('Form data:', Object.fromEntries(formData));

    // Add remember_me to the URL parameters
    const urlWithParams = options?.rememberMe
      ? `${url}?remember_me=true`
      : url;
    
    console.log('Login URL with params:', urlWithParams);
    
    // Use direct axios call to bypass our API layer's automatic trailing slash addition
    const response = await fetch(urlWithParams, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });
    
    console.log('Login response status:', response.status);
    
    if (!response.ok) {
      console.error('Login failed with status:', response.status);
      let errorMessage = `Authentication failed with status ${response.status}`;
      try {
        const errorData = await response.json();
        // FastAPI often returns errors in { "detail": "..." }
        if (errorData && errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch (parseError) {
        // If parsing fails, stick with the status-based message
        console.error('Could not parse error response body:', parseError);
      }
      return { error: errorMessage };
    }

    const data = await response.json() as Token;

    console.log('Login response data:', data);

    // Set token using the data we got from the fetch response
    setToken(data, options?.rememberMe);
    
    // Now that we're authenticated, get user info
    const userResponse = await get<any>('/users/me');
    console.log('User response:', userResponse);
    
    if (userResponse.error) {
      // If error, clean up tokens to avoid half-authenticated state
      removeToken();
      return { error: userResponse.error };
    }
    
    if (userResponse.data) {
      return { user: transformUser(userResponse.data) };
    }

    return { error: 'Failed to get user information' };
  } catch (error) {
    console.error("Login error:", error);
    return { error: 'An unexpected error occurred. Please try again.' };
  }
};

// Register new user
export const register = async (userData: UserCreate): Promise<{ success?: boolean; error?: string }> => {
  const response = await post<User>('/auth/register', userData);

  if (response.error) {
    return { error: response.error };
  }

  return { success: true };
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
    // Important: Don't add trailing slash for refresh to ensure exact path match
    const refreshPath = '/auth/refresh';
    const url = `${API_BASE_URL}${refreshPath}`;
    
    console.log('Refresh token URL:', url);
    
    // Use JSON for refresh token request (this endpoint expects JSON, not form data)
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    
    console.log('Refresh token response status:', response.status);
    
    if (!response.ok) {
      console.error('Refresh token failed with status:', response.status);
      return { success: false, error: `Token refresh failed with status ${response.status}` };
    }
    
    const data = await response.json() as Token;
    console.log('Refresh token response data:', data);
    
    // Preserve the current storage type when refreshing
    const storageType = localStorage.getItem('tokenStorage') || 'local';
    const rememberMe = storageType === 'local';
    setToken(data, rememberMe);
    return { success: true };
  } catch (error) {
    console.error("Token refresh error:", error);
    return { success: false, error: 'An unexpected error occurred while refreshing token' };
  }
};

// Logout user
export const logout = (): void => {
  removeToken();
};
