import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { User } from '@/types';
import {
  getCurrentUser,
  login as loginService,
  logout as logoutService,
  register as registerService,
  refreshToken as refreshTokenService,
} from '@/services/auth';
import { useRouter } from 'next/router';
import { getToken } from '@/services/api';
import { jwtDecode } from 'jwt-decode';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  refreshToken: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // Check if token is expired
  const isTokenExpired = (token: string): boolean => {
    try {
      const decoded: any = jwtDecode(token);
      const currentTime = Date.now() / 1000;

      // Add a buffer of 30 seconds to refresh before actual expiration
      return decoded.exp < currentTime + 30;
    } catch (error) {
      return true;
    }
  };

  // Function to refresh token
  const refreshToken = async (): Promise<boolean> => {
    try {
      const { success } = await refreshTokenService();
      return success;
    } catch (error) {
      console.error('Token refresh error:', error);
      return false;
    }
  };

  useEffect(() => {
    const initAuth = async () => {
      try {
        const token = getToken();
        if (token) {
          // Check if token is expired and needs refreshing
          if (isTokenExpired(token)) {
            const refreshSuccess = await refreshToken();
            if (!refreshSuccess) {
              logoutService();
              setIsLoading(false);
              return;
            }
          }

          const { user, error } = await getCurrentUser();
          if (user) {
            setUser(user);
          } else if (error) {
            console.error('Error fetching user:', error);
            logoutService();
          }
        }
      } catch (error) {
        console.error('Auth initialization error:', error);
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const { user, error } = await loginService(email, password);
      if (user) {
        setUser(user);
        return { success: true };
      }
      return { success: false, error: error || 'Login failed' };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'An unexpected error occurred' };
    }
  };

  const register = async (email: string, password: string) => {
    try {
      const { success, error } = await registerService({ email, password });
      if (success) {
        return { success: true };
      }
      return { success: false, error: error || 'Registration failed' };
    } catch (error) {
      console.error('Registration error:', error);
      return { success: false, error: 'An unexpected error occurred' };
    }
  };

  const logout = () => {
    logoutService();
    setUser(null);
    router.push('/login');
  };

  const value = {
    user,
    isLoading,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin' || false,
    login,
    register,
    logout,
    refreshToken,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
