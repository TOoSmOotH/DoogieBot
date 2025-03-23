import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ApiResponse, PaginatedResponse, PaginationParams, Token } from '@/types';
import { jwtDecode } from 'jwt-decode';

// Define API base URL constants
export const API_BASE_URL = '/api/v1';
export const API_STREAM_URL = '/v1';

// Create axios instance with base configuration
const api = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
});

// Helper function to get the correct URL for different types of requests
export const getApiUrl = (path: string, useBaseUrl = true): string => {
  // Add leading slash if missing
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;

  if (useBaseUrl) {
    // Regular API requests through axios - use /api prefix
    return `${API_BASE_URL}${normalizedPath}`;
  } else {
    // Direct requests (like EventSource) - use /v1 prefix
    return `${API_STREAM_URL}${normalizedPath}`;
  }
};

// Flag to prevent multiple refresh requests
let isRefreshing = false;
// Queue of failed requests to retry after token refresh
let failedQueue: any[] = [];

// Process the queue of failed requests
const processQueue = (error: any = null, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });

  failedQueue = [];
};

// Function to refresh the token
const refreshToken = async (): Promise<string | null> => {
  try {
    const refreshToken = localStorage.getItem('refreshToken');
    if (!refreshToken) {
      return null;
    }

    const refreshUrl = getApiUrl('/auth/refresh');
    console.log('Refreshing token at:', refreshUrl);

    const response = await axios.post<Token>(
      refreshUrl,
      { refresh_token: refreshToken },
      { headers: { 'Content-Type': 'application/json' } }
    );

    if (response.data) {
      localStorage.setItem('token', response.data.access_token);
      localStorage.setItem('refreshToken', response.data.refresh_token);
      return response.data.access_token;
    }

    return null;
  } catch (error) {
    console.error('Error refreshing token:', error);
    return null;
  }
};

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

// Add request interceptor to add auth token
api.interceptors.request.use(
  async (config) => {
    console.log('API Request:', {
      method: config.method,
      url: config.url,
      baseURL: config.baseURL,
      fullUrl: config.baseURL && config.url ? `${config.baseURL}${config.url}` : 'unknown',
      data: config.data,
      headers: config.headers,
    });

    const token = localStorage.getItem('token');

    if (token) {
      // Check if token is expired and needs refreshing
      if (isTokenExpired(token) && !config.url?.includes('/auth/refresh')) {
        if (!isRefreshing) {
          isRefreshing = true;

          const newToken = await refreshToken();

          isRefreshing = false;

          if (newToken) {
            config.headers.Authorization = `Bearer ${newToken}`;
            processQueue(null, newToken);
          } else {
            processQueue(new Error('Failed to refresh token'));
            // Clear tokens and redirect to login
            localStorage.removeItem('token');
            localStorage.removeItem('refreshToken');
            if (typeof window !== 'undefined') {
              window.location.href = '/login';
            }
          }
        } else {
          // Add request to queue if a refresh is already in progress
          return new Promise((resolve, reject) => {
            failedQueue.push({ resolve, reject });
          })
            .then((token) => {
              config.headers.Authorization = `Bearer ${token}`;
              return config;
            })
            .catch((error) => {
              return Promise.reject(error);
            });
        }
      } else {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor to handle errors
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', {
      status: response.status,
      data: response.data,
      headers: response.headers,
      config: {
        method: response.config.method,
        url: response.config.url,
        baseURL: response.config.baseURL,
        fullUrl:
          response.config.baseURL && response.config.url
            ? `${response.config.baseURL}${response.config.url}`
            : 'unknown',
      },
    });
    return response;
  },
  async (error: AxiosError) => {
    console.error('API Error:', {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data,
      config: {
        method: error.config?.method,
        url: error.config?.url,
        baseURL: error.config?.baseURL,
        fullUrl:
          error.config?.baseURL && error.config?.url
            ? `${error.config.baseURL}${error.config.url}`
            : 'unknown',
      },
    });

    const originalRequest: any = error.config;

    // Handle 401 Unauthorized errors
    if (error.response?.status === 401 && !originalRequest._retry) {
      // Mark the request as retried
      originalRequest._retry = true;

      if (!isRefreshing) {
        isRefreshing = true;

        try {
          const newToken = await refreshToken();

          isRefreshing = false;

          if (newToken) {
            // Retry the original request with the new token
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            processQueue(null, newToken);
            return axios(originalRequest);
          } else {
            processQueue(new Error('Failed to refresh token'));
            // Clear tokens and redirect to login
            localStorage.removeItem('token');
            localStorage.removeItem('refreshToken');
            if (typeof window !== 'undefined') {
              window.location.href = '/login';
            }
          }
        } catch (refreshError) {
          processQueue(refreshError);
          // Clear tokens and redirect to login
          localStorage.removeItem('token');
          localStorage.removeItem('refreshToken');
          if (typeof window !== 'undefined') {
            window.location.href = '/login';
          }
        }
      } else {
        // Add request to queue if a refresh is already in progress
        return new Promise((resolve, reject) => {
          failedQueue.push({
            resolve: (token: string) => {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              resolve(axios(originalRequest));
            },
            reject: (err: any) => {
              reject(err);
            },
          });
        });
      }
    }

    return Promise.reject(error);
  }
);

// Helper for handling API errors
const handleApiError = (error: any): ApiResponse<any> => {
  const axiosError = error as AxiosError;
  console.log('API Error Response:', {
    status: axiosError.response?.status,
    data: axiosError.response?.data,
    headers: axiosError.response?.headers,
  });

  if (axiosError.response?.data) {
    // FastAPI returns errors in the format { detail: "error message" }
    if (typeof axiosError.response.data === 'object' && 'detail' in axiosError.response.data) {
      return { error: axiosError.response.data.detail as string };
    }
    // Handle string error responses
    if (typeof axiosError.response.data === 'string') {
      return { error: axiosError.response.data };
    }
    // Handle other object error responses
    if (typeof axiosError.response.data === 'object') {
      if ('error' in axiosError.response.data) {
        return { error: axiosError.response.data.error as string };
      }
      if ('message' in axiosError.response.data) {
        return { error: axiosError.response.data.message as string };
      }
    }
  }
  // Fallback to axios error message or generic error
  return { error: axiosError.message || 'An unexpected error occurred' };
};

// Generic GET request
export const get = async <T>(
  path: string,
  params?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> => {
  try {
    const url = getApiUrl(path);
    console.log('GET request to:', url);
    const response: AxiosResponse<T> = await api.get(url, { ...config, params });
    return { data: response.data };
  } catch (error) {
    return handleApiError(error);
  }
};

// Generic POST request
export const post = async <T>(
  path: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> => {
  try {
    const url = getApiUrl(path);
    console.log('POST request to:', url, 'with data:', data);
    const response: AxiosResponse<T> = await api.post(url, data, config);
    return { data: response.data };
  } catch (error) {
    console.error('POST error:', error);
    return handleApiError(error);
  }
};

// Generic PUT request
export const put = async <T>(
  path: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> => {
  try {
    const url = getApiUrl(path);
    console.log('PUT request to:', url);
    const response: AxiosResponse<T> = await api.put(url, data, config);
    return { data: response.data };
  } catch (error) {
    return handleApiError(error);
  }
};

// Generic DELETE request
export const del = async <T>(
  path: string,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> => {
  try {
    const url = getApiUrl(path);
    console.log('DELETE request to:', url);
    const response: AxiosResponse<T> = await api.delete(url, config);
    return { data: response.data };
  } catch (error) {
    return handleApiError(error);
  }
};

// Paginated GET request
export const getPaginated = async <T>(
  path: string,
  params?: PaginationParams & Record<string, any>,
  config?: AxiosRequestConfig
): Promise<ApiResponse<PaginatedResponse<T>>> => {
  try {
    const url = getApiUrl(path);
    console.log('Paginated GET request to:', url);
    const response: AxiosResponse<PaginatedResponse<T>> = await api.get(url, {
      ...config,
      params: {
        ...params,
        page: params?.page || 1,
        size: params?.size || 10,
      },
    });
    return { data: response.data };
  } catch (error) {
    return handleApiError(error);
  }
};

// Function to set tokens in localStorage
export const setToken = (token: Token): void => {
  localStorage.setItem('token', token.access_token);
  localStorage.setItem('refreshToken', token.refresh_token);
};

// Function to get access token from localStorage
export const getToken = (): string | null => {
  return localStorage.getItem('token');
};

// Function to get refresh token from localStorage
export const getRefreshToken = (): string | null => {
  return localStorage.getItem('refreshToken');
};

// Function to remove tokens from localStorage
export const removeToken = (): void => {
  localStorage.removeItem('token');
  localStorage.removeItem('refreshToken');
};

export default api;
