/** @type {import('next').NextConfig} */
const isDev = process.env.NODE_ENV !== 'production';

const nextConfig = {
  // Enable React strict mode in production, disable in development to prevent double renders
  reactStrictMode: !isDev,
  // Production-specific optimizations
  productionBrowserSourceMaps: false,
  // Development-specific configurations
  ...(isDev && {
    webpackDevMiddleware: (config) => {
      // Required for hot reloading in Docker
      config.watchOptions = {
        poll: 1000, // Check for changes every second
        aggregateTimeout: 300, // Delay before rebuilding
        ignored: /node_modules/,
      };
      return config;
    },
  }),
  // API rewrites for both dev and prod
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://localhost:8000/api/v1/:path*', // Proxy regular API requests (changed to localhost)
      },
      {
        source: '/api/v1/:path*/',
        destination: 'http://localhost:8000/api/v1/:path*/', // Handle trailing slashes
      },
      {
        source: '/v1/:path*',
        destination: 'http://localhost:8000/api/v1/:path*', // Proxy EventSource requests
      },
    ];
  },
};

module.exports = nextConfig;
