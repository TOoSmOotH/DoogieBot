import React, { ReactNode } from 'react';
import Layout from './Layout';
import Link from 'next/link';
import { useRouter } from 'next/router';

interface AdminLayoutProps {
  children: ReactNode;
  title?: string;
  description?: string;
}

const AdminLayout = ({
  children,
  title = 'Admin Dashboard - Doogie Chat Bot',
  description = 'Admin dashboard for Doogie Chat Bot',
}: AdminLayoutProps) => {
  const router = useRouter();

  const isActive = (path: string) => {
    return router.pathname === path;
  };

  const navItems = [
    { href: '/admin', label: 'Dashboard', exact: true },
    { href: '/admin/users', label: 'User Management' },
    { href: '/admin/documents', label: 'Documents' },
    { href: '/admin/llm', label: 'LLM Configuration' },
    { href: '/admin/rag', label: 'RAG Management' },
    { href: '/admin/mcp', label: 'MCP Management' }, // Added MCP link
    { href: '/admin/chats', label: 'Chat Review' },
    { href: '/admin/system', label: 'System Settings' },
  ];

  return (
    <Layout title={title} description={description}>
      <div className="flex flex-col md:flex-row gap-6">
        <aside className="w-full md:w-64 bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
          <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Admin</h2>
          <nav className="space-y-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`block px-3 py-2 rounded-md text-sm font-medium ${
                  (item.exact ? isActive(item.href) : router.pathname.startsWith(item.href))
                    ? 'bg-primary-100 dark:bg-primary-900 text-primary-900 dark:text-primary-100'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </aside>
        <div className="flex-1">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            {children}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default AdminLayout;