import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/contexts/AuthContext';
import AdminLayout from '@/components/layout/AdminLayout';

export function withAdmin<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  title?: string
) {
  return function AdminProtectedComponent(props: P) {
    const { isAuthenticated, isAdmin, isLoading } = useAuth();
    const router = useRouter();

    useEffect(() => {
      if (!isLoading && (!isAuthenticated || !isAdmin)) {
        router.push('/login');
      }
    }, [isLoading, isAuthenticated, isAdmin, router]);

    if (isLoading) {
      return (
        <AdminLayout title={title}>
          <div className="flex justify-center items-center h-64">
            <p>Loading...</p>
          </div>
        </AdminLayout>
      );
    }

    if (!isAuthenticated || !isAdmin) {
      return null;
    }

    return <WrappedComponent {...props} />;
  };
}

export default withAdmin;
