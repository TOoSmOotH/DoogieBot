import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { getMcpConfigs, MCPServerConfigResponse } from '@/services/mcp';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'; // Assuming shadcn/ui table components
import { useErrorHandler } from '@/hooks/useErrorHandler';
import withAdmin from '@/utils/withAdmin'; // Import the HOC
import { Spinner } from '@/components/ui/Spinner';
import MCPServerStatus from '@/components/mcp/MCPServerStatus'; // Import the status component
import AdminLayout from '@/components/layout/AdminLayout'; // Import AdminLayout

const MCPDashboardPage: React.FC = () => {
  const [configs, setConfigs] = useState<MCPServerConfigResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const { handleError } = useErrorHandler();

  const fetchConfigs = useCallback(async () => {
    setLoading(true);
    try {
      const fetchedConfigs = await getMcpConfigs();
      setConfigs(fetchedConfigs);
    } catch (error) {
      handleError(error, 'Failed to load MCP configurations.');
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  useEffect(() => {
    fetchConfigs();
  }, [fetchConfigs]); // Use fetchConfigs directly as dependency

  // Handler to remove a config from the list after successful deletion
  const handleDeleteConfig = (deletedId: string) => {
    setConfigs(prevConfigs => prevConfigs.filter(config => config.id !== deletedId));
  };

  return (
    // Wrap content with AdminLayout
    <AdminLayout title="MCP Server Management">
      {/* Removed outer div and h1 */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Configured Servers</CardTitle>
          <Link href="/admin/mcp/new" passHref>
            <Button>Add New Server</Button>
          </Link>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex justify-center items-center h-40">
              <Spinner />
            </div>
          ) : configs.length === 0 ? (
            <p className="text-center text-gray-500">No MCP servers configured yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Enabled</TableHead>
                  <TableHead>Command</TableHead>
                  <TableHead className="text-right">Actions</TableHead> {/* Align actions right */}
                </TableRow>
              </TableHeader>
              <TableBody>
                {configs.map((config) => (
                  <TableRow key={config.id}>
                    <TableCell className="font-medium">{config.name}</TableCell>
                    <TableCell>{config.enabled ? 'Yes' : 'No'}</TableCell>
                    <TableCell className="font-mono text-sm">
                      {/* Display only the image name or first few args for brevity */}
                      {config.command || 'docker'} {config.args ? config.args.slice(0, 3).join(' ') + (config.args.length > 3 ? '...' : '') : ''}
                    </TableCell>
                    <TableCell className="flex justify-end items-center space-x-1"> {/* Use flex for alignment */}
                      <Link href={`/admin/mcp/${config.id}`} passHref>
                        <Button variant="outline" size="sm">Edit</Button>
                      </Link>
                      {/* Integrate the status component */}
                      <MCPServerStatus
                        configId={config.id}
                        configName={config.name}
                        onDeleteSuccess={handleDeleteConfig}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </AdminLayout>
  );
};

// Wrap the component with the HOC to protect the route
export default withAdmin(MCPDashboardPage);