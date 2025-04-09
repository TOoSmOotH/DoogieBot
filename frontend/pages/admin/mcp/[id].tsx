import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';
import { getMcpConfig, updateMcpConfig, MCPServerConfigUpdate, MCPServerConfigResponse } from '@/services/mcp';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Label } from '@/components/ui/label'; // Assuming shadcn/ui label component
import { TextArea } from '@/components/ui/TextArea';
import { Checkbox } from '@/components/ui/checkbox'; // Assuming shadcn/ui checkbox component
import { useErrorHandler } from '@/hooks/useErrorHandler';
import withAdmin from '@/utils/withAdmin';
import Link from 'next/link';
import { Spinner } from '@/components/ui/Spinner';
import AdminLayout from '@/components/layout/AdminLayout'; // Import AdminLayout

const EditMCPConfigPage: React.FC = () => {
  const router = useRouter();
  const { id } = router.query; // Get the config ID from the URL query
  const { handleError } = useErrorHandler();

  const [config, setConfig] = useState<MCPServerConfigResponse | null>(null);
  const [name, setName] = useState('');
  const [args, setArgs] = useState('');
  const [envVars, setEnvVars] = useState('');
  const [enabled, setEnabled] = useState(true);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  const fetchConfig = useCallback(async (configId: string) => {
    setIsLoading(true);
    try {
      const fetchedConfig = await getMcpConfig(configId);
      setConfig(fetchedConfig);
      setName(fetchedConfig.name);
      setArgs(fetchedConfig.args ? fetchedConfig.args.join(' ') : ''); // Join args array into a string for editing with null check
      setEnvVars(fetchedConfig.env ? JSON.stringify(fetchedConfig.env, null, 2) : ''); // Format env object as JSON string
      setEnabled(fetchedConfig.enabled);
    } catch (error) {
      handleError(error, `Failed to load MCP configuration ${configId}.`);
      // Optionally redirect if config not found or not authorized
      // router.push('/admin/mcp');
    } finally {
      setIsLoading(false);
    }
  }, [handleError]); // Removed router from dependencies as it's stable

  useEffect(() => {
    if (id && typeof id === 'string') {
      fetchConfig(id);
    } else if (router.isReady && !id) {
      // Handle case where router is ready but ID is missing (shouldn't normally happen with file-based routing)
      handleError(new Error("Configuration ID is missing."), "Invalid Route");
      setIsLoading(false);
    }
    // Add router.isReady dependency to ensure 'id' is available
  }, [id, router.isReady, fetchConfig, handleError]);

  const parseArgs = (argsString: string): string[] => {
    return argsString.trim().split(/\s+/);
  };

  const parseEnvVars = (envString: string): Record<string, string> | null => {
    if (!envString.trim()) {
      return {};
    }
    try {
      const parsed = JSON.parse(envString);
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        return parsed;
      }
      throw new Error('Invalid JSON format for environment variables.');
    } catch (error) {
      console.error("Error parsing ENV JSON:", error);
      return null;
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!id || typeof id !== 'string') {
      handleError(new Error("Configuration ID is missing."), "Update Error");
      return;
    }
    setIsSubmitting(true);

    const parsedArgs = parseArgs(args);
    const parsedEnv = parseEnvVars(envVars);

    if (parsedEnv === null) {
      handleError(new Error('Invalid JSON format for Environment Variables. Please use {"KEY": "VALUE", ...} format.'), 'Form Validation Error');
      setIsSubmitting(false);
      return;
    }

     if (parsedArgs.length === 0 || !parsedArgs[0]) {
       handleError(new Error('Arguments cannot be empty.'), 'Form Validation Error');
       setIsSubmitting(false);
       return;
    }

    const updateData: MCPServerConfigUpdate = {
      // Only include fields if they have changed from the original config
      ...(config?.name !== name && { name }),
      // Always include args if they are provided
      ...(args.trim() !== '' && { args: parsedArgs }),
      // Always include env vars if they've been updated
      ...(JSON.stringify(config?.env || {}, null, 2) !== envVars && 
          { env: Object.keys(parsedEnv).length > 0 ? parsedEnv : undefined }),
      // Always include enabled status
      enabled,
    };

    // Check if there are any actual changes
    if (Object.keys(updateData).length === 0) {
        // Corrected: Removed the third argument which caused the type error
        handleError(new Error("No changes detected."), "Update Info");
        setIsSubmitting(false);
        return;
    }


    try {
      await updateMcpConfig(id, updateData);
      router.push('/admin/mcp'); // Redirect to dashboard on success
    } catch (error) {
      handleError(error, `Failed to update MCP configuration ${id}.`);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Determine the title for the layout
  const pageTitle = config ? `Edit MCP Server: ${config.name}` : 'Edit MCP Server';

  return (
    <AdminLayout title={pageTitle}>
      {isLoading ? (
        // Removed outer div
        <div className="flex justify-center items-center h-64"> {/* Added height */}
          <Spinner />
        </div>
      ) : !config ? (
        // Removed outer div
        <div className="text-center"> {/* Added text-center */}
          <p className="text-red-500">Could not load configuration data.</p>
          <Link href="/admin/mcp" passHref>
            <Button variant="outline" className="mt-4">
              Back to Dashboard
            </Button>
          </Link>
        </div>
      ) : (
        // Removed outer div and h1
        <Card>
          <form onSubmit={handleSubmit}>
            <CardHeader>
              <CardTitle>Server Configuration</CardTitle>
              <CardDescription>
                Update the details for the MCP server.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={name}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
                  required
                  placeholder="e.g., filesystem-local"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="args">Docker Arguments</Label>
                <Input
                  id="args"
                  value={args}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setArgs(e.target.value)}
                  required
                  placeholder="run -i --rm mcp/filesystem /data"
                  className="font-mono"
                />
                <p className="text-sm text-muted-foreground">
                  Enter the arguments for the 'docker' command.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="envVars">Environment Variables (JSON)</Label>
                <TextArea
                  id="envVars"
                  value={envVars}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setEnvVars(e.target.value)}
                  placeholder='{ "API_KEY": "your-key", "DB_HOST": "localhost" }'
                  className="font-mono"
                  rows={4}
                />
                <p className="text-sm text-muted-foreground">
                  Enter environment variables as a JSON object, or leave blank if none.
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="enabled"
                  checked={enabled}
                  onCheckedChange={(checked: boolean | 'indeterminate') => setEnabled(Boolean(checked))}
                />
                <Label htmlFor="enabled">Enable this server configuration</Label>
              </div>
            </CardContent>
            <CardFooter className="flex justify-end space-x-2">
              <Link href="/admin/mcp" passHref>
                <Button variant="outline" type="button" disabled={isSubmitting}>
                  Cancel
                </Button>
              </Link>
              <Button type="submit" disabled={isSubmitting || isLoading}>
                {isSubmitting ? 'Saving...' : 'Save Changes'}
              </Button>
            </CardFooter>
          </form>
        </Card>
      )}
    </AdminLayout>
  );
};

export default withAdmin(EditMCPConfigPage);