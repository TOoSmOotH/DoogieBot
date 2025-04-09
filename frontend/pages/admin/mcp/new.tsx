import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { createMcpConfig, MCPServerConfigCreate } from '@/services/mcp';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Label } from '@/components/ui/label'; // Assuming shadcn/ui label component
import { TextArea } from '@/components/ui/TextArea'; // Corrected casing
import { Checkbox } from '@/components/ui/checkbox'; // Assuming shadcn/ui checkbox component
import { useErrorHandler } from '@/hooks/useErrorHandler';
import withAdmin from '@/utils/withAdmin';
import Link from 'next/link';
import AdminLayout from '@/components/layout/AdminLayout'; // Import AdminLayout

const NewMCPConfigPage: React.FC = () => {
  const router = useRouter();
  const { handleError } = useErrorHandler();
  const [name, setName] = useState('');
  const [args, setArgs] = useState(''); // Store args as a single string for easier editing
  const [envVars, setEnvVars] = useState(''); // Store env vars as JSON string
  const [enabled, setEnabled] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const parseArgs = (argsString: string): string[] => {
    // Basic split by space, might need more robust parsing for quoted args later
    return argsString.trim().split(/\s+/);
  };

  const parseEnvVars = (envString: string): Record<string, string> | null => {
    if (!envString.trim()) {
      return {}; // Return empty object if string is empty
    }
    try {
      const parsed = JSON.parse(envString);
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        // Basic validation: ensure it's a non-array object
        // Further validation could check if values are strings
        return parsed;
      }
      throw new Error('Invalid JSON format for environment variables.');
    } catch (error) {
      console.error("Error parsing ENV JSON:", error);
      return null; // Indicate parsing error
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
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

    const configData: MCPServerConfigCreate = {
      name,
      command: 'docker', // Hardcoded as per requirement
      args: parsedArgs,
      env: Object.keys(parsedEnv).length > 0 ? parsedEnv : undefined, // Only include env if not empty
      enabled,
    };

    try {
      await createMcpConfig(configData);
      router.push('/admin/mcp'); // Redirect to dashboard on success
    } catch (error) {
      handleError(error, 'Failed to create MCP configuration.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    // Wrap content with AdminLayout
    <AdminLayout title="Add New MCP Server">
      {/* Removed outer div and h1 */}
      <Card>
        <form onSubmit={handleSubmit}>
          <CardHeader>
            <CardTitle>Server Configuration</CardTitle>
            <CardDescription>
              Configure the details for the new MCP server. Remember, all servers run in Docker.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)} // Added type
                required
                placeholder="e.g., filesystem-local"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="args">Docker Arguments</Label>
              <Input
                id="args"
                value={args}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setArgs(e.target.value)} // Added type
                required
                placeholder="run -i --rm mcp/filesystem /data"
                className="font-mono"
              />
              <p className="text-sm text-muted-foreground">
                Enter the arguments for the 'docker' command (e.g., 'run -i --rm image-name /path').
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="envVars">Environment Variables (JSON)</Label>
              <TextArea
                id="envVars"
                value={envVars}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setEnvVars(e.target.value)} // Added type
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
                // Type for 'checked' in shadcn Checkbox is often boolean | 'indeterminate'
                onCheckedChange={(checked: boolean | 'indeterminate') => setEnabled(Boolean(checked))} // Added type
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
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Creating...' : 'Create Server'}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </AdminLayout>
  );
};

export default withAdmin(NewMCPConfigPage);