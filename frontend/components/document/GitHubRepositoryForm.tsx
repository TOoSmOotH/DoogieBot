import { useState } from 'react';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { importGitHubRepository } from '@/services/document';

interface GitHubRepositoryFormProps {
  onSuccess?: () => void;
  onCancel?: () => void;
}

const GitHubRepositoryForm = ({ onSuccess, onCancel }: GitHubRepositoryFormProps) => {
  const [repoUrl, setRepoUrl] = useState('');
  const [branch, setBranch] = useState('main');
  const [fileTypes, setFileTypes] = useState('rst,txt');
  const [useBackgroundProcessing, setUseBackgroundProcessing] = useState(true);
  const [refreshRepo, setRefreshRepo] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!repoUrl.trim()) {
      setError('Repository URL is required');
      return;
    }

    // Validate GitHub URL format - more flexible pattern
    const githubUrlPattern = /^https?:\/\/github\.com\/[^\/]+\/[^\/]+\/?(?:\?.*)?$/;
    if (!githubUrlPattern.test(repoUrl)) {
      setError(
        'Invalid GitHub repository URL. Format should be: https://github.com/username/repository'
      );
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const { success, error, message, imported_count } = await importGitHubRepository(
        repoUrl,
        branch,
        fileTypes,
        useBackgroundProcessing,
        refreshRepo
      );

      if (error) {
        setError(error);
      } else if (success) {
        const action = refreshRepo ? 'refreshed' : 'imported';
        if (useBackgroundProcessing) {
          setSuccessMessage(
            message ||
              `GitHub repository ${action} in the background. Files will be processed automatically.`
          );
        } else {
          setSuccessMessage(
            message ||
              `Successfully ${action} ${imported_count || 'multiple'} documents from GitHub repository`
          );
        }

        // Wait 2 seconds before closing the dialog to show the success message
        setTimeout(() => {
          setRepoUrl('');
          setBranch('main');
          if (onSuccess) {
            onSuccess();
          }
        }, 2000);
      }
    } catch (err: any) {
      // Try to extract a more detailed error message if available
      const errorMessage =
        err?.response?.data?.detail || err?.message || 'Failed to import GitHub repository';
      setError(errorMessage);
      console.error('Failed to import GitHub repository:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label
          htmlFor="repoUrl"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          GitHub Repository URL
        </label>
        <Input
          id="repoUrl"
          type="text"
          placeholder="https://github.com/username/repository"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
          required
        />
        <p className="text-xs text-gray-500 mt-1">
          Example: https://github.com/username/repository
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Make sure to use the exact repository URL format as shown above. The repository must exist
          and be accessible.
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Note: For private repositories, a GitHub API token must be configured on the server.
        </p>
      </div>

      <div>
        <label
          htmlFor="branch"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          Branch
        </label>
        <Input
          id="branch"
          type="text"
          placeholder="main"
          value={branch}
          onChange={(e) => setBranch(e.target.value)}
        />
        <p className="text-xs text-gray-500 mt-1">Default: main</p>
      </div>

      <div>
        <label
          htmlFor="fileTypes"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          File Types
        </label>
        <Input
          id="fileTypes"
          type="text"
          placeholder="rst,txt,md"
          value={fileTypes}
          onChange={(e) => setFileTypes(e.target.value)}
        />
        <p className="text-xs text-gray-500 mt-1">
          Comma-separated list of file extensions to import (e.g., rst,txt,md)
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center">
          <input
            id="backgroundProcessing"
            type="checkbox"
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            checked={useBackgroundProcessing}
            onChange={(e) => setUseBackgroundProcessing(e.target.checked)}
          />
          <label
            htmlFor="backgroundProcessing"
            className="ml-2 block text-sm text-gray-700 dark:text-gray-300"
          >
            Process in background (recommended for large repositories)
          </label>
        </div>

        <div className="flex items-center">
          <input
            id="refreshRepo"
            type="checkbox"
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            checked={refreshRepo}
            onChange={(e) => setRefreshRepo(e.target.checked)}
          />
          <label
            htmlFor="refreshRepo"
            className="ml-2 block text-sm text-gray-700 dark:text-gray-300"
          >
            Refresh repository (delete existing documents from this repository first)
          </label>
        </div>
      </div>

      <div className="p-3 text-sm text-blue-700 bg-blue-100 rounded-lg dark:bg-blue-200 dark:text-blue-800">
        <p className="font-medium">Tips:</p>
        <ul className="list-disc pl-5 mt-1 space-y-1">
          <li>Enable "Process in background" for better reliability with large repositories</li>
          <li>
            Use "Refresh repository" to update an existing repository instead of creating duplicates
          </li>
          <li>The repository must be public and accessible</li>
          <li>Make sure to specify the correct branch name</li>
          <li>All matching files in the repository will be imported</li>
        </ul>
      </div>

      {error && (
        <div className="p-3 text-sm text-red-700 bg-red-100 rounded-lg dark:bg-red-200 dark:text-red-800">
          {error}
        </div>
      )}

      {successMessage && (
        <div className="p-3 text-sm text-green-700 bg-green-100 rounded-lg dark:bg-green-200 dark:text-green-800">
          {successMessage}
        </div>
      )}

      <div className="flex justify-end space-x-2">
        {onCancel && (
          <Button type="button" variant="outline" onClick={onCancel} disabled={isSubmitting}>
            Cancel
          </Button>
        )}
        <Button type="submit" variant="default" isLoading={isSubmitting}>
          {isSubmitting ? 'Importing...' : 'Import Repository'}
        </Button>
      </div>
    </form>
  );
};

export default GitHubRepositoryForm;
