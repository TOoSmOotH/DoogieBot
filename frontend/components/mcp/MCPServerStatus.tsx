import React, { useState, useEffect, useCallback } from 'react';
import {
  getMcpConfigStatus,
  startMcpServer,
  stopMcpServer,
  restartMcpServer,
  deleteMcpConfig,
  MCPServerStatus as StatusType,
} from '@/services/mcp';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { useErrorHandler } from '@/hooks/useErrorHandler';
import { Badge } from '@/components/ui/badge'; // Assuming shadcn/ui badge component
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui'; // Import from index
import { Trash2, Play, StopCircle, RefreshCw } from 'lucide-react'; // Assuming package will be installed
import ConfirmDialog from '@/components/ui/ConfirmDialog'; // Assuming named export

interface MCPServerStatusProps {
  configId: string;
  configName: string;
  onDeleteSuccess?: (id: string) => void; // Callback after successful deletion
}

const MCPServerStatus: React.FC<MCPServerStatusProps> = ({ configId, configName, onDeleteSuccess }) => {
  const [status, setStatus] = useState<StatusType | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isActionLoading, setIsActionLoading] = useState<boolean>(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<boolean>(false);
  const { handleError } = useErrorHandler();

  const fetchStatus = useCallback(async () => {
    // Don't set isLoading to true here to avoid flickering during polling/refresh
    try {
      const currentStatus = await getMcpConfigStatus(configId);
      setStatus(currentStatus);
    } catch (error) {
      handleError(error, `Failed to fetch status for ${configName}.`);
      setStatus(null); // Reset status on error
    } finally {
      setIsLoading(false); // Only set loading false on initial load or error
    }
  }, [configId, configName, handleError]);

  useEffect(() => {
    setIsLoading(true); // Set loading true only on initial mount
    fetchStatus();
    // Optional: Implement polling to refresh status periodically
    // const intervalId = setInterval(fetchStatus, 15000); // Poll every 15 seconds
    // return () => clearInterval(intervalId);
  }, [fetchStatus]); // fetchStatus is stable due to useCallback

  const handleAction = async (action: () => Promise<StatusType>, actionName: string) => {
    setIsActionLoading(true);
    try {
      const newStatus = await action();
      setStatus(newStatus);
    } catch (error) {
      handleError(error, `Failed to ${actionName} server ${configName}.`);
      // Optionally refetch status on error after a delay
      setTimeout(fetchStatus, 1000);
    } finally {
      setIsActionLoading(false);
    }
  };

  const handleStart = () => handleAction(() => startMcpServer(configId), 'start');
  const handleStop = () => handleAction(() => stopMcpServer(configId), 'stop');
  const handleRestart = () => handleAction(() => restartMcpServer(configId), 'restart');

  const handleDelete = async () => {
    setIsActionLoading(true);
    try {
      await deleteMcpConfig(configId);
      // Optionally notify parent component of successful deletion
      if (onDeleteSuccess) {
        onDeleteSuccess(configId);
      }
      // No need to update status locally as the component might be unmounted
    } catch (error) {
      handleError(error, `Failed to delete server ${configName}.`);
      setIsActionLoading(false); // Ensure loading state is reset on error
    }
    // No finally block needed here as component might unmount
  };

  const getStatusBadgeVariant = (statusString: string | undefined): 'default' | 'destructive' | 'secondary' | 'outline' => {
    switch (statusString) {
      case 'running':
        return 'default'; // Typically green or primary color
      case 'stopped':
        return 'secondary'; // Gray or muted
      case 'error':
        return 'destructive'; // Red
      default:
        return 'outline'; // Default for unknown/loading
    }
  };

  return (
    <TooltipProvider delayDuration={100}>
      <div className="flex items-center space-x-2">
        {isLoading ? (
          // Corrected: Removed invalid size prop
          <Spinner />
        ) : status ? (
          <Tooltip>
            <TooltipTrigger>
              <Badge variant={getStatusBadgeVariant(status.status)}>
                {status.status}
              </Badge>
            </TooltipTrigger>
            {status.status === 'error' && status.error_message && (
              <TooltipContent>
                <p className="max-w-xs break-words">Error: {status.error_message}</p>
              </TooltipContent>
            )}
             {status.status === 'running' && status.container_id && (
              <TooltipContent>
                <p>Container ID: {status.container_id.substring(0, 12)}</p>
              </TooltipContent>
            )}
          </Tooltip>
        ) : (
           <Tooltip>
            <TooltipTrigger>
              <Badge variant="outline">Unknown</Badge>
            </TooltipTrigger>
             <TooltipContent>
                <p>Could not fetch status.</p>
              </TooltipContent>
          </Tooltip>
        )}

        <Button
          variant="ghost"
          size="icon"
          onClick={handleStart}
          disabled={isLoading || isActionLoading || status?.status === 'running'}
          aria-label={`Start ${configName}`}
        >
          <Play className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleStop}
          disabled={isLoading || isActionLoading || status?.status !== 'running'}
          aria-label={`Stop ${configName}`}
        >
          <StopCircle className="h-4 w-4" />
        </Button>
         <Button
          variant="ghost"
          size="icon"
          onClick={handleRestart}
          disabled={isLoading || isActionLoading || status?.status !== 'running'}
          aria-label={`Restart ${configName}`}
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setShowDeleteConfirm(true)}
          disabled={isLoading || isActionLoading}
          className="text-red-500 hover:text-red-700"
          aria-label={`Delete ${configName}`}
        >
          <Trash2 className="h-4 w-4" />
        </Button>

        <ConfirmDialog
            isOpen={showDeleteConfirm}
            onClose={() => setShowDeleteConfirm(false)}
            onConfirm={handleDelete}
            title={`Delete ${configName}?`}
            message="Are you sure you want to delete this MCP server configuration? This action cannot be undone."
            confirmText="Delete"
            // isConfirming={isActionLoading} // Removed invalid prop
        />
      </div>
    </TooltipProvider>
  );
};

export default MCPServerStatus;