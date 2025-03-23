import { useEffect, useState } from 'react';
import Link from 'next/link';
import AdminLayout from '../../components/layout/AdminLayout';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Dialog } from '../../components/ui/Dialog';
import withAdmin from '@/utils/withAdmin';
import { RAGStats, RAGComponentStatus, RAGBuildOptions } from '@/types';
import { getRAGStatus, buildIndexes, toggleRAGComponent, deleteAllChunks } from '@/services/rag';

const RAGManagement = () => {
  const [stats, setStats] = useState<RAGStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRebuildDialog, setShowRebuildDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [rebuildOptions, setRebuildOptions] = useState<RAGBuildOptions>({
    rebuild: false,
    use_bm25: true,
    use_faiss: true,
    use_graph: true,
  });

  useEffect(() => {
    loadRAGStatus();
  }, []);

  const loadRAGStatus = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const { stats, error } = await getRAGStatus();
      if (error) {
        setError(error);
      } else if (stats) {
        setStats(stats);
      }
    } catch (err) {
      setError('Failed to load RAG status');
      console.error('Failed to load RAG status:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleComponent = async (
    component: 'bm25' | 'faiss' | 'graph',
    currentEnabled: boolean
  ) => {
    try {
      const { error } = await toggleRAGComponent(component, !currentEnabled);
      if (error) {
        setError(error);
      } else {
        await loadRAGStatus();
      }
    } catch (err) {
      setError('Failed to toggle component');
      console.error('Failed to toggle component:', err);
    }
  };

  const handleRebuild = async () => {
    setIsIndexing(true);
    setError(null);
    try {
      const { result, error } = await buildIndexes(rebuildOptions);
      if (error) {
        setError(error);
      } else {
        setShowRebuildDialog(false);
        await loadRAGStatus();
      }
    } catch (err) {
      setError('Failed to rebuild indexes');
      console.error('Failed to rebuild indexes:', err);
    } finally {
      setIsIndexing(false);
    }
  };

  const renderComponentStatus = (
    name: string,
    status: RAGComponentStatus,
    component: 'bm25' | 'faiss' | 'graph'
  ) => (
    <Card key={name}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{name}</h3>
          <Button
            variant={status.enabled ? 'default' : 'outline'}
            onClick={() => handleToggleComponent(component, status.enabled)}
            disabled={isIndexing}
          >
            {status.enabled ? 'Enabled' : 'Disabled'}
          </Button>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Status:</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {status.status === 'indexing' ? 'Indexing...' : status.status}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Indexed Documents:</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {status.document_count}
            </span>
          </div>
          {component === 'graph' && status.node_count !== undefined && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">Graph Nodes:</span>
              <span className="font-medium text-gray-900 dark:text-white">{status.node_count}</span>
            </div>
          )}
          {component === 'graph' && status.edge_count !== undefined && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">Graph Edges:</span>
              <span className="font-medium text-gray-900 dark:text-white">{status.edge_count}</span>
            </div>
          )}
          {status.last_indexed && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">Last Indexed:</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {new Date(status.last_indexed).toLocaleString()}
              </span>
            </div>
          )}
          {status.error_message && (
            <div className="text-sm text-red-500">Error: {status.error_message}</div>
          )}
        </div>
      </div>
    </Card>
  );

  return (
    <AdminLayout title="RAG Management" description="Manage Retrieval-Augmented Generation systems">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">RAG Management</h1>
          <div className="flex space-x-2">
            <Link href="/admin/documents">
              <Button variant="outline">Manage Documents</Button>
            </Link>
            <div className="flex space-x-2">
              <Button
                variant="destructive"
                onClick={() => setShowDeleteDialog(true)}
                disabled={isDeleting || isIndexing}
              >
                Delete All Chunks
              </Button>
              <Button
                variant="default"
                onClick={() => setShowRebuildDialog(true)}
                disabled={isIndexing || isDeleting}
              >
                Rebuild Indexes
              </Button>
            </div>
          </div>
        </div>

        {error && (
          <div className="p-4 mb-4 text-sm text-red-700 bg-red-100 rounded-lg dark:bg-red-200 dark:text-red-800">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="text-center py-8">Loading...</div>
        ) : stats ? (
          <div className="grid gap-4">
            {renderComponentStatus('BM25 Index', stats.bm25_status, 'bm25')}
            {renderComponentStatus('FAISS Vector Store', stats.faiss_status, 'faiss')}
            {renderComponentStatus('Graph RAG', stats.graph_status, 'graph')}

            <Card>
              <div className="p-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Overall Statistics
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Total Documents:</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {stats.document_count}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Total Chunks:</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {stats.chunk_count}
                    </span>
                  </div>
                  {stats.last_updated && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Last Updated:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {new Date(stats.last_updated).toLocaleString()}
                      </span>
                    </div>
                  )}
                </div>
                <div className="mt-4">
                  <Link href="/admin/documents">
                    <Button variant="outline" className="w-full">
                      View and Upload Documents
                    </Button>
                  </Link>
                </div>
              </div>
            </Card>
          </div>
        ) : (
          <div className="text-center py-8">No RAG statistics available</div>
        )}
      </div>

      <Dialog
        isOpen={showRebuildDialog}
        onClose={() => setShowRebuildDialog(false)}
        title="Rebuild RAG Indexes"
      >
        <div className="space-y-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Select which indexes to rebuild and whether to rebuild from scratch or update existing
            indexes.
          </p>

          <div className="space-y-2">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="rebuild"
                checked={rebuildOptions.rebuild}
                onChange={(e) =>
                  setRebuildOptions({ ...rebuildOptions, rebuild: e.target.checked })
                }
                className="mr-2"
              />
              <label htmlFor="rebuild">Rebuild from scratch (slower but more thorough)</label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="use_bm25"
                checked={rebuildOptions.use_bm25}
                onChange={(e) =>
                  setRebuildOptions({ ...rebuildOptions, use_bm25: e.target.checked })
                }
                className="mr-2"
              />
              <label htmlFor="use_bm25">Include BM25 Index</label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="use_faiss"
                checked={rebuildOptions.use_faiss}
                onChange={(e) =>
                  setRebuildOptions({ ...rebuildOptions, use_faiss: e.target.checked })
                }
                className="mr-2"
              />
              <label htmlFor="use_faiss">Include FAISS Vector Store</label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="use_graph"
                checked={rebuildOptions.use_graph}
                onChange={(e) =>
                  setRebuildOptions({ ...rebuildOptions, use_graph: e.target.checked })
                }
                className="mr-2"
              />
              <label htmlFor="use_graph">Include Graph RAG</label>
            </div>
          </div>

          <div className="flex justify-end space-x-2 mt-4">
            <Button
              variant="outline"
              onClick={() => setShowRebuildDialog(false)}
              disabled={isIndexing}
            >
              Cancel
            </Button>
            <Button
              variant="default"
              onClick={handleRebuild}
              isLoading={isIndexing}
              disabled={
                !rebuildOptions.use_bm25 && !rebuildOptions.use_faiss && !rebuildOptions.use_graph
              }
            >
              {isIndexing ? 'Rebuilding...' : 'Rebuild'}
            </Button>
          </div>
        </div>
      </Dialog>

      <Dialog
        isOpen={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        title="Delete All Document Chunks"
      >
        <div className="space-y-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Are you sure you want to delete all document chunks? This will remove all processed
            chunks from the database. You will need to rebuild indexes after this operation.
          </p>

          <div className="flex justify-end space-x-2 mt-4">
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={async () => {
                setIsDeleting(true);
                setError(null);
                try {
                  const { error } = await deleteAllChunks();
                  if (error) {
                    setError(error);
                  } else {
                    setShowDeleteDialog(false);
                    await loadRAGStatus();
                  }
                } catch (err) {
                  setError('Failed to delete chunks');
                  console.error('Failed to delete chunks:', err);
                } finally {
                  setIsDeleting(false);
                }
              }}
              isLoading={isDeleting}
            >
              {isDeleting ? 'Deleting...' : 'Delete All Chunks'}
            </Button>
          </div>
        </div>
      </Dialog>
    </AdminLayout>
  );
};

export default withAdmin(RAGManagement, 'RAG Management');
