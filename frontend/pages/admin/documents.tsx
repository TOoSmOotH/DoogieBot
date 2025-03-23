import { useEffect, useState, useRef } from 'react';
import AdminLayout from '../../components/layout/AdminLayout';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Dialog } from '../../components/ui/Dialog';
import { Input } from '../../components/ui/Input';
import ManualDocumentForm from '@/components/document/ManualDocumentForm';
import GitHubRepositoryForm from '@/components/document/GitHubRepositoryForm';
import ZipDocumentForm from '@/components/document/ZipDocumentForm';
import withAdmin from '@/utils/withAdmin';
import { Document, PaginationParams } from '@/types';
import {
  getDocuments,
  getDocument,
  uploadDocument,
  deleteDocument,
  processDocument,
  deleteAllDocumentsAndResetRAG,
  batchProcessDocuments,
} from '@/services/document';

type DocumentTab = 'upload' | 'manual' | 'github' | 'zip';

const DocumentManagement = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDocumentDialog, setShowDocumentDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [documentToEdit, setDocumentToEdit] = useState<{
    id: string;
    title: string;
    content: string;
  } | null>(null);
  const [activeTab, setActiveTab] = useState<DocumentTab>('upload');
  const [uploadTitle, setUploadTitle] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [processingDocumentId, setProcessingDocumentId] = useState<string | null>(null);
  const [isResettingRAG, setIsResettingRAG] = useState(false);
  const [isProcessingAll, setIsProcessingAll] = useState(false);
  const [showResetConfirmDialog, setShowResetConfirmDialog] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [isProcessingSelected, setIsProcessingSelected] = useState(false);
  const [isDeletingSelected, setIsDeletingSelected] = useState(false);
  const [pagination, setPagination] = useState({
    page: 1,
    size: 10,
    total: 0,
    pages: 1,
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filteredDocuments, setFilteredDocuments] = useState<Document[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load documents from the server
  useEffect(() => {
    loadDocuments();
  }, [pagination.page, pagination.size, filterType]);

  // Filter documents client-side based on search term
  useEffect(() => {
    if (!searchTerm.trim()) {
      // If no search term, show all documents
      setFilteredDocuments(documents);
    } else {
      // Filter documents by title (case-insensitive)
      const term = searchTerm.toLowerCase();
      const filtered = documents.filter(
        (doc) =>
          doc.title.toLowerCase().includes(term) ||
          (doc.type && doc.type.toLowerCase().includes(term))
      );
      setFilteredDocuments(filtered);
    }
  }, [documents, searchTerm]);

  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const params: PaginationParams = {
        page: pagination.page,
        size: pagination.size,
      };

      // We'll do client-side filtering for search

      if (filterType && filterType !== 'all') {
        params.doc_type = filterType;
      }

      const { documents, error } = await getDocuments(params);
      if (error) {
        setError(error);
      } else if (documents && documents.items) {
        setDocuments(documents.items);
        setPagination({
          page: documents.page,
          size: documents.size,
          total: documents.total,
          pages: documents.pages,
        });
      } else {
        setDocuments([]);
      }
    } catch (err) {
      setError('Failed to load documents');
      console.error('Failed to load documents:', err);
      setDocuments([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    if (newPage > 0 && newPage <= pagination.pages) {
      setPagination((prev) => ({
        ...prev,
        page: newPage,
      }));
    }
  };

  const handleResetRAG = async () => {
    if (
      confirm(
        'Are you sure you want to delete ALL documents and reset the RAG system? This action cannot be undone.'
      )
    ) {
      setIsResettingRAG(true);
      setError(null);
      try {
        const { success, error } = await deleteAllDocumentsAndResetRAG();
        if (error) {
          setError(error);
        } else if (success) {
          alert('All documents have been deleted and the RAG system has been reset successfully.');
          await loadDocuments();
        }
      } catch (err) {
        setError('Failed to reset RAG system');
        console.error('Failed to reset RAG system:', err);
      } finally {
        setIsResettingRAG(false);
        setShowResetConfirmDialog(false);
      }
    }
  };

  const handleProcessAllDocuments = async () => {
    if (
      confirm(
        'Are you sure you want to process all documents? This may take some time for large document collections.'
      )
    ) {
      setIsProcessingAll(true);
      setError(null);
      try {
        // Call the reprocess-all endpoint which processes all documents in the database
        const response = await fetch('/api/v1/documents/reprocess-all', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
          body: JSON.stringify({
            force_embeddings: true,
            // Explicitly set chunking parameters to ensure documents are properly chunked
            chunk_size: 1000,
            chunk_overlap: 200,
          }),
        });

        const result = await response.json();

        if (!response.ok) {
          setError(result.detail || 'Failed to process all documents');
        } else {
          alert(
            `Processing started for ${result.document_count} documents. This will continue in the background.`
          );
          // After a short delay, reload the documents to show updated status
          setTimeout(() => {
            loadDocuments();
          }, 2000);
        }
      } catch (err) {
        setError('Failed to process all documents');
        console.error('Failed to process all documents:', err);
      } finally {
        setIsProcessingAll(false);
      }
    }
  };

  const handleToggleSelectDocument = (documentId: string) => {
    setSelectedDocuments((prev) => {
      if (prev.includes(documentId)) {
        return prev.filter((id) => id !== documentId);
      } else {
        return [...prev, documentId];
      }
    });
  };

  const handleSelectAll = () => {
    const visibleDocuments = filteredDocuments;
    const allSelected = visibleDocuments.every((doc) => selectedDocuments.includes(doc.id));

    if (allSelected) {
      // If all visible documents are selected, deselect them
      setSelectedDocuments((prev) =>
        prev.filter((id) => !visibleDocuments.some((doc) => doc.id === id))
      );
    } else {
      // Otherwise, select all visible documents
      const visibleIds = visibleDocuments.map((doc) => doc.id);
      setSelectedDocuments((prev) => {
        const newSelection = [...prev];
        visibleIds.forEach((id) => {
          if (!newSelection.includes(id)) {
            newSelection.push(id);
          }
        });
        return newSelection;
      });
    }
  };

  const handlePageSizeChange = (newSize: number) => {
    setPagination((prev) => ({
      ...prev,
      page: 1, // Reset to first page when changing page size
      size: newSize,
    }));
  };

  const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    setPagination((prev) => ({
      ...prev,
      page: 1, // Reset to first page when changing search term
    }));
  };

  const handleFilterTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFilterType(e.target.value);
    setPagination((prev) => ({
      ...prev,
      page: 1, // Reset to first page when changing filter type
    }));
  };

  const handleProcessSelected = async () => {
    if (selectedDocuments.length === 0) {
      alert('Please select at least one document to process');
      return;
    }

    if (
      confirm(`Are you sure you want to process ${selectedDocuments.length} selected document(s)?`)
    ) {
      setIsProcessingSelected(true);
      setError(null);
      try {
        const { result, error } = await batchProcessDocuments(selectedDocuments);
        if (error) {
          setError(error);
        } else if (result) {
          alert(
            `Processing started for ${selectedDocuments.length} document(s). This will continue in the background.`
          );
          // After a short delay, reload the documents to show updated status
          setTimeout(() => {
            loadDocuments();
          }, 2000);
        }
      } catch (err) {
        setError('Failed to process selected documents');
        console.error('Failed to process selected documents:', err);
      } finally {
        setIsProcessingSelected(false);
      }
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedDocuments.length === 0) {
      alert('Please select at least one document to delete');
      return;
    }

    if (
      confirm(
        `Are you sure you want to delete ${selectedDocuments.length} selected document(s)? This action cannot be undone.`
      )
    ) {
      setIsDeletingSelected(true);
      setError(null);
      try {
        // Delete documents one by one
        for (const docId of selectedDocuments) {
          await deleteDocument(docId);
        }
        alert(`Successfully deleted ${selectedDocuments.length} document(s).`);
        setSelectedDocuments([]);
        await loadDocuments();
      } catch (err) {
        setError('Failed to delete selected documents');
        console.error('Failed to delete selected documents:', err);
      } finally {
        setIsDeletingSelected(false);
      }
    }
  };

  const handleUpload = async () => {
    if (!fileInputRef.current?.files?.length) {
      setUploadError('Please select a file to upload');
      return;
    }

    const file = fileInputRef.current.files[0];
    setIsUploading(true);
    setUploadError(null);

    try {
      const { document, error } = await uploadDocument(file, uploadTitle || file.name);
      if (error) {
        setUploadError(error);
      } else {
        setShowDocumentDialog(false);
        setUploadTitle('');
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        await loadDocuments();
      }
    } catch (err) {
      setUploadError('Failed to upload document');
      console.error('Failed to upload document:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (documentId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) {
      return;
    }

    try {
      const { success, error } = await deleteDocument(documentId);
      if (error) {
        setError(error);
      } else if (success) {
        await loadDocuments();
      }
    } catch (err) {
      setError('Failed to delete document');
      console.error('Failed to delete document:', err);
    }
  };

  const handleProcessDocument = async (documentId: string) => {
    setProcessingDocumentId(documentId);
    try {
      const { status, error } = await processDocument(documentId);
      if (error) {
        setError(error);
      } else if (status) {
        alert(`Document processed successfully: ${status.message}`);
        await loadDocuments();
      }
    } catch (err) {
      setError('Failed to process document');
      console.error('Failed to process document:', err);
    } finally {
      setProcessingDocumentId(null);
    }
  };

  const handleManualDocumentSuccess = async () => {
    setShowDocumentDialog(false);
    setShowEditDialog(false);
    setDocumentToEdit(null);
    await loadDocuments();
  };

  const handleEditDocument = async (documentId: string) => {
    try {
      const { document, error } = await getDocument(documentId);
      if (error) {
        setError(error);
        return;
      }

      if (document) {
        // Only allow editing manual documents
        if (document.type === 'manual') {
          setDocumentToEdit({
            id: document.id,
            title: document.title,
            content: document.content || '',
          });
          setShowEditDialog(true);
        } else {
          alert('Only manual documents can be edited.');
        }
      }
    } catch (err) {
      setError('Failed to load document for editing');
      console.error('Failed to load document for editing:', err);
    }
  };

  const getFileTypeIcon = (fileType: string) => {
    switch (fileType.toLowerCase()) {
      case 'pdf':
        return '📄';
      case 'docx':
        return '📝';
      case 'md':
        return '📋';
      case 'txt':
        return '📃';
      case 'json':
      case 'jsonl':
        return '📊';
      case 'manual':
        return '📌';
      default:
        return '📁';
    }
  };

  const renderTabButton = (tab: DocumentTab, label: string) => (
    <button
      className={`px-4 py-2 font-medium text-sm rounded-t-lg ${
        activeTab === tab
          ? 'bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
          : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
      }`}
      onClick={() => setActiveTab(tab)}
    >
      {label}
    </button>
  );

  return (
    <AdminLayout title="Document Management" description="Upload and manage documents for RAG">
      <div className="space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Document Management</h1>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="destructive"
              onClick={handleResetRAG}
              isLoading={isResettingRAG}
              disabled={isResettingRAG || documents.length === 0}
              className="text-sm"
            >
              Reset RAG
            </Button>
            <Button
              variant="secondary"
              onClick={handleProcessAllDocuments}
              isLoading={isProcessingAll}
              disabled={isProcessingAll || documents.length === 0}
              className="text-sm"
            >
              Process All
            </Button>
            <Button
              variant="default"
              onClick={() => setShowDocumentDialog(true)}
              className="text-sm"
            >
              Add Document
            </Button>
          </div>
        </div>

        {error && (
          <div className="p-4 mb-4 text-sm text-red-700 bg-red-100 rounded-lg dark:bg-red-200 dark:text-red-800">
            {error}
          </div>
        )}

        {/* Filter and pagination controls */}
        <div className="flex flex-col md:flex-row gap-4 mb-4">
          <div className="flex-1 flex flex-col sm:flex-row gap-2">
            <div className="flex-1">
              <label
                htmlFor="filter"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
              >
                Search Documents
              </label>
              <Input
                id="filter"
                type="text"
                placeholder="Search by title..."
                value={searchTerm}
                onChange={handleFilterChange}
                className="w-full"
              />
            </div>
            <div className="w-full sm:w-40">
              <label
                htmlFor="filterType"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
              >
                Document Type
              </label>
              <select
                id="filterType"
                value={filterType}
                onChange={handleFilterTypeChange}
                className="w-full h-10 px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="all">All Types</option>
                <option value="pdf">PDF</option>
                <option value="docx">DOCX</option>
                <option value="md">Markdown</option>
                <option value="txt">Text</option>
                <option value="json">JSON</option>
                <option value="yaml">YAML</option>
                <option value="yml">YML</option>
                <option value="manual">Manual</option>
                <option value="github">GitHub</option>
              </select>
            </div>
            <div className="w-full sm:w-40">
              <label
                htmlFor="pageSize"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
              >
                Items Per Page
              </label>
              <select
                id="pageSize"
                value={pagination.size}
                onChange={(e) => handlePageSizeChange(Number(e.target.value))}
                className="w-full h-10 px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="10">10</option>
                <option value="25">25</option>
                <option value="50">50</option>
                <option value="100">100</option>
                <option value="1000">1000</option>
              </select>
            </div>
          </div>
        </div>

        {/* Batch action buttons */}
        {documents && documents.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            <Button
              variant="outline"
              onClick={handleSelectAll}
              className="text-sm"
              disabled={isLoading}
            >
              {filteredDocuments.length > 0 &&
              filteredDocuments.every((doc) => selectedDocuments.includes(doc.id))
                ? 'Deselect All'
                : 'Select All'}
            </Button>
            <Button
              variant="outline"
              onClick={handleProcessSelected}
              isLoading={isProcessingSelected}
              disabled={isProcessingSelected || selectedDocuments.length === 0}
              className="text-sm"
            >
              Process Selected
            </Button>
            <Button
              variant="outline"
              className="text-red-500 hover:text-red-700 text-sm"
              onClick={handleDeleteSelected}
              isLoading={isDeletingSelected}
              disabled={isDeletingSelected || selectedDocuments.length === 0}
            >
              Delete Selected
            </Button>
            <div className="ml-auto text-sm text-gray-500 flex items-center">
              {selectedDocuments.length > 0 && (
                <span>{selectedDocuments.length} document(s) selected</span>
              )}
            </div>
          </div>
        )}

        {isLoading ? (
          <div className="text-center py-8">Loading...</div>
        ) : documents && documents.length > 0 ? (
          <>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-gray-50 dark:bg-gray-800 text-left">
                    <th className="p-3 border-b border-gray-200 dark:border-gray-700">
                      <span className="sr-only">Select</span>
                    </th>
                    <th className="p-3 border-b border-gray-200 dark:border-gray-700">Document</th>
                    <th className="p-3 border-b border-gray-200 dark:border-gray-700">Type</th>
                    <th className="p-3 border-b border-gray-200 dark:border-gray-700">Uploaded</th>
                    <th className="p-3 border-b border-gray-200 dark:border-gray-700">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDocuments.map((doc) => (
                    <tr
                      key={doc.id}
                      className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <td className="p-3">
                        <input
                          type="checkbox"
                          checked={selectedDocuments.includes(doc.id)}
                          onChange={() => handleToggleSelectDocument(doc.id)}
                          className="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                        />
                      </td>
                      <td className="p-3">
                        <div className="flex items-center">
                          <span className="text-xl mr-2">{getFileTypeIcon(doc.type)}</span>
                          <span
                            className="font-medium text-gray-900 dark:text-white"
                            title={doc.title}
                          >
                            {doc.title}
                          </span>
                        </div>
                      </td>
                      <td className="p-3 text-gray-500 dark:text-gray-400">{doc.type}</td>
                      <td className="p-3 text-gray-500 dark:text-gray-400">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </td>
                      <td className="p-3">
                        <div className="flex space-x-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleProcessDocument(doc.id)}
                            isLoading={processingDocumentId === doc.id}
                            disabled={processingDocumentId !== null}
                            className="text-xs"
                          >
                            {processingDocumentId === doc.id ? 'Processing...' : 'Process'}
                          </Button>

                          {doc.type === 'manual' && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleEditDocument(doc.id)}
                              className="text-blue-500 hover:text-blue-700 text-xs"
                            >
                              Edit
                            </Button>
                          )}

                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDelete(doc.id)}
                            className="text-red-500 hover:text-red-700 text-xs"
                          >
                            Delete
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {pagination.pages > 1 && (
              <div className="flex justify-center mt-6">
                <nav className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(pagination.page - 1)}
                    disabled={pagination.page === 1 || isLoading}
                  >
                    Previous
                  </Button>

                  <div className="flex items-center space-x-1 overflow-x-auto max-w-md">
                    {(() => {
                      // Logic to show limited page numbers with ellipses
                      const visiblePageNumbers = [];
                      const totalPages = pagination.pages;
                      const currentPage = pagination.page;

                      // Always show first page
                      visiblePageNumbers.push(1);

                      // Calculate range of pages to show around current page
                      const delta = 2; // Number of pages to show on each side of current page
                      const leftBound = Math.max(2, currentPage - delta);
                      const rightBound = Math.min(totalPages - 1, currentPage + delta);

                      // Add ellipsis after first page if needed
                      if (leftBound > 2) {
                        visiblePageNumbers.push('ellipsis-left');
                      }

                      // Add pages around current page
                      for (let i = leftBound; i <= rightBound; i++) {
                        visiblePageNumbers.push(i);
                      }

                      // Add ellipsis before last page if needed
                      if (rightBound < totalPages - 1) {
                        visiblePageNumbers.push('ellipsis-right');
                      }

                      // Always show last page if there is more than one page
                      if (totalPages > 1) {
                        visiblePageNumbers.push(totalPages);
                      }

                      // Render the page buttons
                      return visiblePageNumbers.map((page, index) => {
                        if (page === 'ellipsis-left' || page === 'ellipsis-right') {
                          return (
                            <span
                              key={page}
                              className="w-8 h-8 flex items-center justify-center text-gray-500"
                            >
                              ...
                            </span>
                          );
                        }

                        return (
                          <Button
                            key={`page-${page}`}
                            variant={page === currentPage ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => handlePageChange(page as number)}
                            disabled={isLoading}
                            className="w-8 h-8 p-0 flex-shrink-0"
                          >
                            {page}
                          </Button>
                        );
                      });
                    })()}
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(pagination.page + 1)}
                    disabled={pagination.page === pagination.pages || isLoading}
                  >
                    Next
                  </Button>
                </nav>
              </div>
            )}

            <div className="text-center text-sm text-gray-500 mt-2">
              {searchTerm
                ? `Showing ${filteredDocuments.length} of ${documents.length} documents (filtered from ${pagination.total} total)`
                : `Showing ${documents.length} of ${pagination.total} documents`}
            </div>
          </>
        ) : (
          <div className="text-center py-8 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-gray-600 dark:text-gray-400">No documents available</p>
            <p className="text-sm mt-2">
              Upload documents or add manual content to use in the RAG system
            </p>
          </div>
        )}
      </div>

      {/* Add Document Dialog */}
      <Dialog
        isOpen={showDocumentDialog}
        onClose={() => setShowDocumentDialog(false)}
        title="Add Document"
      >
        <div className="space-y-4">
          <div className="flex border-b border-gray-200 dark:border-gray-700">
            {renderTabButton('upload', 'Upload File')}
            {renderTabButton('zip', 'Upload ZIP')}
            {renderTabButton('manual', 'Manual Entry')}
            {renderTabButton('github', 'GitHub')}
          </div>

          {activeTab === 'upload' ? (
            <div className="space-y-4 pt-2">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Upload a document to be processed for the RAG system. Supported formats: PDF, DOCX,
                Markdown, TXT, JSON, JSONL, YAML, YML.
              </p>

              <div className="space-y-4">
                <div>
                  <label
                    htmlFor="title"
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                  >
                    Document Title (optional)
                  </label>
                  <Input
                    id="title"
                    type="text"
                    placeholder="Enter document title"
                    value={uploadTitle}
                    onChange={(e) => setUploadTitle(e.target.value)}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    If not provided, the filename will be used
                  </p>
                </div>

                <div>
                  <label
                    htmlFor="file"
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                  >
                    Document File
                  </label>
                  <input
                    id="file"
                    type="file"
                    ref={fileInputRef}
                    className="block w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 cursor-pointer dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
                    accept=".pdf,.docx,.md,.txt,.json,.jsonl,.yaml,.yml"
                  />
                </div>
              </div>

              {uploadError && (
                <div className="p-3 text-sm text-red-700 bg-red-100 rounded-lg dark:bg-red-200 dark:text-red-800">
                  {uploadError}
                </div>
              )}

              <div className="flex justify-end space-x-2 mt-4">
                <Button
                  variant="outline"
                  onClick={() => setShowDocumentDialog(false)}
                  disabled={isUploading}
                >
                  Cancel
                </Button>
                <Button variant="default" onClick={handleUpload} isLoading={isUploading}>
                  {isUploading ? 'Uploading...' : 'Upload'}
                </Button>
              </div>
            </div>
          ) : activeTab === 'zip' ? (
            <div className="pt-2">
              <ZipDocumentForm
                onSuccess={() => {
                  setShowDocumentDialog(false);
                  loadDocuments();
                }}
                onError={(error) => {
                  setError(typeof error === 'string' ? error : 'Failed to upload ZIP file');
                }}
              />
            </div>
          ) : activeTab === 'manual' ? (
            <div className="pt-2">
              <ManualDocumentForm
                onSuccess={handleManualDocumentSuccess}
                onCancel={() => setShowDocumentDialog(false)}
              />
            </div>
          ) : (
            <div className="pt-2">
              <GitHubRepositoryForm
                onSuccess={handleManualDocumentSuccess}
                onCancel={() => setShowDocumentDialog(false)}
              />
            </div>
          )}
        </div>
      </Dialog>

      {/* Edit Document Dialog */}
      <Dialog
        isOpen={showEditDialog}
        onClose={() => {
          setShowEditDialog(false);
          setDocumentToEdit(null);
        }}
        title="Edit Document"
      >
        <div className="pt-2">
          <ManualDocumentForm
            onSuccess={handleManualDocumentSuccess}
            onCancel={() => {
              setShowEditDialog(false);
              setDocumentToEdit(null);
            }}
            documentToEdit={documentToEdit}
          />
        </div>
      </Dialog>
    </AdminLayout>
  );
};

export default withAdmin(DocumentManagement, 'Document Management');
