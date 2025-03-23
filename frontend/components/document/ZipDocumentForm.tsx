import React, { useState } from 'react';
import { uploadZipFile } from '../../services/document';
import { Button } from '../ui/Button';

interface ZipDocumentFormProps {
  onSuccess?: () => void;
  onError?: (error: any) => void;
}

const ZipDocumentForm: React.FC<ZipDocumentFormProps> = ({ onSuccess, onError }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [processingStatus, setProcessingStatus] = useState<
    'idle' | 'uploading' | 'processing' | 'complete' | 'error'
  >('idle');
  const [generateEmbeddings, setGenerateEmbeddings] = useState(true);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      // Check if the file is a zip file
      if (
        selectedFile.type === 'application/zip' ||
        selectedFile.name.toLowerCase().endsWith('.zip')
      ) {
        setFile(selectedFile);
      } else {
        alert('Please select a valid ZIP file');
        e.target.value = '';
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      alert('Please select a ZIP file to upload');
      return;
    }

    try {
      // Update UI state
      setIsUploading(true);
      setProcessingStatus('uploading');
      setProgress('Uploading ZIP file...');

      // Upload the file
      const result = await uploadZipFile(file, generateEmbeddings);

      // Update UI with success
      setProcessingStatus('processing');
      setProgress(
        'ZIP file uploaded successfully. Documents are being processed in the background. ' +
          'This may take some time depending on the number and size of documents. ' +
          'You can continue using the application while processing occurs.'
      );
      setFile(null);

      // Set a timeout to change status to complete after a delay
      setTimeout(() => {
        setProcessingStatus('complete');
        // If onSuccess callback is provided, call it
        if (onSuccess) {
          onSuccess();
        }
      }, 5000);
    } catch (error) {
      // Handle errors
      console.error('Error uploading ZIP file:', error);
      setProcessingStatus('error');

      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setProgress(
        `Error uploading ZIP file: ${errorMessage}. Please try again with a smaller file or fewer documents.`
      );

      if (onError) {
        onError(error);
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">
        Upload ZIP Archive
      </h2>
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
        Upload a ZIP file containing multiple documents. Each document will be processed
        individually. Supported formats: PDF, DOCX, Markdown, RST, TXT, JSON, JSONL, YAML.
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            ZIP File
          </label>
          <input
            type="file"
            accept=".zip"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 dark:text-gray-300
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-md file:border-0
                      file:text-sm file:font-semibold
                      file:bg-blue-50 file:text-blue-700
                      hover:file:bg-blue-100
                      dark:file:bg-gray-700 dark:file:text-gray-200"
            disabled={isUploading}
          />
          {file && (
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Selected: {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
            </p>
          )}
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            id="generateEmbeddings"
            checked={generateEmbeddings}
            onChange={(e) => setGenerateEmbeddings(e.target.checked)}
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            disabled={isUploading}
          />
          <label
            htmlFor="generateEmbeddings"
            className="ml-2 block text-sm text-gray-700 dark:text-gray-300"
          >
            Generate embeddings (required for semantic search)
          </label>
        </div>

        <div className="flex justify-end">
          <Button type="submit" disabled={!file || isUploading} isLoading={isUploading}>
            {isUploading ? 'Uploading...' : 'Upload ZIP'}
          </Button>
        </div>

        {progress && (
          <div
            className={`mt-4 p-3 rounded-md ${
              processingStatus === 'error'
                ? 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                : processingStatus === 'complete'
                  ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                  : 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
            }`}
          >
            <div className="flex items-center">
              {processingStatus === 'uploading' || processingStatus === 'processing' ? (
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-500"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              ) : processingStatus === 'complete' ? (
                <svg
                  className="-ml-1 mr-3 h-5 w-5 text-green-500"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : processingStatus === 'error' ? (
                <svg
                  className="-ml-1 mr-3 h-5 w-5 text-red-500"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : null}
              <span>{progress}</span>
            </div>
          </div>
        )}
      </form>
    </div>
  );
};

export default ZipDocumentForm;
