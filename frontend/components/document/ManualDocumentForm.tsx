import { useState, useEffect } from 'react';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { createManualDocument, updateDocumentContent } from '@/services/document';

interface ManualDocumentFormProps {
  onSuccess?: () => void;
  onCancel?: () => void;
  documentToEdit?: {
    id: string;
    title: string;
    content: string;
  } | null;
}

const ManualDocumentForm = ({ onSuccess, onCancel, documentToEdit }: ManualDocumentFormProps) => {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isEditing = !!documentToEdit;

  useEffect(() => {
    if (documentToEdit) {
      setTitle(documentToEdit.title);
      setContent(documentToEdit.content);
    }
  }, [documentToEdit]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!title.trim()) {
      setError('Title is required');
      return;
    }

    if (!content.trim()) {
      setError('Content is required');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      if (isEditing && documentToEdit) {
        // Update existing document
        const { document, error } = await updateDocumentContent(documentToEdit.id, title, content);

        if (error) {
          setError(error);
        } else {
          if (onSuccess) {
            onSuccess();
          }
        }
      } else {
        // Create new document
        const { document, error } = await createManualDocument(title, content);

        if (error) {
          setError(error);
        } else {
          setTitle('');
          setContent('');
          if (onSuccess) {
            onSuccess();
          }
        }
      }
    } catch (err) {
      setError(`Failed to ${isEditing ? 'update' : 'create'} manual document`);
      console.error(`Failed to ${isEditing ? 'update' : 'create'} manual document:`, err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label
          htmlFor="title"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          Document Title
        </label>
        <Input
          id="title"
          type="text"
          placeholder="Enter document title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
        />
      </div>

      <div>
        <label
          htmlFor="content"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          Document Content
        </label>
        <textarea
          id="content"
          placeholder="Enter document content"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="w-full h-64 px-3 py-2 text-gray-700 border rounded-lg focus:outline-none dark:bg-gray-700 dark:text-white dark:border-gray-600"
          required
        />
      </div>

      {error && (
        <div className="p-3 text-sm text-red-700 bg-red-100 rounded-lg dark:bg-red-200 dark:text-red-800">
          {error}
        </div>
      )}

      <div className="flex justify-end space-x-2">
        {onCancel && (
          <Button type="button" variant="outline" onClick={onCancel} disabled={isSubmitting}>
            Cancel
          </Button>
        )}
        <Button type="submit" variant="default" isLoading={isSubmitting}>
          {isSubmitting
            ? isEditing
              ? 'Saving...'
              : 'Creating...'
            : isEditing
              ? 'Save Changes'
              : 'Create Document'}
        </Button>
      </div>
    </form>
  );
};

export default ManualDocumentForm;
