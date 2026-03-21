// components/ImageUploader.tsx
'use client';

import { useCallback } from 'react';
import { useStore } from '@/lib/store';
import { uploadImage } from '@/lib/api';

export default function ImageUploader() {
  const { setUploadedFile, setSessionId, setLoading, reset } = useStore();

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
      }

      // Reset previous state
      reset();

      // Create preview URL
      const url = URL.createObjectURL(file);
      setUploadedFile(file, url);

      // Upload to backend
      setLoading(true, 'Uploading image...');
      try {
        const response = await uploadImage(file);
        setSessionId(response.session_id);
      } catch (error) {
        console.error('Upload failed:', error);
        alert('Failed to upload image');
        setUploadedFile(null, null);
      } finally {
        setLoading(false);
      }
    },
    [setUploadedFile, setSessionId, setLoading, reset]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        // Create a fake input event
        const input = document.createElement('input');
        input.type = 'file';
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;

        handleFileChange({ target: input } as any);
      }
    },
    [handleFileChange]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  return (
    <div
      className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-12 text-center hover:border-blue-500 transition-colors"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <div className="space-y-4">
        <div className="flex justify-center">
          <svg
            className="w-16 h-16 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        <div>
          <label htmlFor="file-upload" className="cursor-pointer">
            <span className="text-blue-600 dark:text-blue-400 hover:underline font-medium">
              Upload a chest X-ray
            </span>
            <span className="text-gray-600 dark:text-gray-400"> or drag and drop</span>
          </label>
          <input
            id="file-upload"
            name="file-upload"
            type="file"
            className="sr-only"
            accept="image/*"
            onChange={handleFileChange}
          />
        </div>
        <p className="text-sm text-gray-500">PNG, JPG, JPEG up to 10MB</p>
      </div>
    </div>
  );
}
