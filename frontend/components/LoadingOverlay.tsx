// components/LoadingOverlay.tsx
'use client';

import { useStore } from '@/lib/store';

export default function LoadingOverlay() {
  const { isLoading, loadingMessage } = useStore();

  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-8 shadow-2xl max-w-sm w-full mx-4">
        <div className="flex flex-col items-center space-y-4">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600"></div>
          <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
            {loadingMessage || 'Processing...'}
          </p>
        </div>
      </div>
    </div>
  );
}
