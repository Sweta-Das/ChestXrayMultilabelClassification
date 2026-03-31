'use client';

import ImageUploader from '@/components/ImageUploader';
import ImageViewer from '@/components/ImageViewer';
import PredictionResults from '@/components/PredictionResults';
import ActionButtons from '@/components/ActionButtons';
import LoadingOverlay from '@/components/LoadingOverlay';
import ReportPreview from '@/components/ReportPreview';
import BBoxComparisonViewer from '@/components/BBoxComparisonViewer';
import { useStore } from '@/lib/store';

export default function Home() {
  const { uploadedImageUrl, predictions, reset } = useStore();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <LoadingOverlay />

      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                🩺 AI Chest X-ray Analysis
              </h1>
              <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                Advanced medical imaging with GradCAM explainability
              </p>
            </div>
            {uploadedImageUrl && (
              <button
                onClick={reset}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg font-medium transition-colors"
              >
                ↺ New Analysis
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!uploadedImageUrl ? (
          /* Upload Screen */
          <div className="max-w-2xl mx-auto">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
              <div className="mb-6">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">
                  Upload Chest X-ray
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Upload a chest X-ray image to begin AI-assisted analysis with advanced
                  explainability features.
                </p>
              </div>
              <ImageUploader />
              <div className="mt-8 grid grid-cols-3 gap-4 text-center">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    14
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Disease Classes
                  </div>
                </div>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    GradCAM
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Explainability
                  </div>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    RAG
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Medical Reports
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Analysis Screen */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column: Image Viewer */}
            <div className="space-y-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                  X-ray Image
                </h2>
                <ImageViewer />
              </div>

              {/* Action Buttons */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                <ActionButtons />
              </div>
            </div>

            {/* Right Column: Results */}
            <div className="space-y-6">
              {predictions && (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                  <PredictionResults />
                </div>
              )}

              {/* Report Preview */}
              <ReportPreview />

              {/* BBox vs Grad-CAM Comparison */}
              <BBoxComparisonViewer />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600 dark:text-gray-400">
            <p>
              <strong>MedFusionNet</strong> AI Model | PyTorch Grad-CAM Explainability | RAG Medical Reports
            </p>
            <p className="mt-2">
              ⚠️ For research and educational purposes only. Not for clinical diagnosis.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
