'use client';

import { useState } from 'react';
import Image from 'next/image';

import { generateBBoxComparison } from '@/lib/api';
import { useStore } from '@/lib/store';

export default function BBoxComparisonViewer() {
  const { sessionId, predictions, selectedDisease, setLoading } = useStore();
  const [comparisonData, setComparisonData] = useState<null | {
    found: boolean;
    image_index?: string;
    disease?: string;
    available_diseases?: string[];
    bbox?: { x: number; y: number; w: number; h: number };
    gradcam_probability?: number;
    bbox_heatmap_energy_ratio?: number;
    triptych?: string;
    message?: string;
  }>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!sessionId) return;

    setError(null);
    setLoading(true, 'Comparing Grad-CAM with ground-truth bbox...');
    try {
      const response = await generateBBoxComparison(sessionId, selectedDisease || undefined);
      setComparisonData(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'BBox comparison failed.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId) return null;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 space-y-4">
      <div className="flex flex-col gap-2">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          BBox vs Grad-CAM Comparison
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Uses NIH ground-truth bounding boxes when the uploaded image matches an annotated dataset entry.
        </p>
      </div>

      <button
        onClick={handleCompare}
        disabled={!predictions}
        className="inline-flex items-center justify-center rounded-lg bg-rose-600 px-4 py-2 font-semibold text-white transition-colors hover:bg-rose-700 disabled:cursor-not-allowed disabled:bg-rose-300 dark:disabled:bg-rose-900/30"
      >
        {predictions ? 'Compare with Ground Truth BBox' : 'Run AI Analysis First'}
      </button>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-300">
          {error}
        </div>
      )}

      {comparisonData && (
        <div className="space-y-4">
          {!comparisonData.found ? (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-900/20 dark:text-amber-200">
              {comparisonData.message || 'No comparison available for this image.'}
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-700/50">
                  <span className="block text-gray-500 dark:text-gray-400">Image</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {comparisonData.image_index}
                  </span>
                </div>
                <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-700/50">
                  <span className="block text-gray-500 dark:text-gray-400">Disease</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {comparisonData.disease}
                  </span>
                </div>
                <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-700/50">
                  <span className="block text-gray-500 dark:text-gray-400">Grad-CAM probability</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {(comparisonData.gradcam_probability ?? 0).toFixed(3)}
                  </span>
                </div>
                <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-700/50">
                  <span className="block text-gray-500 dark:text-gray-400">BBox energy ratio</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {(comparisonData.bbox_heatmap_energy_ratio ?? 0).toFixed(4)}
                  </span>
                </div>
              </div>

              {comparisonData.triptych && (
                <div className="relative overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700 aspect-[3/1] bg-black/5 dark:bg-black/20">
                  <Image
                    src={comparisonData.triptych}
                    alt="Ground truth bbox versus Grad-CAM triptych"
                    fill
                    unoptimized
                    className="object-contain"
                    sizes="(max-width: 768px) 100vw, 900px"
                  />
                </div>
              )}

              {comparisonData.available_diseases && comparisonData.available_diseases.length > 0 && (
                <details className="rounded-lg border border-gray-200 dark:border-gray-700">
                  <summary className="cursor-pointer p-4 font-medium text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800">
                    Available bbox labels ({comparisonData.available_diseases.length})
                  </summary>
                  <div className="border-t border-gray-200 p-4 text-sm text-gray-700 dark:border-gray-700 dark:text-gray-300">
                    {comparisonData.available_diseases.join(', ')}
                  </div>
                </details>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
