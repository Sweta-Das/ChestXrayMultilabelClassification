// components/PredictionResults.tsx
'use client';

import { generateGradCAM } from '@/lib/api';
import { useStore } from '@/lib/store';

export default function PredictionResults() {
  const {
    sessionId,
    predictions,
    topDiseases,
    selectedDisease,
    gradcamData,
    setSelectedDisease,
    mergeGradcamData,
    setLoading,
  } = useStore();

  const handleSelectDisease = async (disease: string) => {
    setSelectedDisease(disease);

    if (!sessionId || !predictions || gradcamData?.[disease]) {
      return;
    }

    const diseaseIndex = Object.keys(predictions).findIndex((name) => name === disease);
    setLoading(true, `Loading heatmap for ${disease}...`);
    try {
      const response = await generateGradCAM(sessionId, diseaseIndex >= 0 ? diseaseIndex : 0, 1, 0.1);
      mergeGradcamData(response.heatmaps);
    } catch (error) {
      console.error('GradCAM fetch failed:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!predictions) return null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Analysis Results
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Select a disease, then click Explain to generate its GradCAM heatmap
        </p>
      </div>

      {/* Top Diseases */}
      <div className="space-y-3">
        {topDiseases.map((item) => (
          <div
            key={item.disease}
            onClick={() => handleSelectDisease(item.disease)}
            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
              selectedDisease === item.disease
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                {item.disease}
              </h3>
              <span
                className={`text-lg font-bold ${
                  item.probability > 0.5
                    ? 'text-red-600 dark:text-red-400'
                    : item.probability > 0.3
                    ? 'text-orange-600 dark:text-orange-400'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                {(item.probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  item.probability > 0.5
                    ? 'bg-red-600'
                    : item.probability > 0.3
                    ? 'bg-orange-500'
                    : 'bg-blue-500'
                }`}
                style={{ width: `${item.probability * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* All Predictions (Collapsible) */}
      <details className="border border-gray-200 dark:border-gray-700 rounded-lg">
        <summary className="cursor-pointer p-4 font-medium text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800">
          View All Predictions ({Object.keys(predictions).length})
        </summary>
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
          {Object.entries(predictions)
            .sort(([, a], [, b]) => b - a)
            .map(([disease, prob]) => (
              <div
                key={disease}
                className="flex justify-between items-center text-sm"
              >
                <span className="text-gray-700 dark:text-gray-300">{disease}</span>
                <span className="font-mono text-gray-600 dark:text-gray-400">
                  {(prob * 100).toFixed(2)}%
                </span>
              </div>
            ))}
        </div>
      </details>
    </div>
  );
}
