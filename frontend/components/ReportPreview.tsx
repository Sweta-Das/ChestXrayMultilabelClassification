// components/ReportPreview.tsx
'use client';

import { useStore } from '@/lib/store';

export default function ReportPreview() {
  const { reportData } = useStore();

  if (!reportData) return null;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 space-y-4">
      <div className="border-b border-gray-200 dark:border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Medical Report
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Patient ID: <span className="font-mono">{reportData.patientId}</span>
        </p>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Findings
        </h3>
        <div className="space-y-2">
          {Object.entries(reportData.findings).map(([disease, prob]) => (
            <div
              key={disease}
              className="flex justify-between items-center text-sm bg-gray-50 dark:bg-gray-700 p-2 rounded"
            >
              <span className="text-gray-700 dark:text-gray-300">{disease}</span>
              <span className="font-semibold text-gray-900 dark:text-gray-100">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Clinical Impression
        </h3>
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <p className="text-gray-800 dark:text-gray-200 leading-relaxed whitespace-pre-wrap">
            {reportData.impression}
          </p>
        </div>
      </div>
    </div>
  );
}
