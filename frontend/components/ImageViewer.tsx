// components/ImageViewer.tsx
'use client';

import { useState } from 'react';
import { useStore } from '@/lib/store';
import Image from 'next/image';

type ViewMode = 'original' | 'heatmap' | 'overlay';

export default function ImageViewer() {
  const {
    uploadedImageUrl,
    gradcamData,
    selectedDisease,
    heatmapOpacity,
    setHeatmapOpacity,
  } = useStore();

  const [viewMode, setViewMode] = useState<ViewMode>('original');

  if (!uploadedImageUrl) return null;

  const currentHeatmap = selectedDisease && gradcamData?.[selectedDisease];

  const getImageSrc = () => {
    if (!currentHeatmap) return uploadedImageUrl;

    switch (viewMode) {
      case 'heatmap':
        return currentHeatmap.heatmap;
      case 'overlay':
        return currentHeatmap.overlay;
      default:
        return uploadedImageUrl;
    }
  };

  return (
    <div className="space-y-4">
      {/* View Mode Selector */}
      <div className="flex gap-2 justify-center">
        <button
          onClick={() => setViewMode('original')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'original'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
          }`}
        >
          Original
        </button>
        <button
          onClick={() => setViewMode('heatmap')}
          disabled={!currentHeatmap}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'heatmap'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed'
          }`}
        >
          Heatmap
        </button>
        <button
          onClick={() => setViewMode('overlay')}
          disabled={!currentHeatmap}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'overlay'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed'
          }`}
        >
          Overlay
        </button>
      </div>

      {/* Image Display */}
      <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '1' }}>
        <img
          src={getImageSrc()}
          alt="Chest X-ray"
          className="w-full h-full object-contain"
        />
      </div>

      {/* Opacity Slider (for overlay mode) */}
      {viewMode === 'overlay' && currentHeatmap && (
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Heatmap Opacity: {Math.round(heatmapOpacity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={heatmapOpacity * 100}
            onChange={(e) => setHeatmapOpacity(Number(e.target.value) / 100)}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
          />
        </div>
      )}

      {/* Info Text */}
      {selectedDisease && currentHeatmap && (
        <div className="text-sm text-gray-600 dark:text-gray-400 text-center">
          Showing GradCAM for <span className="font-semibold">{selectedDisease}</span> (
          {(currentHeatmap.probability * 100).toFixed(1)}%)
        </div>
      )}
    </div>
  );
}
