// lib/store.ts
/**
 * Zustand state management for application
 */

import { create } from 'zustand';

interface Disease {
  disease: string;
  probability: number;
}

interface AppState {
  // Patient info
  patientAge: string;
  setPatientAge: (age: string) => void;

  // Session
  sessionId: string | null;
  setSessionId: (id: string | null) => void;

  // Upload
  uploadedFile: File | null;
  uploadedImageUrl: string | null;
  setUploadedFile: (file: File | null, url: string | null) => void;

  // Predictions
  predictions: Record<string, number> | null;
  topDiseases: Disease[];
  setPredictions: (predictions: Record<string, number>, topDiseases: Disease[]) => void;

  // GradCAM
  gradcamData: Record<string, any> | null;
  selectedDisease: string | null;
  heatmapOpacity: number;
  setGradcamData: (data: Record<string, any>) => void;
  mergeGradcamData: (data: Record<string, any>) => void;
  setSelectedDisease: (disease: string | null) => void;
  setHeatmapOpacity: (opacity: number) => void;

  // Report
  reportData: {
    patientId: string;
    impression: string;
    findings: Record<string, number>;
  } | null;
  setReportData: (data: any) => void;

  // UI State
  isLoading: boolean;
  loadingMessage: string;
  setLoading: (loading: boolean, message?: string) => void;

  // Reset
  reset: () => void;
}

export const useStore = create<AppState>((set) => ({
  // Patient info
  patientAge: '48',
  setPatientAge: (age) => set({ patientAge: age }),

  // Session
  sessionId: null,
  setSessionId: (id) => set({ sessionId: id }),

  // Upload
  uploadedFile: null,
  uploadedImageUrl: null,
  setUploadedFile: (file, url) => set({ uploadedFile: file, uploadedImageUrl: url }),

  // Predictions
  predictions: null,
  topDiseases: [],
  setPredictions: (predictions, topDiseases) => set({ predictions, topDiseases }),

  // GradCAM
  gradcamData: null,
  selectedDisease: null,
  heatmapOpacity: 0.5,
  setGradcamData: (data) => set({ gradcamData: data }),
  mergeGradcamData: (data) =>
    set((state) => ({
      gradcamData: {
        ...(state.gradcamData || {}),
        ...data,
      },
    })),
  setSelectedDisease: (disease) => set({ selectedDisease: disease }),
  setHeatmapOpacity: (opacity) => set({ heatmapOpacity: opacity }),

  // Report
  reportData: null,
  setReportData: (data) => set({ reportData: data }),

  // UI State
  isLoading: false,
  loadingMessage: '',
  setLoading: (loading, message = '') => set({ isLoading: loading, loadingMessage: message }),

  // Reset
  reset: () =>
    set({
      sessionId: null,
      patientAge: '48',
      uploadedFile: null,
      uploadedImageUrl: null,
      predictions: null,
      topDiseases: [],
      gradcamData: null,
      selectedDisease: null,
      heatmapOpacity: 0.5,
      reportData: null,
      isLoading: false,
      loadingMessage: '',
    }),
}));
