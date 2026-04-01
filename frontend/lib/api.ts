// lib/api.ts
/**
 * API client for communicating with FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL
  || (typeof window !== 'undefined'
    && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
      ? 'http://localhost:8000'
      : '');

export interface Disease {
  disease: string;
  probability: number;
}

export interface PredictionResponse {
  session_id: string;
  predictions: Record<string, number>;
  top_diseases: Disease[];
  timestamp: string;
}

export interface GradCAMResponse {
  session_id: string;
  heatmaps: Record<string, {
    probability: number;
    original: string;
    heatmap: string;
    overlay: string;
  }>;
}

export interface ReportResponse {
  session_id: string;
  patient_id: string;
  report_path: string;
  impression: string;
  findings: Record<string, number>;
}

export interface SessionData {
  file_path: string;
  age: number;
  uploaded_at: string;
  predictions?: Record<string, number> | null;
  gradcam?: unknown;
  report?: unknown;
  probs_list?: number[] | null;
}

/**
 * Upload chest X-ray image
 */
export async function uploadImage(
  file: File,
  age?: string
): Promise<{ session_id: string; filename: string }> {
  const formData = new FormData();
  formData.append('file', file);
  if (age !== undefined && age.trim() !== '') {
    formData.append('age', age);
  }

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to upload image');
  }

  return response.json();
}

/**
 * Run model inference
 */
export async function runInference(sessionId: string): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('session_id', sessionId);

  const response = await fetch(`${API_BASE_URL}/api/inference`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Inference failed');
  }

  return response.json();
}

/**
 * Generate GradCAM heatmaps
 */
export async function generateGradCAM(
  sessionId: string,
  diseaseIndex?: number,
  topK: number = 5,
  threshold: number = 0.1
): Promise<GradCAMResponse> {
  console.log('[API] Generating GradCAM for session:', sessionId);
  
  const response = await fetch(`${API_BASE_URL}/api/gradcam`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      top_k: topK,
      threshold: threshold,
      disease_index: diseaseIndex,
    }),
  });

  console.log('[API] GradCAM response status:', response.status);
  
  if (!response.ok) {
    let message = 'GradCAM generation failed';
    try {
      const err = await response.json();
      if (err?.detail) message = `GradCAM failed: ${err.detail}`;
    } catch { /* keep default */ }
    throw new Error(message);
  }

  const data = await response.json();
  console.log('[API] GradCAM diseases:', Object.keys(data.heatmaps));
  return data;
}

/**
 * Generate medical report
 */
export async function generateReport(
  sessionId: string,
  patientId?: string
): Promise<ReportResponse> {
  console.log('[API] Generating report for session:', sessionId);
  console.log('[API] Backend URL:', API_BASE_URL);
  
  try {
    const requestBody = {
      session_id: sessionId,
      patient_id: patientId,
    };
    console.log('[API] Request body:', requestBody);
    
    const response = await fetch(`${API_BASE_URL}/api/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    console.log('[API] Response status:', response.status, response.statusText);
    
    if (!response.ok) {
      let message = 'Report generation failed';
      try {
        const err = await response.json();
        if (err?.detail) {
          message = `Report generation failed: ${err.detail}`;
        }
      } catch (e) {
        // keep default error message
        console.error('[API] Could not parse error response:', e);
      }
      throw new Error(message);
    }

    const result = await response.json();
    console.log('[API] Report generated successfully');
    return result;
  } catch (error) {
    console.error('[API] Report generation error:', error);
    throw error;
  }
}

/**
 * Download report PDF
 */
export function getDownloadUrl(sessionId: string): string {
  return `${API_BASE_URL}/api/download/${sessionId}`;
}

/**
 * Get session data
 */
export async function getSession(sessionId: string): Promise<SessionData> {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}`);

  if (!response.ok) {
    throw new Error('Failed to fetch session');
  }

  return response.json() as Promise<SessionData>;
}

/**
 * Delete session
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to delete session');
  }
}
