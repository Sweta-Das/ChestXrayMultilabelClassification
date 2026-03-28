// components/ActionButtons.tsx
'use client';

import { useStore } from '@/lib/store';
import { runInference, generateGradCAM, generateReport, getDownloadUrl } from '@/lib/api';

export default function ActionButtons() {
  const {
    sessionId,
    predictions,
    selectedDisease,
    gradcamData,
    reportData,
    setLoading,
    setPredictions,
    setGradcamData,
    setReportData,
    setSelectedDisease,
  } = useStore();

  const handleRunInference = async () => {
    if (!sessionId) return;

    setLoading(true, 'Running AI analysis...');
    try {
      const response = await runInference(sessionId);
      setPredictions(response.predictions, response.top_diseases);

      // Auto-select first disease
      if (response.top_diseases.length > 0) {
        setSelectedDisease(response.top_diseases[0].disease);
      }
    } catch (error) {
      console.error('Inference failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateGradCAM = async () => {
    if (!sessionId || !predictions) return;

    const diseaseIndex = selectedDisease
      ? Object.keys(predictions).findIndex((disease) => disease === selectedDisease)
      : 0;

    setLoading(true, 'Generating explainability heatmaps...');
    try {
      const response = await generateGradCAM(sessionId, diseaseIndex >= 0 ? diseaseIndex : 0, 5, 0.1);
      setGradcamData(response.heatmaps);
    } catch (error) {
      console.error('GradCAM failed:', error);
      alert('Heatmap generation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!sessionId || !predictions) return;

    setLoading(true, 'Generating medical report...');
    try {
      const response = await generateReport(sessionId);
      setReportData({
        patientId: response.patient_id,
        impression: response.impression,
        findings: response.findings,
      });
    } catch (error) {
      console.error('Report generation failed:', error);
      const message = error instanceof Error ? error.message : 'Report generation failed. Please try again.';
      alert(message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    if (!sessionId) return;
    const url = getDownloadUrl(sessionId);
    window.open(url, '_blank');
  };

  return (
    <div className="space-y-3">
      {/* Run Inference */}
      {sessionId && !predictions && (
        <button
          onClick={handleRunInference}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg"
        >
          🔬 Run AI Analysis
        </button>
      )}

      {/* Generate GradCAM */}
      {predictions && !gradcamData && (
        <button
          onClick={handleGenerateGradCAM}
          className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg"
        >
          🔥 Explain Selected Disease
        </button>
      )}

      {/* Generate Report */}
      {predictions && !reportData && (
        <button
          onClick={handleGenerateReport}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg"
        >
          📄 Generate Medical Report
        </button>
      )}

      {/* Download Report */}
      {reportData && (
        <button
          onClick={handleDownloadReport}
          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg"
        >
          📥 Download Report PDF
        </button>
      )}
    </div>
  );
}
