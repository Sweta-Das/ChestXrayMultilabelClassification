# rag/pipeline.py

from rag.disease_selector import select_diseases
from rag.retriever import retrieve_context
from rag.report_agent import generate_report


def generate_medical_report(disease_probs, threshold=0.1):
    selected = select_diseases(disease_probs, threshold)
    context = retrieve_context(selected.keys())
    report = generate_report(selected, context)
    return report
