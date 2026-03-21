# rag/medical_reasoner.py

from rag.retriever import retrieve_context
from rag.report_agent import generate_report
from rag.disease_selector import select_diseases


def generate_structured_content(disease_probs, top_k=5):
    """
    Generate structured medical content for report generation.

    - Uses unified disease selection logic
    - Guarantees non-empty RAG context
    - Preserves original output schema
    """

    # ✅ Use the centralized selector (threshold + Top-K fallback)
    selected = select_diseases(disease_probs, min_k=top_k)

    # Retrieve medical knowledge
    context = retrieve_context(selected.keys())

    # 🛑 Guardrail: prevent empty-context RAG
    if not context.strip():
        context = (
            "No detailed disease-specific medical context was retrieved. "
            "Base the impression on radiological patterns implied by the "
            "provided disease likelihoods."
        )

    # Generate impression text
    impression = generate_report(selected, context)

    return {
        "findings": selected,
        "impression": impression
    }
