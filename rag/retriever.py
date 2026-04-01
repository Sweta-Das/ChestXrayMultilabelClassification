# rag/retriever.py

from pathlib import Path



KB_PATH = Path(__file__).resolve().parents[1] / "knowledge_base" / "medical_docs.txt"



def retrieve_context(diseases):
    """
    Retrieve disease-specific sections from the medical knowledge base.

    Fixes:
    - Robust disease name matching
    - Stable retrieval order
    - Guardrail against empty context
    """

    if not KB_PATH.exists():
        return ""

    text = KB_PATH.read_text(encoding="utf-8")

    retrieved_sections = []

    # Ensure stable order and clean names
    for disease in list(diseases):
        disease_clean = disease.strip()

        header = f"===== DISEASE: {disease_clean} ====="
        start = text.find(header)

        # Skip if disease not found in KB
        if start == -1:
            continue

        start += len(header)
        end = text.find("===== DISEASE:", start)

        if end == -1:
            section = text[start:].strip()
        else:
            section = text[start:end].strip()

        if section:
            retrieved_sections.append(
                f"{disease_clean}:\n{section}"
            )

    # Guardrail: never return empty context silently
    if not retrieved_sections:
        return (
            "No disease-specific reference sections were retrieved from the "
            "knowledge base. Base the interpretation on general radiological "
            "patterns implied by the disease likelihoods."
        )

    return "\n\n".join(retrieved_sections)
