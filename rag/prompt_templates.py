# rag/prompt_templates.py

def build_medical_report_prompt(predictions, context):
    findings = "\n".join(
        f"- {disease}: {prob * 100:.1f}% likelihood"
        for disease, prob in predictions.items()
    )

    return f"""
You are an AI system assisting in drafting radiology report IMPRESSIONS.

You are given:
1) AI-estimated disease likelihoods from a chest X-ray
2) Authoritative medical reference text

STRICT INSTRUCTIONS:
- Write ONLY the IMPRESSION section
- Base your reasoning STRICTLY on the provided medical context
- Do NOT repeat probabilities verbatim
- Do NOT list diseases one-by-one
- Group related radiographic patterns
- Use cautious language (e.g., "may represent", "suggestive of")
- Do NOT make a diagnosis or recommend treatment
- Do NOT add disclaimers or report headers

==============================
AI FINDINGS (FOR CONTEXT)
==============================
{findings}

==============================
AUTHORITATIVE MEDICAL CONTEXT
==============================
{context}

==============================
WRITE ONLY THE IMPRESSION BELOW
==============================
"""
