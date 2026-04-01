# rag/report_agent.py

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from rag.prompt_templates import build_medical_report_prompt

# Robust .env loading for report generation in all entry points
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


def generate_report(predictions, context):
    """
    Generate a medical report impression using OpenAI.
    Preserves existing error handling and API behavior,
    while enforcing strong RAG grounding.
    """

    prompt = build_medical_report_prompt(predictions, context)

    try:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return (
                "Medical report generation is unavailable because OPENAI_API_KEY "
                "is not set. Please configure a valid key and retry."
            )

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical imaging AI assistant.\n"
                        "You MUST base your response strictly on the provided "
                        "medical reference context.\n"
                        "Write ONLY a radiology-style IMPRESSION section.\n"
                        "Do NOT repeat probabilities verbatim.\n"
                        "Do NOT use placeholder phrases such as "
                        "'information related to'.\n"
                        "Do NOT provide diagnosis or treatment.\n"
                        "If context is limited, reason cautiously from the "
                        "radiological patterns implied by the findings."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=600
        )

        return response.choices[0].message.content.strip()

    except OpenAIError:
        return (
            "Medical report generation is currently unavailable.\n\n"
            "The AI language service quota may be exhausted or temporarily "
            "unavailable. Please try again later."
        )
