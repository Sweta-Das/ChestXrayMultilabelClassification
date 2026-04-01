from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime
import uuid


def generate_patient_id():
    return f"PID-{uuid.uuid4().hex[:8].upper()}"


def create_medical_report_pdf(report_text, output_path, patient_id):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=A4)

    elements = []

    elements.append(Paragraph(
        "<b>AI-Assisted Medical Imaging Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 0.3 * inch))

    meta = f"""
    <b>Patient ID:</b> {patient_id}<br/>
    <b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Modality:</b> Chest X-ray<br/>
    <b>Analysis Type:</b> AI-assisted (CTransCNN + RAG)
    """
    elements.append(Paragraph(meta, styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    for block in report_text.split("\n\n"):
        if block.strip().isupper():
            elements.append(Paragraph(block, styles["Heading2"]))
        else:
            elements.append(Paragraph(block.replace("\n", "<br/>"), styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
