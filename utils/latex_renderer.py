import subprocess
from pathlib import Path


def render_latex_report(template_path, output_dir, context, filename="medical_report"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / f"{filename}.tex"
    pdf_path = tex_path.with_suffix(".pdf")

    # Fill template
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    for key, value in context.items():
        template = template.replace("{{" + key + "}}", str(value))

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(template)

    # Run LaTeX twice
    latex_cmd = ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode", tex_path.name]

    for _ in range(2):
        result = subprocess.run(
            latex_cmd,
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    # Only fail if PDF was NOT produced
    if not pdf_path.exists():
        raise RuntimeError(
            "LaTeX failed to generate PDF.\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    # Clean auxiliary files
    for ext in [".aux", ".log", ".out"]:
        aux_file = output_dir / f"{filename}{ext}"
        if aux_file.exists():
            aux_file.unlink()

    return pdf_path