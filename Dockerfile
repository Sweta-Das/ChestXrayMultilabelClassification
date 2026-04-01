FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LATEX_BIN=pdflatex

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    latexmk \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user/app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
  && python -m pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user
RUN mkdir -p /tmp/chestxray-app && chown -R user:user /tmp/chestxray-app

COPY --chown=user:user . .
COPY --chown=user:user --from=frontend-builder /app/frontend/out ./frontend/out

USER user

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
