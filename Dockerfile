FROM python:3.11-slim
WORKDIR /app

# System deps for MeCab/fugashi compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

ENV TRANSCRIPT_AI_PROVIDER=auto

CMD ["uvicorn", "main:app", \
    "--host=0.0.0.0", \
    "--port=7860"]