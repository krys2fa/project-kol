# Minimal Python base
FROM python:3.11-slim

# System deps for onnxruntime and audio libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to OpenAI/Groq agent; override via CMD/args if needed
ENV PYTHONUNBUFFERED=1

# At runtime, set env vars:
# - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
# - OPENAI_API_KEY (Groq key), OPENAI_BASE_URL=https://api.groq.com/openai/v1
# - DEEPGRAM_API_KEY, CARTESIA_API_KEY
# - RAG_TOP_K=1, DG_ENDPOINT_MS=15 (optional tuning)

CMD ["python", "voice_agent_openai.py", "connect", "--room", "demo-room"]
