# Voice RAG Agent (LiveKit + LlamaIndex + Groq/Deepgram/Cartesia)

Real‑time voice assistant that retrieves answers from your docs with LlamaIndex and talks over LiveKit. It supports two run modes:

- Local model (Ollama) with `voice_agent.py` (optional for dev)
- Cloud model (Groq via OpenAI‑compatible API) with `voice_agent_openai.py` (recommended for local + production)

Speech stack: Silero VAD, Deepgram STT, Cartesia TTS. A minimal Vercel token server mints LiveKit tokens for client access.

## Features

- LiveKit realtime audio and agent pipeline
- RAG over files in `docs/` (example: a restaurant menu)
- LLM via LlamaIndex: Groq (OpenAI‑compatible) or local Ollama
- Low‑latency tuning: warmed LLM, reduced RAG top‑k, fast VAD and endpointing
- Token minting endpoint for clients on Vercel (`token-server-vercel/`)

## Repo layout

```
voice_agent.py                # Local Ollama + RAG + Deepgram + Cartesia
voice_agent_openai.py         # Groq(OpenAI-compatible) + RAG + Deepgram + Cartesia
requirements.txt              # Python deps
Dockerfile                    # Optional container for hosting (not required)
DEPLOY.md                     # Fly.io container deploy notes (optional)
docs/restaurant_menu.txt      # Sample RAG content
chat-engine-storage/          # Created at runtime to persist LlamaIndex

# Token server (Vercel)
token-server-vercel/api/token.js   # GET /api/token?room=...&identity=...
token-server-vercel/package.json
token-server-vercel/vercel.json
token-server-vercel/README.md
```

## Requirements

- Python 3.11+
- A virtual environment (recommended)
- Accounts/keys:
  - LiveKit Cloud (URL, API Key, API Secret)
  - Deepgram (STT)
  - Cartesia (TTS)
  - Groq (OpenAI‑compatible LLM)
- Optional for local model: Ollama installed and a small model pulled (e.g. `llama3.2:1b`)

## Environment variables

Create a `.env` in the repo root (do not commit secrets):

```
# LiveKit Cloud
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

# LLM (Groq via OpenAI-compatible API)
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant

# STT / TTS
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...

# Latency tuning (optional)
RAG_TOP_K=1
DG_ENDPOINT_MS=15
VAD_MIN_SILENCE_S=0.35
VAD_PREFIX_PAD_S=0.2
VAD_ACTIVATION=0.5

# Optional local LLM override (for voice_agent.py)
OLLAMA_MODEL=llama3.2:1b
```

Notes:

- Do not store `LIVEKIT_TOKEN` in .env for production; use the Vercel token server instead.
- `RAG_TOP_K` controls how many docs are retrieved; smaller is faster.

## Install dependencies (one‑time)

```powershell
# Windows PowerShell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Run locally (recommended: Groq path)

```powershell
# Uses voice_agent_openai.py (Groq + Deepgram + Cartesia)
.\.venv\Scripts\python.exe voice_agent_openai.py connect --room demo-room
```

Join the room from a client (browser):

- Deploy the Vercel token server below, then fetch a token:
  - `https://<your-vercel-app>.vercel.app/api/token?room=demo-room&identity=alice`
- Use the token and your `LIVEKIT_URL` in example.livekit.io or your own client.

### Live demo (ready-made links)

- Step 1: Generate a token (change `identity` to your name):
  - https://token-server-kol.vercel.app/api/token?room=demo-room&identity=john
- Step 2: Copy the `token` value from the JSON response.
- Step 3: Open the LiveKit Custom client and paste the token at the end of the URL:
  - Base URL:
    - https://example.livekit.io/custom?liveKitUrl=wss://project-kol-3u9osc9x.livekit.cloud&token=
  - Full URL (example):
    - `https://example.livekit.io/custom?liveKitUrl=wss://project-kol-3u9osc9x.livekit.cloud&token=<PASTE_TOKEN_HERE>`
  - Press Enter; you should connect to `demo-room` and hear the agent.

## Optional: Local model mode (Ollama)

```powershell
# Ensure Ollama is running and model exists
# ollama pull llama3.2:1b
.\.venv\Scripts\python.exe voice_agent.py connect --room demo-room
```

## RAG data

- Place `.txt`/`.md` files in `docs/`
- First run builds an index into `chat-engine-storage/`
- To force reindex: delete `chat-engine-storage/` and run again

## Token server on Vercel

Minimal serverless function that mints LiveKit access tokens.

Folder: `token-server-vercel/`

Local dev (optional):

```powershell
cd token-server-vercel
npm install
npx vercel dev
# http://localhost:3000/api/token?room=demo-room&identity=alice
```

Deploy via CLI:

```powershell
npm i -g vercel
vercel login
cd token-server-vercel
vercel
# Add env vars for Production (repeat for Preview if desired)
vercel env add LIVEKIT_API_KEY production
vercel env add LIVEKIT_API_SECRET production
vercel env add LIVEKIT_URL production  # optional
vercel --prod
```

Or via Vercel Dashboard:

- Create a project from `token-server-vercel/` folder
- Add env vars (Production + Preview):
  - `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, optional `LIVEKIT_URL`
- Deploy; test:
  - `https://<your-vercel-app>.vercel.app/api/token?room=demo-room&identity=alice`

Response:

```json
{ "token": "<JWT>" }
```

Security:

- Keep LiveKit API Secret only in Vercel env vars
- Optionally add a shared header/API key or room allowlist in `api/token.js`

## Production hosting (no Docker)

If you need an always‑on agent without Docker/Fly, use a PaaS like Railway as a background worker:

- Build command: `pip install -r requirements.txt`
- Start command: `python start_railway.py`
- Variables: same as `.env` above
- Choose a background/worker process (no HTTP port needed)

Alternatively, you can containerize with the provided `Dockerfile` (optional) or use Fly.io (see `DEPLOY.md`).

### Railway setup (step-by-step)

1. Create New → Empty Project in Railway, then “Deploy from GitHub” and select this repo.
2. In the new Service:

- Build Command: `pip install -r requirements.txt`
- Start Command: `python start_railway.py`

3. Set Environment Variables (same as your `.env`, but add `ROOM`):

- LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
- GROQ_API_KEY
- OPENAI_BASE_URL=https://api.groq.com/openai/v1
- OPENAI_MODEL=llama-3.1-8b-instant
- DEEPGRAM_API_KEY, DEEPGRAM_MODEL=nova-3-general
- CARTESIA_API_KEY
- RAG_TOP_K=1, DG_ENDPOINT_MS=15 (optional)
- ROOM=demo-room (or any room you want the worker to join)

4. Deploy. Watch logs for “Connecting to room <ROOM>”.
5. Test from the browser using your Vercel token endpoint and example.livekit.io (see Live demo above).

## Performance and latency tips

- Groq model: `OPENAI_MODEL=llama-3.1-8b-instant` is fast; try even smaller if available
- `RAG_TOP_K=1` for speedy retrieval
- Deepgram endpointing: `DG_ENDPOINT_MS=15` trims silence
- Silero VAD: tune `VAD_MIN_SILENCE_S`, `VAD_PREFIX_PAD_S`, `VAD_ACTIVATION`
- Local model mode warms the model and sets `keep_alive` to reduce cold starts

## Troubleshooting

- STT errors (Deepgram API key required): ensure `DEEPGRAM_API_KEY` is set
- OpenAI/Groq errors: verify `OPENAI_API_KEY` and `OPENAI_BASE_URL`
- AssemblyAI model deprecated: the app defaults to Deepgram to avoid this
- Slow first response: ask a short question first to warm the model; subsequent turns are faster
- RAG not updating: delete `chat-engine-storage/` and rerun

## Security

- Rotate any keys you shared during development
- Do not commit `.env`; prefer platform secrets (Vercel/hosted PaaS)

## License

MIT (or your preferred license)
