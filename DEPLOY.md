# Deploy (Option B: Fully hosted on free tiers)

This uses:

- LiveKit Cloud (room + media)
- Vercel (free) for a token server
- Fly.io (free allowance) to run the agent container
- Groq (OpenAI-compatible) as the LLM
- Deepgram for STT, Cartesia for TTS

## 0) Prereqs

- LiveKit project (get URL, API key/secret).
- Groq key (acts as OPENAI_API_KEY) and set OPENAI_BASE_URL=https://api.groq.com/openai/v1
- Deepgram API key, Cartesia API key.
- Docker installed locally; Fly.io account (`flyctl`).

## 1) Configure env

Create `.env` locally (for testing) and gather the same values for Fly secrets:

```
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

OPENAI_API_KEY=...            # your Groq key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant

DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...

RAG_TOP_K=1
DG_ENDPOINT_MS=15
```

## 2) Token server on Vercel (free)

- Deploy any minimal LiveKit token endpoint (Next.js or serverless) that mints access tokens using LIVEKIT_URL/KEY/SECRET.
- Resulting endpoint: `https://<your-app>.vercel.app/api/token?room=<room>&identity=<name>`
- Share this URL with your client and agent.

## 3) Build & run locally (optional)

```
docker build -t voice-agent .
docker run --env-file .env --rm voice-agent
```

## 4) Fly.io deploy

```
flyctl launch --name voice-agent --no-deploy
flyctl secrets set \
  LIVEKIT_URL=... \
  LIVEKIT_API_KEY=... \
  LIVEKIT_API_SECRET=... \
  OPENAI_API_KEY=... \
  OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
  DEEPGRAM_API_KEY=... \
  CARTESIA_API_KEY=...

flyctl deploy
```

The default CMD connects to room `demo-room`. To change it:

```
flyctl scale count 0
flyctl deploy --build-arg ROOM=my-room
```

Or edit CMD in `Dockerfile`.

## 5) Try it

- Start your LiveKit client and join `demo-room` using your Vercel token server.
- The agent should greet and respond using Groq + Deepgram + Cartesia.

## Notes

- To switch models: set `OPENAI_MODEL`.
- To tune latency: `RAG_TOP_K`, `DG_ENDPOINT_MS`.
- If you prefer OpenAI directly, keep `OPENAI_BASE_URL` default and use an OpenAI key.
