import logging
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.job import AutoSubscribe
from livekit.agents.llm import (
    ChatContext,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, silero, llama_index
from livekit.plugins.deepgram import STT as DeepgramSTT

load_dotenv()

logger = logging.getLogger("voice-assistant")
try:
    # Prefer OpenAI-compatible client that accepts custom model names (e.g., Groq)
    from llama_index.llms.groq import Groq as GroqLLM  # type: ignore
    _OPENAI_LIKE = False
except Exception:
    # Fallback to standard OpenAI client (may not support non-OpenAI model names)
    from llama_index.llms.openai import OpenAI as OpenAILLM  # type: ignore
    _OPENAI_LIKE = False
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Configure OpenAI-compatible LLM (Groq) via env
# Set OPENAI_API_KEY to your Groq API key
# Set OPENAI_BASE_URL=https://api.groq.com/openai/v1
# Optionally override model via OPENAI_MODEL or GROQ_MODEL
api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
model_name = (
    os.getenv("OPENAI_MODEL")
    or os.getenv("GROQ_MODEL")
    or "llama-3.1-8b-instant"
)
# Allow overriding context window explicitly for non-OpenAI models
context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
Settings.llm = GroqLLM(model=model_name, api_key=api_key)
# context_window override no longer needed with Groq adapter

# check if storage already exists
PERSIST_DIR = "./chat-engine-storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    chat_context = ChatContext().append(
        role="system",
        text=(
            "You are a friendly, accommodating restaurant call center agent for our restaurant. "
            "Your job is to help callers place takeout orders or book table reservations. "
            "Use only information found in the provided documents (e.g., menu and policies) and do not invent or guess. "
            "If the information is not available in the documents, clearly say it is not available. "
            "Keep answers short and concise, and avoid unpronounceable punctuation or emojis. "
            "For reservations, briefly collect name, party size, date, time, and phone number. "
            "For orders, briefly confirm item names, options, quantities, and pickup vs. delivery. "
            "Ask only minimal clarifying questions when needed, then provide the next step."
        ),
    )
    
    # Reduce retrieval to speed up RAG
    top_k = int(os.getenv("RAG_TOP_K", "1"))
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, similarity_top_k=top_k)



    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    # Prefer Deepgram STT for reliability/cost; requires DEEPGRAM_API_KEY
    endpoint_ms = int(os.getenv("DG_ENDPOINT_MS", "15"))
    dg_model = os.getenv("DEEPGRAM_MODEL", "nova-3-general")
    stt_impl = DeepgramSTT(
        model=dg_model,
        endpointing_ms=endpoint_ms,
        interim_results=True,
        no_delay=True,
        filler_words=False,
        energy_filter=True,
    )

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=stt_impl,
        llm=llama_index.LLM(chat_engine=chat_engine),
        tts=cartesia.TTS(
            model="sonic-2",
            voice="bf0a246a-8642-498a-9950-80c35e9276b5",
        ),
        chat_ctx=chat_context,
    )

    agent.start(ctx.room, participant)

    await agent.say(
        "Hey there! How can I help you today?",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    print("Starting voice agent...")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )