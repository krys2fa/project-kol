import logging
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.job import AutoSubscribe
from livekit.agents.llm import ChatContext
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, silero, llama_index
from livekit.plugins.deepgram import STT as DeepgramSTT

load_dotenv()

logger = logging.getLogger("voice-assistant")

# Prefer Groq-native LLM; fall back to OpenAI client if unavailable
try:
    from llama_index.llms.groq import Groq as GroqLLM  # type: ignore
    _USE_OPENAI_CLIENT = False
except Exception:
    from llama_index.llms.openai import OpenAI as OpenAILLM  # type: ignore
    _USE_OPENAI_CLIENT = True

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.core.chat_engine.types import ChatMode


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _ensure_index():
    """Load a prebuilt index; optionally build locally if allowed.

    - Requires prebuilt storage at ./chat-engine-storage when REQUIRE_INDEX=1
    - Only imports HuggingFaceEmbedding when building locally.
    """
    persist_dir = "./chat-engine-storage"
    require_index = os.getenv("REQUIRE_INDEX", "").lower() in ("1", "true", "yes")

    if not os.path.exists(persist_dir):
        if require_index:
            raise RuntimeError(
                "chat-engine-storage not found. Prebuild the index locally (run the agent once) "
                "and commit the 'chat-engine-storage' folder, or unset REQUIRE_INDEX to allow building."
            )
        # Heavy import only when building locally
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        documents = SimpleDirectoryReader("docs").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        return index

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)


def prewarm(proc: JobProcess):
    # Faster end-of-speech detection
    min_silence_s = _get_float_env("VAD_MIN_SILENCE_S", 0.35)
    activation = _get_float_env("VAD_ACTIVATION", 0.5)
    prefix_pad_s = _get_float_env("VAD_PREFIX_PAD_S", 0.2)
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=min_silence_s,
        activation_threshold=activation,
        prefix_padding_duration=prefix_pad_s,
    )


async def entrypoint(ctx: JobContext):
    # Configure LLM (Groq via LlamaIndex Settings)
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("GROQ_MODEL")
        or "llama-3.1-8b-instant"
    )

    if _USE_OPENAI_CLIENT:
        # This path will only work with real OpenAI models; kept for safety.
        Settings.llm = OpenAILLM(model=model_name, api_key=api_key)
    else:
        Settings.llm = GroqLLM(model=model_name, api_key=api_key)

    # Restaurant call center behavior; doc-grounded and concise
    chat_context = ChatContext().append(
        role="system",
        text=(
            "You are a helpful restaurant call center agent for Kol Restaurant. "
            "Answer only using information from the provided documents. If the answer isn't in the docs, say you don't know and offer to connect a human. "
            "Be brief and conversational; no emojis or special punctuation."
        ),
    )

    def _build_bm25_chat_engine(top_k: int):
        # Lazy import to keep base image minimal
        from llama_index.core.retrievers import BM25Retriever
        from llama_index.core.chat_engine import ContextChatEngine

        documents = SimpleDirectoryReader("docs").load_data()
        retriever = BM25Retriever.from_defaults(documents=documents, similarity_top_k=top_k)
        return ContextChatEngine.from_defaults(retriever=retriever)

    top_k = int(os.getenv("RAG_TOP_K", "1"))
    chat_engine = None
    try:
        index = _ensure_index()
        chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, similarity_top_k=top_k)
    except Exception as e:
        # Fall back to BM25 when embeddings backends are missing
        msg = str(e)
        if isinstance(e, (ImportError, ModuleNotFoundError)) or "llama_index.embeddings" in msg or "llama-index-embeddings-openai" in msg:
            logging.warning("Embeddings unavailable; falling back to BM25 retriever.")
            chat_engine = _build_bm25_chat_engine(top_k)
        else:
            raise

    logging.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logging.info(f"Starting voice assistant for participant {participant.identity}")

    # Deepgram STT
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
        "Hi, this is Kol Restaurant. How can I help you today?",
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