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
    import logging
    import os
    import sys
    from dotenv import load_dotenv
    from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
    from livekit.agents.job import AutoSubscribe
    from livekit.agents.llm import ChatContext
    from livekit.agents.pipeline import VoicePipelineAgent
    from livekit.plugins import cartesia, silero, llama_index
    from livekit.plugins.deepgram import STT as DeepgramSTT

    load_dotenv()

    logger = logging.getLogger("voice-assistant")
    try:
        from llama_index.llms.groq import Groq as GroqLLM  # type: ignore
        _OPENAI_LIKE = False
    except Exception:
        from llama_index.llms.openai import OpenAI as OpenAILLM  # type: ignore
        _OPENAI_LIKE = False
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, Settings
    from llama_index.core.chat_engine.types import ChatMode
    load_dotenv()

    # Configure OpenAI-compatible LLM (Groq) via env
    api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("GROQ_MODEL")
        or "llama-3.1-8b-instant"
    )
    Settings.llm = GroqLLM(model=model_name, api_key=api_key)

    # check if storage already exists
    PERSIST_DIR = "./chat-engine-storage"
    REQUIRE_INDEX = os.getenv("REQUIRE_INDEX", "").lower() in ("1", "true", "yes")
    if not os.path.exists(PERSIST_DIR):
        if REQUIRE_INDEX:
            raise RuntimeError(
                "chat-engine-storage not found. Prebuild the index locally (run the agent once) "
                "and commit the 'chat-engine-storage' folder, or unset REQUIRE_INDEX to allow building."
            )
        # Only import heavy embeddings when we need to build the index locally
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        documents = SimpleDirectoryReader("docs").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)



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
        import logging
        import os
        import sys
        from dotenv import load_dotenv
        from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
        from livekit.agents.job import AutoSubscribe
        from livekit.agents.llm import ChatContext
        from livekit.agents.pipeline import VoicePipelineAgent
        from livekit.plugins import cartesia, silero, llama_index
        from livekit.plugins.deepgram import STT as DeepgramSTT
        from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, Settings
        from llama_index.core.chat_engine.types import ChatMode

        load_dotenv()

        logger = logging.getLogger("voice-assistant")
        try:
            from llama_index.llms.groq import Groq as GroqLLM  # type: ignore
            _OPENAI_LIKE = False
        except Exception:
            from llama_index.llms.openai import OpenAI as OpenAILLM  # type: ignore
            _OPENAI_LIKE = False

        # Configure OpenAI-compatible LLM (Groq) via env
        api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("GROQ_MODEL")
            or "llama-3.1-8b-instant"
        )
        Settings.llm = GroqLLM(model=model_name, api_key=api_key)

        # check if storage already exists
        PERSIST_DIR = "./chat-engine-storage"
        REQUIRE_INDEX = os.getenv("REQUIRE_INDEX", "").lower() in ("1", "true", "yes")
        if not os.path.exists(PERSIST_DIR):
            if REQUIRE_INDEX:
                raise RuntimeError(
                    "chat-engine-storage not found. Prebuild the index locally (run the agent once) "
                    "and commit the 'chat-engine-storage' folder, or unset REQUIRE_INDEX to allow building."
                )
            # Only import heavy embeddings when we need to build the index locally
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            documents = SimpleDirectoryReader("docs").load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)