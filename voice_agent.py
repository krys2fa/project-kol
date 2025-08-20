import logging
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.job import AutoSubscribe
from livekit.agents.llm import (
    ChatContext,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, silero, llama_index, assemblyai
from livekit.plugins.deepgram import STT as DeepgramSTT

load_dotenv()

logger = logging.getLogger("voice-assistant")
from llama_index.llms.ollama import Ollama
import os as _os
try:
    import ollama as _py_ollama  # python client for warming the model
except Exception:
    _py_ollama = None
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
# Use a lightweight local model to avoid RAM issues and OpenAI quota
OLLAMA_MODEL = _os.getenv("OLLAMA_MODEL", "llama3.2:1b")
llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=120.0,
    # modest generation limits to reduce latency
    additional_kwargs={
        "num_predict": 128,
        "num_ctx": 1024,
        "temperature": 0.6,
    "keep_alive": "10m",
    },
)
Settings.llm = llm
Settings.embed_model = embed_model

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


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


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
    # Warm up Ollama model so first response is faster
    if _py_ollama is not None:
        try:
            _py_ollama.generate(
                model=OLLAMA_MODEL,
                prompt="Hi",
                options={"num_predict": 5},
                keep_alive="10m",
            )
        except Exception:
            pass


async def entrypoint(ctx: JobContext):
    chat_context = ChatContext().append(
        role="system",
        text=(
            "You are a funny, witty assistant."
            "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
        ),
    )
    
    # Reduce retrieval to speed up RAG
    top_k = int(os.getenv("RAG_TOP_K", "1"))
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, similarity_top_k=top_k)
    logger.info(f"Connecting to room {ctx.room.name}")
   
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
   
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")
    
    # Use Deepgram STT by default to avoid AssemblyAI model deprecation errors
    # Requires DEEPGRAM_API_KEY in your environment/.env
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
            voice="794f9389-aac1-45b6-b726-9d9369183238",
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