import os
import sys
import subprocess


def ensure_requirements():
    try:
        import livekit  # noqa: F401
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
        )


def main():
    ensure_requirements()

    from livekit.agents import WorkerOptions, cli  # type: ignore
    from voice_agent_openai import entrypoint, prewarm  # imports after deps are ready

    room = os.getenv("ROOM", os.getenv("LIVEKIT_ROOM", "demo-room"))
    # Inject CLI args so livekit-agents connects directly to the room
    sys.argv = [sys.argv[0], "connect", "--room", room]

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )


if __name__ == "__main__":
    main()
