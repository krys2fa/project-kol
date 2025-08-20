import os
import sys
from livekit.agents import WorkerOptions, cli
from voice_agent_openai import entrypoint, prewarm


def main():
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
