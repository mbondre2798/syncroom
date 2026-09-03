"""
Call-assist agent (multi-party).

This agent never speaks on the call. For EVERY participant in the room
(developers, PM, stakeholder - any number of them), it:

  1. Subscribes to their audio track and transcribes it with Deepgram,
     tagging each utterance with that participant's identity/name/role
     (read from the LiveKit token attributes the server set).
  2. POSTs each finalized utterance to the FastAPI server at
     /calls/{call_id}/transcript. The SERVER owns everything downstream:
     it stores the transcript (retrieval-only, never rendered), embeds it,
     and - if a stakeholder asked a question - runs RAG and pushes a draft to
     the Project Manager's panel over the app websocket.

Why the agent no longer does intent/RAG/suggestion itself: routing a
question typed in chat and a question spoken on a call through ONE server-side
path (see server._maybe_suggest) means both behave identically and there's no
fragile "send a data message to a specific participant" targeting. The agent
is now a dumb, robust transcription pipe.

Transcripts are deliberately NOT published back into the room as bubbles -
per spec §5 they must not be visible in any UI. Only debug events are
broadcast (on 'call.debug'), and only when CALL_ASSIST_DEBUG is on.

SDK VERSION SENSITIVITY: this drives LiveKit's low-level STT streaming
(`rtc.AudioStream`, `stt.stream()`, `SpeechEventType`) directly because we
transcribe multiple independent humans at once rather than running one
bot<->user AgentSession. Method names match livekit-agents at time of
writing; if you hit an AttributeError, check the installed version - it's
almost always a renamed method, not a design flaw. Run `python agent.py dev`
and watch the logs first.
"""
import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer
from livekit.agents.stt import SpeechEventType
from livekit.plugins import deepgram

load_dotenv(".env")

SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://localhost:8000")
DEBUG = os.getenv("CALL_ASSIST_DEBUG", "true").lower() != "false"

server = AgentServer()


def _log(event, **fields):
    extras = " ".join(f"{k}={v!r}" for k, v in fields.items())
    print(f"[call-assist] {event} {extras}")


async def _publish_debug(room, event, **fields):
    _log(event, **fields)
    if not DEBUG:
        return
    payload = {"event": event, "ts": time.time(), **fields}
    try:
        await room.local_participant.send_text(json.dumps(payload), topic="call.debug")
    except Exception as e:
        _log("debug_publish_failed", error=str(e))


async def _post_transcript(call_id, speaker_id, speaker_name, speaker_role, text):
    """Hand the finalized line to the server. The server stores it, embeds it,
    and drives the PM suggestion panel if it's a stakeholder question."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            await client.post(
                f"{SERVER_BASE_URL}/calls/{call_id}/transcript",
                json={"speaker_id": speaker_id, "speaker_name": speaker_name,
                      "speaker_role": speaker_role, "text": text},
            )
    except Exception as e:
        _log("post_transcript_failed", error=str(e))


def _call_id_from_room(room_name: str) -> str:
    return room_name[len("call-"):] if room_name.startswith("call-") else room_name


@server.rtc_session(agent_name="call-assist-agent")
async def call_assist_agent(ctx: agents.JobContext):
    stt_provider = deepgram.STT(model="nova-3", language="multi")
    call_id = _call_id_from_room(ctx.room.name)
    _log("session_start", room=ctx.room.name, call_id=call_id)

    tasks: list[asyncio.Task] = []

    async def handle_participant_audio(track, participant):
        attrs = participant.attributes or {}
        speaker_role = attrs.get("role", "developer")
        speaker_name = attrs.get("name", participant.identity)
        speaker_id = attrs.get("user_id", participant.identity)

        await _publish_debug(ctx.room, "track_subscribed",
                             participant=participant.identity, role=speaker_role)

        stt_stream = stt_provider.stream()
        audio_stream = rtc.AudioStream(track)

        async def feed_audio():
            try:
                async for audio_event in audio_stream:
                    stt_stream.push_frame(audio_event.frame)
            finally:
                stt_stream.end_input()

        feed_task = asyncio.create_task(feed_audio())
        try:
            async for event in stt_stream:
                if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    interim = event.alternatives[0].text.strip() if event.alternatives else ""
                    if interim:
                        await _publish_debug(ctx.room, "stt_interim",
                                             speaker=speaker_name, text=interim)
                    continue
                if event.type != SpeechEventType.FINAL_TRANSCRIPT:
                    continue
                text = event.alternatives[0].text.strip()
                if not text:
                    continue

                await _publish_debug(ctx.room, "stt_final",
                                     speaker=speaker_name, role=speaker_role, text=text)
                # Hand off to the server. Everything else (store, embed,
                # intent-detect, retrieve, draft, deliver to PM) happens there.
                await _post_transcript(call_id, speaker_id, speaker_name, speaker_role, text)
        finally:
            feed_task.cancel()

    def on_participant_connected(participant):
        role = (participant.attributes or {}).get("role", "unknown")
        asyncio.create_task(
            _publish_debug(ctx.room, "participant_connected",
                           identity=participant.identity, role=role))

    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            tasks.append(asyncio.create_task(handle_participant_audio(track, participant)))

    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("track_subscribed", on_track_subscribed)

    await ctx.connect()
    _log("connected_to_room", room=ctx.room.name)

    stop_event = asyncio.Event()
    ctx.room.on("disconnected", lambda: stop_event.set())
    await stop_event.wait()

    for t in tasks:
        t.cancel()


if __name__ == "__main__":
    agents.cli.run_app(server)
