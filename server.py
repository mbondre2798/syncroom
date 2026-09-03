"""
FastAPI server for the Teams-style multi-user RAG assistant.

Responsibilities:
  * Auth (login/logout/me) backed by auth.py.
  * Groups: list-my-groups, members, messages - all membership-enforced so a
    user only ever sees projects they belong to.
  * Real-time group chat over a single authenticated websocket (/ws). New
    messages are pushed to every connected member of that group.
  * Attachments: upload (stored on disk + text extracted), list, and serve
    for inline previews.
  * Calls: start / join / end, minting LiveKit tokens tagged with the user's
    real role. Multi-party - any number of group members can join.
  * Transcript ingestion from the agent: /calls/{id}/transcript. Transcripts
    are stored for retrieval ONLY (never returned to any chat UI).
  * The suggestion engine lives HERE, not in the agent: any stakeholder
    QUESTION - whether typed in chat or spoken on a call - runs intent
    detection -> retrieval -> generation, and the draft is pushed only to the
    Project Manager(s) of that group over the same /ws channel. One code path
    for both chat and voice (spec §6).
"""
import asyncio
import json
import os
import uuid
import mimetypes
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import (FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect,
                     HTTPException, Header, Depends, Query)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import auth
import rag
import seed

load_dotenv(".env")

AGENT_NAME = "call-assist-agent"     # must match agent.py's rtc_session agent_name
ATTACH_DIR = Path(__file__).parent / "attachments"
ATTACH_DIR.mkdir(exist_ok=True)

# LiveKit is only needed when a call is actually started, so import lazily.
_lk_api = None
def _livekit():
    global _lk_api
    if _lk_api is None:
        from livekit import api as lk_api
        _lk_api = lk_api
    return _lk_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    seed.run()                       # idempotent; seeds users/groups/docs/index on first boot
    # Load the embedding model now (blocking call in a thread) so the ~1-3s
    # cost happens once at boot, not on the first live chat message or call
    # transcript. Safe to skip silently if it fails - it'll just lazy-load
    # on first use instead, same as before.
    try:
        await asyncio.to_thread(rag.warm_up)
    except Exception as e:
        print(f"[startup] embedder warm-up skipped ({type(e).__name__}: {e})")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ==========================================================================
# auth plumbing
# ==========================================================================

def current_user(authorization: str | None = Header(default=None)) -> dict:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    uid = auth.user_id_for_token(token)
    if not uid:
        raise HTTPException(401, "not authenticated")
    user = db.get_user(uid)
    if not user:
        raise HTTPException(401, "unknown user")
    return user


def require_membership(group_id: str, user: dict):
    if not db.group_exists(group_id):
        raise HTTPException(404, "group not found")
    if not db.is_member(group_id, user["id"]):
        raise HTTPException(403, "not a member of this group")


class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(body: LoginBody):
    u = db.get_user_by_username(body.username)
    if not u or not auth.verify_password(body.password, u["password_hash"]):
        raise HTTPException(401, "invalid username or password")
    token = auth.issue_token(u["id"])
    return {"token": token, "user": db.public_user(u)}


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        auth.revoke_token(authorization[7:])
    return {"ok": True}


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return db.public_user(user)


# ==========================================================================
# websocket hub: authenticated, role-aware, group-scoped delivery
# ==========================================================================

class Hub:
    def __init__(self):
        # websocket -> {"user_id", "role"}
        self.conns: dict[WebSocket, dict] = {}

    async def connect(self, ws: WebSocket, user: dict):
        await ws.accept()
        self.conns[ws] = {"user_id": user["id"], "role": user["role"]}

    def disconnect(self, ws: WebSocket):
        self.conns.pop(ws, None)

    async def _send(self, ws: WebSocket, message: dict):
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            self.disconnect(ws)

    async def to_group(self, group_id: str, message: dict, roles: set[str] | None = None):
        """Deliver to every connected member of `group_id`. If `roles` is given,
        only members whose role is in that set (used to keep suggestions
        PM-only)."""
        member_ids = {m["id"] for m in db.list_group_members(group_id)}
        for ws, meta in list(self.conns.items()):
            if meta["user_id"] in member_ids and (roles is None or meta["role"] in roles):
                await self._send(ws, message)


hub = Hub()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, token: str = Query(default="")):
    uid = auth.user_id_for_token(token)
    user = db.get_user(uid) if uid else None
    if not user:
        await websocket.close(code=4401)
        return
    await hub.connect(websocket, user)
    try:
        while True:
            await websocket.receive_text()   # keepalive; we don't expect inbound payloads
    except WebSocketDisconnect:
        hub.disconnect(websocket)


# ==========================================================================
# groups
# ==========================================================================

@app.get("/groups")
def list_groups(user: dict = Depends(current_user)):
    return db.list_groups_for_user(user["id"])


@app.get("/groups/{group_id}/members")
def group_members(group_id: str, user: dict = Depends(current_user)):
    require_membership(group_id, user)
    return db.list_group_members(group_id)


@app.get("/groups/{group_id}/messages")
def group_messages(group_id: str, user: dict = Depends(current_user)):
    require_membership(group_id, user)
    return db.list_messages(group_id)


class MessageBody(BaseModel):
    content: str


@app.post("/groups/{group_id}/messages")
async def post_message(group_id: str, body: MessageBody, user: dict = Depends(current_user)):
    require_membership(group_id, user)
    text = body.content.strip()
    if not text:
        raise HTTPException(400, "empty message")

    msg = db.add_message(group_id, user["id"], user["name"], user["role"], text)
    msg["attachments"] = []

    # incrementally embed the new line so retrieval reflects it immediately
    await asyncio.to_thread(rag.add_single_text, group_id, "message", "chat", f"{user['name']}: {text}")

    # broadcast to every connected member of this group (real-time chat)
    await hub.to_group(group_id, {"type": "message", "group_id": group_id, "message": msg})

    # suggestion engine: stakeholder questions (typed) drive the PM's panel
    if user["role"] == "stakeholder":
        asyncio.create_task(_maybe_suggest(group_id, text, user["name"], msg_id=msg["id"]))

    return msg


# ==========================================================================
# suggestion engine (shared by chat + call transcripts)
# ==========================================================================

async def _maybe_suggest(group_id: str, text: str, asker_name: str, msg_id=None):
    """Run intent detection; on a QUESTION, retrieve + draft and push the
    suggestion to the group's Project Manager(s) only. Failures here (e.g. no
    GROQ_API_KEY configured) are swallowed so they never break chat/calls."""
    # TEMP DEBUG: remove once the missing-suggestion issue is diagnosed.
    print(f"[suggest] DEBUG start group_id={group_id!r} asker={asker_name!r} text={text!r}")
    try:
        is_question = await asyncio.to_thread(rag.classify_intent, text)
        print(f"[suggest] DEBUG is_question={is_question}")
        if msg_id is not None and is_question:
            with db.get_db() as conn:
                conn.execute("UPDATE messages SET is_question = 1 WHERE id = ?", (msg_id,))
        if not is_question:
            print("[suggest] DEBUG classified as STATEMENT -> not running RAG, stopping here")
            return
        chunks = await asyncio.to_thread(rag.search, text, group_id)
        print(f"[suggest] DEBUG retrieved {len(chunks)} chunks: "
              f"{[(c['source_type'], c['source_id'], round(c['score'], 3)) for c in chunks]}")
        suggestion = await asyncio.to_thread(rag.generate_suggestion, text, chunks)
        print(f"[suggest] DEBUG generated suggestion ({len(suggestion)} chars): {suggestion[:200]!r}")
        member_ids = {m["id"] for m in db.list_group_members(group_id)}
        connected_pms = [meta["user_id"] for meta in hub.conns.values()
                         if meta["role"] == "project_manager" and meta["user_id"] in member_ids]
        print(f"[suggest] DEBUG connected PM sockets for this group: {connected_pms}")
        await hub.to_group(
            group_id,
            {"type": "suggestion", "group_id": group_id,
             "question": text, "asker": asker_name, "suggestion": suggestion},
            roles={"project_manager"},   # PM-only, per the role matrix
        )
        print("[suggest] DEBUG suggestion sent to hub.to_group")
    except Exception as e:
        import traceback
        print(f"[suggest] skipped ({type(e).__name__}: {e})")
        traceback.print_exc()


# ==========================================================================
# attachments
# ==========================================================================

def _extract_text(raw: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""
    # treat everything else as text; binary formats degrade to empty-ish text
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


@app.post("/groups/{group_id}/attachments")
async def upload_attachment(group_id: str, file: UploadFile = File(...),
                            caption: str = Form(default=""),
                            user: dict = Depends(current_user)):
    require_membership(group_id, user)
    raw = await file.read()
    name = file.filename or "upload"
    mime = file.content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"

    # store on disk under ./attachments/<group_id>/<uuid>_<name>
    gdir = ATTACH_DIR / group_id
    gdir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{name}"
    (gdir / stored_name).write_bytes(raw)
    rel_path = f"{group_id}/{stored_name}"

    extracted = _extract_text(raw, name)

    # a file share is also a chat message, so it renders inline in the thread
    content = caption.strip() or f"📎 {name}"
    msg = db.add_message(group_id, user["id"], user["name"], user["role"], content)
    att = db.add_attachment(group_id, msg["id"], name, rel_path, mime, len(raw), extracted)

    # embed the document text into the group's index right away
    if extracted:
        await asyncio.to_thread(rag.add_single_text, group_id, "attachment", name, extracted)

    msg["attachments"] = [att]
    await hub.to_group(group_id, {"type": "message", "group_id": group_id, "message": msg})
    return msg


@app.get("/attachments/{att_id}")
def get_attachment_file(att_id: int, user: dict = Depends(current_user)):
    att = db.get_attachment(att_id)
    if not att:
        raise HTTPException(404, "attachment not found")
    require_membership(att["group_id"], user)
    path = ATTACH_DIR / att["stored_path"]
    if not path.exists():
        raise HTTPException(404, "file missing on disk")
    return FileResponse(path, media_type=att["mime"], filename=att["filename"])


@app.post("/groups/{group_id}/reindex")
async def reindex_group(group_id: str, user: dict = Depends(current_user)):
    """Rebuild the whole group index - picks up any docs dropped into
    ./kb_documents/<group_id>/ outside the chat UI (spec §5 root folder)."""
    require_membership(group_id, user)
    n = await asyncio.to_thread(rag.build_group_index, group_id)
    return {"group_id": group_id, "chunks": n}


# ==========================================================================
# calls (multi-party, Teams-style)
# ==========================================================================

def _make_token(user: dict, room_name: str) -> str:
    api = _livekit()
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(user["id"])
        .with_name(user["name"])
        .with_attributes({"role": user["role"], "user_id": user["id"], "name": user["name"]})
        .with_grants(api.VideoGrants(
            room_join=True, room=room_name,
            can_publish=True, can_subscribe=True, can_publish_data=True,
        ))
        .with_room_config(api.RoomConfiguration(
            agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]
        ))
        .with_ttl(timedelta(hours=2))
        .to_jwt()
    )


@app.post("/groups/{group_id}/calls/start")
async def start_call(group_id: str, user: dict = Depends(current_user)):
    require_membership(group_id, user)

    existing = db.get_live_call_for_group(group_id)
    if existing:
        # someone already started one; just hand this user a token to join it
        token = _make_token(user, existing["room_name"])
        return {"call_id": existing["id"], "room_name": existing["room_name"],
                "token": token, "url": os.environ["LIVEKIT_URL"], "joined_existing": True}

    call_id = str(uuid.uuid4())
    room_name = f"call-{call_id}"
    call = db.create_call_session(group_id, room_name, started_by=user["id"], call_id=call_id)
    db.set_call_status(call_id, "live")

    token = _make_token(user, room_name)

    # ring every other member of the group
    await hub.to_group(group_id, {
        "type": "incoming_call", "group_id": group_id, "call_id": call_id,
        "room_name": room_name, "started_by": user["name"],
    })
    return {"call_id": call_id, "room_name": room_name, "token": token,
            "url": os.environ["LIVEKIT_URL"], "joined_existing": False}


@app.post("/calls/{call_id}/join")
def join_call(call_id: str, user: dict = Depends(current_user)):
    call = db.get_call_session(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    require_membership(call["group_id"], user)
    token = _make_token(user, call["room_name"])
    return {"call_id": call_id, "room_name": call["room_name"], "token": token,
            "url": os.environ["LIVEKIT_URL"]}


@app.get("/calls/{call_id}")
def get_call(call_id: str):
    """Used by the agent to resolve which group a room belongs to. No auth:
    the agent is a trusted server-side process on the same host."""
    call = db.get_call_session(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    return call


@app.post("/calls/{call_id}/end")
async def end_call(call_id: str, user: dict = Depends(current_user)):
    call = db.get_call_session(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    require_membership(call["group_id"], user)
    db.set_call_status(call_id, "ended")
    await hub.to_group(call["group_id"], {"type": "call_ended", "call_id": call_id,
                                          "group_id": call["group_id"]})
    return {"status": "ended"}


# ==========================================================================
# transcript ingestion from the agent (retrieval-only; never rendered)
# ==========================================================================

class TranscriptLine(BaseModel):
    speaker_id: str
    speaker_name: str
    speaker_role: str   # developer | project_manager | stakeholder
    text: str


@app.post("/calls/{call_id}/transcript")
async def ingest_transcript(call_id: str, line: TranscriptLine):
    """The agent posts each finalized utterance here (server-to-server, no
    user auth). We store it for retrieval, embed it incrementally, and - if a
    stakeholder asked a question - drive the PM suggestion panel. The line is
    NEVER returned to any chat UI (spec §5)."""
    call = db.get_call_session(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    group_id = call["group_id"]

    text = line.text.strip()
    if not text:
        return {"ok": True}

    db.add_transcript(group_id, call_id, line.speaker_id, line.speaker_name,
                      line.speaker_role, text)
    await asyncio.to_thread(rag.add_single_text, group_id, "transcript", call_id,
                            f"{line.speaker_name} ({line.speaker_role}): {text}")

    if line.speaker_role == "stakeholder":
        asyncio.create_task(_maybe_suggest(group_id, text, line.speaker_name))

    return {"ok": True}


# frontend (served last so it doesn't shadow the API routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")