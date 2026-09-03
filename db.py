"""
SQLite persistence layer for the Teams-style multi-user RAG assistant.

Everything the app needs lives in one file: app.db. Stdlib sqlite3 only, so
this module can be imported by both the FastAPI server process and the
LiveKit agent process without pulling in FastAPI-specific dependencies.

Key differences from the original single-user prototype:
  * Real `users` with roles (developer | project_manager | stakeholder).
  * `groups` (one per project) with an explicit `group_members` join table -
    membership is enforced at the API layer, so retrieval and chat never leak
    across projects.
  * `messages` carry the real sender's user id + role, and an `is_question`
    flag (populated for stakeholder chat messages so the suggestion engine can
    fire on chat, not just on calls).
  * `transcripts` is a DEDICATED table (not folded into messages). Per §5 of
    the spec transcripts are stored ONLY as retrieval fodder and are never
    rendered in any UI, so they must not appear in the chat thread.
  * `embeddings` are strictly group-scoped (group_id NOT NULL). There is no
    global KB any more - each group has its own isolated index.
"""
import sqlite3
import time
import uuid
import struct
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "app.db"


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # server + agent both hit the db
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE,
                name          TEXT NOT NULL,
                role          TEXT NOT NULL,   -- developer | project_manager | stakeholder
                avatar        TEXT,            -- emoji or initials, kept trivial for the POC
                password_hash TEXT NOT NULL,
                created_at    REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS groups (
                id         TEXT PRIMARY KEY,    -- stable slug, e.g. grp_bank_chatbot
                name       TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS group_members (
                group_id TEXT NOT NULL,
                user_id  TEXT NOT NULL,
                PRIMARY KEY (group_id, user_id),
                FOREIGN KEY (group_id) REFERENCES groups(id),
                FOREIGN KEY (user_id)  REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id    TEXT NOT NULL,
                sender_id   TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                sender_role TEXT NOT NULL,
                content     TEXT NOT NULL,
                is_question INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL,
                FOREIGN KEY (group_id) REFERENCES groups(id)
            );

            CREATE TABLE IF NOT EXISTS attachments (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id       TEXT NOT NULL,
                message_id     INTEGER,        -- the chat message it rode in on (nullable)
                filename       TEXT NOT NULL,
                stored_path    TEXT NOT NULL,   -- relative path under ./attachments
                mime           TEXT NOT NULL,
                size_bytes     INTEGER NOT NULL,
                extracted_text TEXT,
                created_at     REAL NOT NULL,
                FOREIGN KEY (group_id) REFERENCES groups(id)
            );

            -- Every embedding row is scoped to exactly one group. No global KB.
            CREATE TABLE IF NOT EXISTS embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id    TEXT NOT NULL,
                source_type TEXT NOT NULL,   -- message | attachment | transcript | doc
                source_id   TEXT,
                chunk_text  TEXT NOT NULL,
                vector      BLOB NOT NULL,
                created_at  REAL NOT NULL,
                FOREIGN KEY (group_id) REFERENCES groups(id)
            );

            -- Transcripts: stored for retrieval only, never rendered (spec §5).
            CREATE TABLE IF NOT EXISTS transcripts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id     TEXT NOT NULL,
                call_id      TEXT NOT NULL,
                speaker_id   TEXT NOT NULL,
                speaker_name TEXT NOT NULL,
                speaker_role TEXT NOT NULL,
                text         TEXT NOT NULL,
                is_question  INTEGER NOT NULL DEFAULT 0,
                created_at   REAL NOT NULL,
                FOREIGN KEY (group_id) REFERENCES groups(id)
            );

            CREATE TABLE IF NOT EXISTS call_sessions (
                id           TEXT PRIMARY KEY,
                group_id     TEXT NOT NULL,
                room_name    TEXT NOT NULL,
                started_by   TEXT NOT NULL,
                status       TEXT NOT NULL,   -- ringing | live | ended
                initiated_at REAL NOT NULL,
                ended_at     REAL,
                FOREIGN KEY (group_id) REFERENCES groups(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_group    ON messages(group_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_group  ON embeddings(group_id);
            CREATE INDEX IF NOT EXISTS idx_transcripts_group ON transcripts(group_id);
            CREATE INDEX IF NOT EXISTS idx_members_user      ON group_members(user_id);
            """
        )


# --------------------------------------------------------------------------
# users
# --------------------------------------------------------------------------

def create_user(user_id, username, name, role, avatar, password_hash):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, username, name, role, avatar, password_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, username, name, role, avatar, password_hash, time.time()),
        )
    return {"id": user_id, "username": username, "name": name, "role": role, "avatar": avatar}


def get_user_by_username(username):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user(user_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def user_exists(username):
    with get_db() as conn:
        return conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone() is not None


def public_user(row):
    """Strip password_hash before sending a user object to a client."""
    if not row:
        return None
    return {"id": row["id"], "username": row["username"], "name": row["name"],
            "role": row["role"], "avatar": row["avatar"]}


# --------------------------------------------------------------------------
# groups + membership
# --------------------------------------------------------------------------

def create_group(group_id, name):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO groups (id, name, created_at) VALUES (?, ?, ?)",
            (group_id, name, time.time()),
        )
    return {"id": group_id, "name": name}


def group_exists(group_id):
    with get_db() as conn:
        return conn.execute("SELECT 1 FROM groups WHERE id = ?", (group_id,)).fetchone() is not None


def add_member(group_id, user_id):
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO group_members (group_id, user_id) VALUES (?, ?)",
            (group_id, user_id),
        )


def is_member(group_id, user_id):
    with get_db() as conn:
        return conn.execute(
            "SELECT 1 FROM group_members WHERE group_id = ? AND user_id = ?",
            (group_id, user_id),
        ).fetchone() is not None


def list_groups_for_user(user_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT g.* FROM groups g "
            "JOIN group_members m ON m.group_id = g.id "
            "WHERE m.user_id = ? ORDER BY g.created_at ASC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_group_members(group_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT u.* FROM users u "
            "JOIN group_members m ON m.user_id = u.id "
            "WHERE m.group_id = ? ORDER BY u.role, u.name",
            (group_id,),
        ).fetchall()
        return [public_user(dict(r)) for r in rows]


def get_group(group_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        return dict(row) if row else None


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------

def add_message(group_id, sender_id, sender_name, sender_role, content, is_question=False):
    ts = time.time()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO messages (group_id, sender_id, sender_name, sender_role, content, is_question, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (group_id, sender_id, sender_name, sender_role, content, int(is_question), ts),
        )
        msg_id = cur.lastrowid
    return {"id": msg_id, "group_id": group_id, "sender_id": sender_id,
            "sender_name": sender_name, "sender_role": sender_role,
            "content": content, "is_question": bool(is_question), "created_at": ts}


def list_messages(group_id, limit=1000):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE group_id = ? ORDER BY created_at ASC LIMIT ?",
            (group_id, limit),
        ).fetchall()
        msgs = [dict(r) for r in rows]
    # attach any files that rode in on each message
    by_msg = {}
    for a in list_attachments(group_id):
        by_msg.setdefault(a["message_id"], []).append(a)
    for m in msgs:
        m["is_question"] = bool(m["is_question"])
        m["attachments"] = by_msg.get(m["id"], [])
    return msgs


# --------------------------------------------------------------------------
# attachments
# --------------------------------------------------------------------------

def add_attachment(group_id, message_id, filename, stored_path, mime, size_bytes, extracted_text):
    ts = time.time()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO attachments (group_id, message_id, filename, stored_path, mime, size_bytes, extracted_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (group_id, message_id, filename, stored_path, mime, size_bytes, extracted_text, ts),
        )
        att_id = cur.lastrowid
    return {"id": att_id, "group_id": group_id, "message_id": message_id,
            "filename": filename, "stored_path": stored_path, "mime": mime,
            "size_bytes": size_bytes}


def list_attachments(group_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, group_id, message_id, filename, stored_path, mime, size_bytes, created_at "
            "FROM attachments WHERE group_id = ? ORDER BY created_at ASC",
            (group_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_attachment(att_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM attachments WHERE id = ?", (att_id,)).fetchone()
        return dict(row) if row else None


def list_attachments_with_text(group_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM attachments WHERE group_id = ? ORDER BY created_at ASC",
            (group_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# transcripts (retrieval-only, never rendered)
# --------------------------------------------------------------------------

def add_transcript(group_id, call_id, speaker_id, speaker_name, speaker_role, text, is_question=False):
    ts = time.time()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO transcripts (group_id, call_id, speaker_id, speaker_name, speaker_role, text, is_question, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (group_id, call_id, speaker_id, speaker_name, speaker_role, text, int(is_question), ts),
        )
        tid = cur.lastrowid
    return {"id": tid, "group_id": group_id, "call_id": call_id, "text": text}


def list_transcripts(group_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM transcripts WHERE group_id = ? ORDER BY created_at ASC",
            (group_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# embeddings (group-scoped, brute-force cosine at the rag layer)
# --------------------------------------------------------------------------

def _pack(vector):
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def add_embedding(group_id, source_type, source_id, chunk_text, vector):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO embeddings (group_id, source_type, source_id, chunk_text, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (group_id, source_type, source_id, chunk_text, _pack(vector), time.time()),
        )


def clear_group_embeddings(group_id):
    with get_db() as conn:
        conn.execute("DELETE FROM embeddings WHERE group_id = ?", (group_id,))


def group_embeddings_exist(group_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) c FROM embeddings WHERE group_id = ?", (group_id,)
        ).fetchone()
        return row["c"] > 0


def get_group_candidates(group_id):
    """Every embedded chunk for ONE group. Cross-group leakage is impossible
    here because the query is filtered to a single group_id."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM embeddings WHERE group_id = ?", (group_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["vector"] = _unpack(d["vector"])
        out.append(d)
    return out


# --------------------------------------------------------------------------
# call sessions
# --------------------------------------------------------------------------

def create_call_session(group_id, room_name, started_by, call_id=None):
    call_id = call_id or str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO call_sessions (id, group_id, room_name, started_by, status, initiated_at) "
            "VALUES (?, ?, ?, ?, 'ringing', ?)",
            (call_id, group_id, room_name, started_by, time.time()),
        )
    return {"id": call_id, "group_id": group_id, "room_name": room_name, "status": "ringing"}


def get_call_session(call_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM call_sessions WHERE id = ?", (call_id,)).fetchone()
        return dict(row) if row else None


def get_live_call_for_group(group_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM call_sessions WHERE group_id = ? AND status IN ('ringing','live') "
            "ORDER BY initiated_at DESC LIMIT 1",
            (group_id,),
        ).fetchone()
        return dict(row) if row else None


def set_call_status(call_id, status):
    with get_db() as conn:
        if status == "ended":
            conn.execute(
                "UPDATE call_sessions SET status = ?, ended_at = ? WHERE id = ?",
                (status, time.time(), call_id),
            )
        else:
            conn.execute("UPDATE call_sessions SET status = ? WHERE id = ?", (status, call_id))
