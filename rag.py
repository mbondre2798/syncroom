"""
RAG pipeline - now strictly per-group.

Every index is scoped to a single group_id. There is no global knowledge
base and no cross-group retrieval: `search()` only ever looks at chunks whose
group_id matches, so project A can never surface project B's content.

Sources unified per group (spec §5):
  1. group chat history          -> source_type 'message'
  2. past + live call transcripts -> source_type 'transcript'
  3. in-chat uploaded documents   -> source_type 'attachment'
  4. docs in ./kb_documents/<group_id>/ (dropped outside the chat UI)
                                   -> source_type 'doc'

Embeddings: local sentence-transformers (BAAI/bge-small-en-v1.5 by default),
CPU is fine for a few hundred short chunks per group. Vector search is
brute-force cosine in Python - deliberately no vector DB. At the stated scale
that's a few hundred vectors; if a group's index grows into the thousands,
replace get_group_candidates()/search() with pgvector or Chroma without
touching callers.

LLM (intent classification + suggestion drafting) is Groq, lazily
constructed so the whole app can boot and be tested without a GROQ_API_KEY -
you only pay for the key when you actually classify/generate.
"""
import os
import math
import mimetypes
import threading
from pathlib import Path

from dotenv import load_dotenv

import db

load_dotenv(".env")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
GROQ_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
KB_ROOT = Path(__file__).parent / "kb_documents"   # ./kb_documents/<group_id>/*

# Keep the downloaded model files inside the project instead of the user's
# home directory cache, so the whole app (code + model) is self-contained
# and portable. Override with MODEL_CACHE_DIR if you want it elsewhere.
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(Path(__file__).parent / "model_cache"))

_embedder = None
_embedder_lock = threading.Lock()
_groq_client = None


# --------------------------------------------------------------------------
# lazy clients (so import / boot / tests don't require model download or keys)
# --------------------------------------------------------------------------

def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    # Lock so concurrent callers (chat message + call transcript arriving at
    # the same time) don't each independently load their own copy of the
    # model - only the first one loads it, the rest wait and reuse it.
    with _embedder_lock:
        if _embedder is None:
            # Optional override for local testing without downloading torch:
            # set RAG_FAKE_EMBEDDER=1 to use a deterministic hash-based stub.
            if os.getenv("RAG_FAKE_EMBEDDER") == "1":
                _embedder = _FakeEmbedder()
            else:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=MODEL_CACHE_DIR)
    return _embedder


def warm_up():
    """Force the embedder to load now instead of on the first real request.
    Call this once at server startup (see server.py's lifespan) so the ~1-3s
    model-load cost happens during boot, not during someone's first live
    chat message or call transcript."""
    _get_embedder()




def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


class _FakeEmbedder:
    """Deterministic 64-dim bag-of-hashed-tokens vector. Only for offline
    tests of the plumbing (chunking, storage, per-group isolation, ranking).
    Not semantically meaningful - do NOT use in real runs."""
    DIM = 64

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for tok in t.lower().split():
                v[hash(tok) % self.DIM] += 1.0
            if normalize_embeddings:
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                v = [x / n for x in v]
            out.append(_Vec(v))
        return out


class _Vec(list):
    def tolist(self):
        return list(self)


def embed_texts(texts):
    if not texts:
        return []
    vectors = _get_embedder().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def chunk_text(text, chunk_size=800, overlap=120):
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# reading root-folder docs
# --------------------------------------------------------------------------

def _read_doc(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(path.read_bytes()))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# --------------------------------------------------------------------------
# building a group's index (all four sources)
# --------------------------------------------------------------------------

def build_group_index(group_id: str):
    """(Re)build the entire per-group vector index from scratch: chat history,
    attachments, transcripts, and root-folder docs. Idempotent - clears first
    so repeated calls don't accumulate stale duplicate chunks."""
    db.clear_group_embeddings(group_id)

    pending = []   # (source_type, source_id, chunk_text)

    # 1. chat history -> rolling ~800-char blocks (more coherent than 1/msg)
    buffer = ""
    for m in db.list_messages(group_id):
        line = f"{m['sender_name']} ({m['sender_role']}): {m['content']}\n"
        if len(buffer) + len(line) > 800 and buffer:
            pending.append(("message", "chat", buffer))
            buffer = ""
        buffer += line
    if buffer:
        pending.append(("message", "chat", buffer))

    # 2. in-chat attachments
    for att in db.list_attachments_with_text(group_id):
        if att.get("extracted_text"):
            for ch in chunk_text(att["extracted_text"]):
                pending.append(("attachment", att["filename"], ch))

    # 3. transcripts (past + whatever's been persisted from a live call)
    buffer = ""
    for t in db.list_transcripts(group_id):
        line = f"{t['speaker_name']} ({t['speaker_role']}): {t['text']}\n"
        if len(buffer) + len(line) > 800 and buffer:
            pending.append(("transcript", t["call_id"], buffer))
            buffer = ""
        buffer += line
    if buffer:
        pending.append(("transcript", "call", buffer))

    # 4. root-folder docs: ./kb_documents/<group_id>/*
    group_dir = KB_ROOT / group_id
    if group_dir.is_dir():
        for path in sorted(group_dir.glob("**/*")):
            if path.is_file() and path.suffix.lower() in (".md", ".txt", ".pdf"):
                text = _read_doc(path)
                for ch in chunk_text(text):
                    pending.append(("doc", path.name, ch))

    if not pending:
        return 0

    vectors = embed_texts([p[2] for p in pending])
    for (source_type, source_id, chunk), vector in zip(pending, vectors):
        db.add_embedding(group_id, source_type, source_id, chunk, vector)
    return len(pending)


def add_single_text(group_id: str, source_type: str, source_id: str, text: str):
    """Incrementally embed one new item (a chat message, a freshly finalized
    transcript line, or a just-uploaded doc) without rebuilding the whole
    group index. Used on the live path so retrieval reflects what was just
    said, not only what existed at index-build time."""
    for ch in chunk_text(text):
        vec = embed_texts([ch])[0]
        db.add_embedding(group_id, source_type, source_id, ch, vec)


# --------------------------------------------------------------------------
# querying
# --------------------------------------------------------------------------

def search(query: str, group_id: str, top_k: int = 5):
    candidates = db.get_group_candidates(group_id)
    if not candidates:
        return []
    qv = embed_texts([query])[0]
    scored = [{**c, "score": _cosine(qv, c["vector"])} for c in candidates]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def classify_intent(text: str) -> bool:
    """True if `text` is a question/request that expects an answer, False for
    statements/greetings/narration. One cheap, temperature-0 LLM call.

    NOTE: GROQ_MODEL defaults to a reasoning model (gpt-oss-120b), which can
    emit hidden/visible reasoning tokens before the final word. max_tokens is
    kept generous (not 1-5) so the reasoning has room to finish and the
    actual QUESTION/STATEMENT token isn't cut off, and reasoning_effort is
    set to "low" to keep that preamble short. Matching is a substring search
    over the whole response rather than a strict startswith, so it still
    works even if some reasoning text precedes the verdict.
    """
    _messages = [
        {"role": "system", "content": (
            "Classify the sentence as QUESTION or STATEMENT. QUESTION means the "
            "speaker is asking something or requesting information/action that "
            "expects an answer. STATEMENT means they are just remarking, greeting, "
            "or narrating - nothing to answer. Reply with exactly one word: "
            "QUESTION or STATEMENT."
        )},
        {"role": "user", "content": text},
    ]
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL, messages=_messages,
            max_tokens=200, temperature=0, reasoning_effort="low",
        )
    except TypeError:
        # SDK/model combo doesn't accept reasoning_effort - retry without it.
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL, messages=_messages, max_tokens=200, temperature=0,
        )
    raw = resp.choices[0].message.content or ""
    # TEMP DEBUG: remove once classification is confirmed fixed.
    print(f"[classify_intent] DEBUG raw_model_output={raw!r} finish_reason={resp.choices[0].finish_reason!r}")
    upper = raw.strip().upper()
    # Reasoning models often restate both words while explaining themselves
    # before giving the verdict, so if both appear, trust whichever comes LAST.
    if "QUESTION" in upper and "STATEMENT" in upper:
        is_question = upper.rfind("QUESTION") > upper.rfind("STATEMENT")
    elif "QUESTION" in upper:
        is_question = True
    elif "STATEMENT" in upper:
        is_question = False
    else:
        is_question = False  # couldn't tell; default to not interrupting the PM
    return is_question


def generate_suggestion(question: str, context_chunks) -> str:
    context_block = "\n\n---\n\n".join(c["chunk_text"] for c in context_chunks) \
        or "(no relevant context found)"
    resp = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are assisting a Project Manager. A stakeholder just asked the "
                "question below in this project's group. Using ONLY the provided "
                "context (the group's chat history, call transcripts, and documents), "
                "draft a short, direct answer the PM can read and say/send immediately. "
                "If the context doesn't cover it, say so plainly - do not invent facts."
            )},
            {"role": "user", "content": (
                f"Context:\n{context_block}\n\nQuestion just asked: {question}\n\nSuggested answer:"
            )},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()