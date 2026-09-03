"""
Minimal auth for the prototype.

Deliberately dependency-free (stdlib hashlib/hmac/secrets). This is NOT
production auth - no rate limiting, no rotation, sessions are in-memory and
die with the process. It's enough to give each device a real identity/role
from a login instead of the old "Device role" dropdown, which is what the
spec's role matrix needs to mean anything.

Swap `_SESSIONS` for Redis / signed JWTs and PBKDF2 params up before this
goes anywhere near real users.
"""
import hashlib
import hmac
import secrets

_PBKDF2_ROUNDS = 120_000
_SESSIONS: dict[str, str] = {}   # token -> user_id (in-memory; resets on restart)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt, digest = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def issue_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = user_id
    return token


def user_id_for_token(token: str | None) -> str | None:
    if not token:
        return None
    return _SESSIONS.get(token)


def revoke_token(token: str) -> None:
    _SESSIONS.pop(token, None)
