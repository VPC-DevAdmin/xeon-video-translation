"""Optional token accounts and internal service authentication."""

from contextvars import ContextVar
import hashlib
import hmac
import json
from fastapi import HTTPException
from .config import settings

principal = ContextVar("principal", default="local")


def authenticate(headers):
    internal = headers.get("x-internal-key", "")
    if settings.internal_api_key and hmac.compare_digest(internal, settings.internal_api_key):
        return headers.get("x-owner-id", "local")
    accounts = json.loads(settings.auth_tokens_json)
    if not accounts:
        return "local"
    value = headers.get("authorization", "")
    supplied = value[7:] if value.startswith("Bearer ") else ""
    digest = hashlib.sha256(supplied.encode()).digest()
    for token, owner in accounts.items():
        if hmac.compare_digest(digest, hashlib.sha256(token.encode()).digest()):
            return owner
    raise HTTPException(401, "authentication required")


def check_owner(meta):
    if meta and meta.get("owner_id", "local") != principal.get():
        raise HTTPException(404, "job not found")
