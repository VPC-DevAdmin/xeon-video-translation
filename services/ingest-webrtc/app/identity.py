from contextvars import ContextVar
from fastapi import HTTPException

owner = ContextVar("owner", default="local")


def require(session):
    if session is None or getattr(session, "owner", "local") != owner.get():
        raise HTTPException(404, "unknown session")
    return session
