"""Independent SSE subscribers with replay and durable terminal snapshots."""

import json
from fastapi import APIRouter, HTTPException, Header
from sse_starlette.sse import EventSourceResponse
from ..pipeline.orchestrator import get_job, get_queue

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}/events")
async def job_events(job_id: str, last_event_id: str | None = Header(None)):
    try:
        state = get_job(job_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if state is None:
        raise HTTPException(404, "job not found")
    from ..security import check_owner

    check_owner(state.to_dict())
    log = get_queue(job_id)
    try:
        cursor = max(0, int(last_event_id or 0))
    except ValueError:
        cursor = 0

    if log is not None and cursor > log.sequence:
        cursor = 0  # A restart creates a new in-memory event sequence.

    async def events():
        yield {"event": "snapshot", "data": json.dumps(state.to_dict())}
        if log is None:
            yield {"event": "stream_end", "data": "{}"}
            return
        async for seq, msg in log.subscribe(cursor):
            yield {"id": str(seq), "event": msg["event"], "data": json.dumps(msg["data"])}

    return EventSourceResponse(events())
