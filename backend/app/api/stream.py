"""Server-Sent Events stream of pipeline progress."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ..pipeline.orchestrator import get_job, get_queue

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}/events")
async def job_events(job_id: str):
    state = get_job(job_id)
    if state is None:
        raise HTTPException(404, "job not found")

    queue = get_queue(job_id)
    if queue is None:
        # Job is no longer live (already finished and dropped from memory). Replay
        # one synthetic event so clients connecting late still see the final state.
        async def replay():
            yield {"event": "job_completed" if state.status == "completed" else "error",
                   "data": json.dumps(state.to_dict())}
        return EventSourceResponse(replay())

    async def event_gen():
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                # Heartbeat to keep the connection alive through proxies.
                yield {"event": "ping", "data": "{}"}
                continue
            event = msg.get("event", "message")
            data = json.dumps(msg.get("data", {}), default=str)
            yield {"event": event, "data": data}
            if event in ("job_completed", "stream_end", "error"):
                break

    return EventSourceResponse(event_gen())
