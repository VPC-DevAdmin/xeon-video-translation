"""Bounded in-process event history; durable snapshots live in meta.json."""

import asyncio
from collections import deque


class EventLog:
    def __init__(self):
        self.messages = deque(maxlen=256)
        self.changed = asyncio.Event()
        self.sequence = 0

    def put_nowait(self, message):
        self.sequence += 1
        self.messages.append((self.sequence, message))
        self.changed.set()

    async def put(self, message):
        self.put_nowait(message)

    async def subscribe(self, after=0):
        while True:
            self.changed.clear()
            pending = [(n, m) for n, m in self.messages if n > after]
            if pending:
                for n, message in pending:
                    after = n
                    yield n, message
                    if message["event"] == "stream_end":
                        return
            else:
                try:
                    await asyncio.wait_for(self.changed.wait(), 20)
                except asyncio.TimeoutError:
                    yield after, {"event": "ping", "data": {}}
