"""Priority between inference calls; running native work is never preempted."""

import asyncio
import time
import itertools
from contextlib import asynccontextmanager


class SpeechScheduler:
    def __init__(self):
        self.busy = False
        self.waiters = []
        self.order = itertools.count()

    def _release(self):
        while self.waiters:
            now = time.monotonic()
            # Waiting 30 seconds gains one priority level; batch cannot starve.
            self.waiters.sort(key=lambda item: (item[0] - (now - item[3]) / 30, item[1]))
            _, _, future, _ = self.waiters.pop(0)
            if not future.done():
                future.set_result(None)
                return
        self.busy = False

    @asynccontextmanager
    async def lease(self, priority=10):
        if self.busy:
            future = asyncio.get_running_loop().create_future()
            self.waiters.append((priority, next(self.order), future, time.monotonic()))
            try:
                await future
            except asyncio.CancelledError:
                if future.done() and not future.cancelled():
                    self._release()
                else:
                    future.cancel()
                raise
        else:
            self.busy = True
        try:
            yield
        finally:
            self._release()
