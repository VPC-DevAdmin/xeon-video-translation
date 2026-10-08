"""Bounded lookahead overlaps network token generation with speech synthesis."""

import queue
import threading


def prefetch(iterator, capacity=2):
    buffer = queue.Queue(maxsize=capacity)
    stop = threading.Event()
    sentinel = object()

    def put(item):
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            for value in iterator:
                if stop.is_set():
                    break
                put(value)
        except BaseException as exc:
            put(exc)
        finally:
            try:
                iterator.close()
            except AttributeError:
                pass
            put(sentinel)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            value = buffer.get()
            if value is sentinel:
                return
            if isinstance(value, BaseException):
                raise value
            yield value
    finally:
        stop.set()
        # Network readers have a bounded socket timeout; do not block an event
        # loop waiting for it. This producer never touches the speech model.
        thread.join(timeout=0.2)
