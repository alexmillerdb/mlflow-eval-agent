"""Async-to-sync streaming bridge for Streamlit.

Streamlit's st.write_stream() requires a synchronous generator, but our agent
uses async iterators. This module provides a thread-based bridge to convert
between the two.

Key design decisions:
- Queue-based communication between threads
- New event loop per thread (critical for avoiding "Event loop is closed" errors)
- Error propagation via sentinel tuple
"""

import asyncio
import queue
import threading
from typing import AsyncIterator, Callable, Generator, TypeVar

T = TypeVar("T")

# Sentinel to signal completion
_DONE = object()


def async_to_sync_generator(
    async_gen_factory: Callable[[], AsyncIterator[T]],
) -> Generator[T, None, None]:
    """Convert an async generator factory to a sync generator.

    Args:
        async_gen_factory: A callable that returns an async iterator.
                          Must be a factory (not the iterator itself) because
                          we need to create a fresh iterator in the worker thread.

    Yields:
        Items from the async iterator.

    Raises:
        Any exception raised by the async iterator.

    Example:
        async def my_async_gen():
            for i in range(3):
                yield i
                await asyncio.sleep(0.1)

        for item in async_to_sync_generator(my_async_gen):
            print(item)
    """
    result_queue: queue.Queue = queue.Queue()

    def worker():
        """Run async iterator in a new event loop on a separate thread."""
        # Critical: create a new event loop for this thread
        # This avoids "Event loop is closed" errors when Streamlit reruns
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def consume():
            try:
                async_gen = async_gen_factory()
                async for item in async_gen:
                    result_queue.put(item)
            except Exception as e:
                # Propagate error to main thread via sentinel tuple
                result_queue.put(("__error__", e))
            finally:
                result_queue.put(_DONE)

        try:
            loop.run_until_complete(consume())
        finally:
            loop.close()

    # Start worker thread
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Yield results from queue
    while True:
        item = result_queue.get()

        if item is _DONE:
            break

        # Check for error sentinel
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
            raise item[1]

        yield item

    # Wait for thread to finish
    thread.join(timeout=5.0)
