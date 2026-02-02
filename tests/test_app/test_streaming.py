"""Tests for the async-to-sync streaming bridge."""

import asyncio
import pytest

from src.app.streaming import async_to_sync_generator


def test_async_to_sync_basic():
    """Test basic async to sync conversion."""
    async def async_gen():
        for i in range(5):
            yield i
            await asyncio.sleep(0.01)

    results = list(async_to_sync_generator(async_gen))
    assert results == [0, 1, 2, 3, 4]


def test_async_to_sync_empty():
    """Test empty async generator."""
    async def async_gen():
        return
        yield  # Make it a generator

    results = list(async_to_sync_generator(async_gen))
    assert results == []


def test_async_to_sync_single_item():
    """Test single item generator."""
    async def async_gen():
        yield "only"

    results = list(async_to_sync_generator(async_gen))
    assert results == ["only"]


def test_async_to_sync_error_propagation():
    """Test that errors are propagated from async to sync."""
    async def async_gen():
        yield 1
        yield 2
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        list(async_to_sync_generator(async_gen))


def test_async_to_sync_error_at_start():
    """Test error at the start of iteration."""
    async def async_gen():
        raise RuntimeError("Immediate error")
        yield  # Never reached

    with pytest.raises(RuntimeError, match="Immediate error"):
        list(async_to_sync_generator(async_gen))


def test_async_to_sync_mixed_types():
    """Test generator yielding different types."""
    async def async_gen():
        yield "text"
        yield 42
        yield {"key": "value"}
        yield [1, 2, 3]

    results = list(async_to_sync_generator(async_gen))
    assert results == ["text", 42, {"key": "value"}, [1, 2, 3]]


def test_async_to_sync_large_stream():
    """Test streaming a larger number of items."""
    count = 100

    async def async_gen():
        for i in range(count):
            yield i

    results = list(async_to_sync_generator(async_gen))
    assert len(results) == count
    assert results == list(range(count))
