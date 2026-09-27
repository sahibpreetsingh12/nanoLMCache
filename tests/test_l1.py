"""
Unit tests for the L1 tier.

What has to hold before cache.py can trust it:
1. A stored buffer comes back byte-identical
2. Byte accounting matches what is actually held -- after put, overwrite, delete
3. A put that does not fit is refused, and refusing costs nothing
4. Entries are independent of the caller's tensor (L1 keeps its own copy)
5. A miss is None, not an exception

No engine, no pool, no block tables: L1 only ever sees keys and buffers, and the
tests stay on that side of the seam on purpose.
"""

import pytest
import torch

from l1 import L1Cache, buffer_bytes


# =============================================================================
# Helper Functions
# =============================================================================


def make_buffer(num_tokens=4, fill=1.0, layers=2, heads=2, head_dim=4):
    """A buffer shaped like gather()'s output: [layers, 2, tokens, heads, dim]."""
    return torch.full((layers, 2, num_tokens, heads, head_dim), fill)


# =============================================================================
# Round trip
# =============================================================================


def test_put_then_get_returns_identical_buffer():
    cache = L1Cache(capacity_bytes=1 << 20)
    buffer = torch.randn(2, 2, 4, 2, 4)

    assert cache.put(123, buffer) is True
    assert torch.equal(cache.get(123), buffer)


def test_get_missing_key_returns_none():
    cache = L1Cache(capacity_bytes=1 << 20)
    assert cache.get(999) is None
    assert cache.contains(999) is False


def test_contains_tracks_put_and_delete():
    cache = L1Cache(capacity_bytes=1 << 20)
    cache.put(7, make_buffer())

    assert cache.contains(7) is True
    assert cache.delete(7) is True
    assert cache.contains(7) is False
    assert cache.delete(7) is False  # already gone


# =============================================================================
# Byte accounting
# =============================================================================


def test_usage_tracks_contents():
    cache = L1Cache(capacity_bytes=1 << 20)
    one = make_buffer()
    size = buffer_bytes(one)

    assert cache.usage_bytes == 0
    cache.put(1, one)
    assert cache.usage_bytes == size
    cache.put(2, make_buffer())
    assert cache.usage_bytes == 2 * size
    cache.delete(1)
    assert cache.usage_bytes == size
    cache.clear()
    assert cache.usage_bytes == 0
    assert cache.num_entries == 0


def test_overwriting_a_key_does_not_double_count():
    """Same key, different size: the old bytes must come off the books."""
    cache = L1Cache(capacity_bytes=1 << 20)
    cache.put(1, make_buffer(num_tokens=8))
    big = make_buffer(num_tokens=4)
    cache.put(1, big)

    assert cache.num_entries == 1
    assert cache.usage_bytes == buffer_bytes(big)


def test_usage_ratio():
    buffer = make_buffer()
    cache = L1Cache(capacity_bytes=4 * buffer_bytes(buffer))
    cache.put(1, buffer)
    assert cache.usage_ratio == pytest.approx(0.25)


def test_dtype_changes_the_byte_cost():
    """Bytes, not elements: half the width is half the budget."""
    wide = torch.zeros(2, 2, 4, 2, 4, dtype=torch.float32)
    narrow = wide.to(torch.float16)
    assert buffer_bytes(narrow) * 2 == buffer_bytes(wide)


# =============================================================================
# Capacity
# =============================================================================


def test_put_refused_when_it_does_not_fit():
    buffer = make_buffer()
    cache = L1Cache(capacity_bytes=buffer_bytes(buffer))  # room for exactly one
    assert cache.put(1, buffer) is True
    assert cache.put(2, make_buffer(fill=2.0)) is False

    # A refusal changes nothing: no entry, no bytes, no eviction of the incumbent.
    assert cache.contains(2) is False
    assert cache.num_entries == 1
    assert cache.usage_bytes == buffer_bytes(buffer)
    assert torch.equal(cache.get(1), buffer)


def test_buffer_larger_than_capacity_is_refused():
    cache = L1Cache(capacity_bytes=16)
    assert cache.put(1, make_buffer(num_tokens=64)) is False
    assert cache.usage_bytes == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        L1Cache(capacity_bytes=0)


# =============================================================================
# Isolation
# =============================================================================


def test_entry_is_not_affected_by_later_writes_to_the_callers_buffer():
    """
    The caller's buffer is scratch space -- gather() may hand back the same
    tensor for the next chunk. A cache entry that aliased it would change
    content without changing key.
    """
    cache = L1Cache(capacity_bytes=1 << 20)
    buffer = make_buffer(fill=1.0)
    cache.put(1, buffer)

    buffer.fill_(99.0)

    assert torch.equal(cache.get(1), make_buffer(fill=1.0))


def test_keys_are_in_insertion_order():
    cache = L1Cache(capacity_bytes=1 << 20)
    for key in (30, 10, 20):
        cache.put(key, make_buffer())
    assert cache.keys() == [30, 10, 20]
