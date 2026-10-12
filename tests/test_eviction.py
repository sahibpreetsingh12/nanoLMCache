"""
Tests for Stage 4: eviction.

What has to hold:
1. LRUPolicy orders by last USE, not by insertion        (already passes)
2. The controller acts only at the watermark              WHEN
3. It frees ratio * capacity, coldest first, then stops   HOW MUCH + WHICH
4. It never credits bytes it did not free                 stale keys
5. Wired into KVCache, a hit protects an entry            the whole point

The controller tests use uint8 buffers so every entry's size in bytes is just
its length -- the numbers match the worked example in controller.py.

The rule from the design doc: predict which key dies BEFORE running, then
check. Every expected victim list below was written down first.
"""

import math

import pytest
import torch

from cache import KVCache
from controller import EvictionController
from engine import BlockAllocator, make_pool
from eviction import LRUPolicy
from l1 import L1Cache


# =============================================================================
# Helper Functions
# =============================================================================


def fill(l1: L1Cache, policy: LRUPolicy, sizes: dict[int, int]) -> None:
    """Put one buffer of exactly `n` bytes per key, touching in dict order."""
    for key, n in sizes.items():
        assert l1.put(key, torch.zeros(n, dtype=torch.uint8))
        policy.touch(key)


def setup(sizes: dict[int, int], capacity=1000, watermark=0.9, ratio=0.2):
    l1 = L1Cache(capacity_bytes=capacity)
    policy = LRUPolicy()
    fill(l1, policy, sizes)
    return l1, policy, EvictionController(l1, policy, watermark, ratio)


# The worked example from eviction.py / controller.py: 950 of 1000 bytes.
EXAMPLE = {3: 80, 7: 150, 1: 300, 9: 420}


# =============================================================================
# LRUPolicy (implemented already -- these should pass now)
# =============================================================================


def test_policy_orders_by_last_use_not_insertion():
    p = LRUPolicy()
    for k in (100, 200, 300):
        p.touch(k)
    p.touch(100)  # reused: a dict would leave it first
    assert p.coldest() == [200, 300, 100]


def test_policy_forget_unknown_key_is_silent():
    p = LRUPolicy()
    p.touch(1)
    p.forget(999)
    assert p.coldest() == [1]


def test_policy_coldest_is_a_snapshot():
    # The controller forgets keys while walking the ranking. That must not
    # raise "OrderedDict mutated during iteration".
    p = LRUPolicy()
    for k in (1, 2, 3):
        p.touch(k)
    for k in p.coldest():
        p.forget(k)
    assert p.coldest() == []


# =============================================================================
# WHEN
# =============================================================================


def test_no_eviction_below_watermark():
    l1, policy, ctl = setup({1: 899})  # 0.899
    assert ctl.should_evict() is False
    assert ctl.maybe_evict() == []
    assert l1.contains(1)


def test_eviction_at_watermark():
    _, _, ctl = setup({1: 900})  # exactly 0.9 -- the >= decision
    assert ctl.should_evict() is True


# =============================================================================
# HOW MUCH
# =============================================================================


def test_budget_is_a_fraction_of_capacity_not_usage():
    _, _, ctl = setup({1: 950})
    assert ctl.bytes_to_free() == 200
    _, _, ctl = setup({1: 100})
    assert ctl.bytes_to_free() == 200  # same budget, whatever the usage


# =============================================================================
# WHICH, and stopping
# =============================================================================


def test_worked_example():
    # Predicted: k3 (80) leaves, 120 to go; k7 (150) leaves, budget met.
    l1, _, ctl = setup(EXAMPLE)
    assert ctl.evict() == [3, 7]
    assert l1.usage_bytes == 950 - 80 - 150
    assert l1.contains(1) and l1.contains(9)


def test_a_hit_changes_the_victims():
    # Same cache, but k3 was just used. Predicted: k7 (150), then k1 (300).
    l1, policy, ctl = setup(EXAMPLE)
    policy.touch(3)
    assert ctl.evict() == [7, 1]
    assert l1.contains(3)


def test_evicted_keys_leave_the_policy():
    _, policy, ctl = setup(EXAMPLE)
    ctl.evict()
    assert policy.coldest() == [1, 9]


def test_stale_key_is_not_credited():
    # k3 leaves L1 behind the policy's back. If the controller counted its 80
    # bytes, it would stop after k7 having really freed only 150.
    l1, _, ctl = setup(EXAMPLE)
    l1.delete(3)
    evicted = ctl.evict()
    assert 3 not in evicted
    assert evicted == [7, 1]
    assert ctl.bytes_evicted == 450


def test_stops_when_the_ranking_runs_out():
    # Budget 1000, only 950 held: evict everything, then stop -- no loop, no error.
    l1, policy, ctl = setup(EXAMPLE, ratio=1.0)
    assert ctl.evict() == [3, 7, 1, 9]
    assert l1.num_entries == 0
    assert policy.coldest() == []


def test_stats_accumulate():
    _, _, ctl = setup(EXAMPLE)
    ctl.evict()  # k3 + k7
    ctl.evict()  # k1
    assert ctl.num_evictions == 3
    assert ctl.bytes_evicted == 80 + 150 + 300


@pytest.mark.parametrize("watermark, ratio", [(0, 0.2), (1.5, 0.2), (0.9, 0), (0.9, 2)])
def test_invalid_config_is_rejected(watermark, ratio):
    with pytest.raises(ValueError):
        EvictionController(L1Cache(100), LRUPolicy(), watermark, ratio)


# =============================================================================
# Wired into KVCache
# =============================================================================
#
# engine.py defaults: 2 layers, block_size 4, 2 heads, head_dim 4, float32.
# With chunk_size 4, one chunk's gathered buffer is 2*2*4*2*4*4 = 512 bytes.

CHUNK = 4
ENTRY = 512


def make_kv(capacity_entries: int, watermark: float, ratio: float):
    pool = make_pool(num_blocks=64)
    l1 = L1Cache(capacity_bytes=capacity_entries * ENTRY)
    policy = LRUPolicy()
    controller = EvictionController(l1, policy, watermark, ratio)
    kv = KVCache(pool, BlockAllocator(64), l1, CHUNK, policy, controller)
    return kv, l1


def run(kv: KVCache, tokens: list[int]):
    result = kv.process(tokens)
    kv.release(result)
    return result


def test_entry_size_assumption():
    # Everything below depends on this number; check it rather than trust it.
    kv, l1 = make_kv(10, 0.9, 0.2)
    run(kv, list(range(CHUNK)))
    assert l1.usage_bytes == ENTRY


def test_l1_stays_under_the_watermark():
    # 5 entries of room, act at 4 (0.8), free 2 (0.4).
    # Predicted: the 4th put evicts chunks 0,1; the 6th put evicts chunks 2,3.
    kv, l1 = make_kv(5, watermark=0.8, ratio=0.4)
    result = run(kv, list(range(6 * CHUNK)))
    keys = [c.key for c in result.chunks]
    assert result.evicted == keys[:4]
    assert l1.usage_ratio < 0.8


def test_a_hit_protects_an_entry():
    # 5 entries of room, act at 4, free 1.
    kv, l1 = make_kv(5, watermark=0.8, ratio=0.2)
    a = run(kv, list(range(1, 2 * CHUNK + 1)))  # a0, a1      -> 2 entries
    b = run(kv, list(range(101, 101 + CHUNK)))  # b0          -> 3 entries
    again = run(kv, list(range(1, 2 * CHUNK + 1)))  # a0, a1 hit: now the most recent
    assert again.hits == 2
    c = run(kv, list(range(201, 201 + CHUNK)))  # c0 -> 4 entries -> evict one

    # Predicted: b0 is the coldest. Without touch-on-hit it would be a0 (FIFO).
    assert c.evicted == [b.chunks[0].key]
    assert all(l1.contains(ch.key) for ch in a.chunks)


def test_without_a_controller_stage3_behaviour_is_unchanged():
    pool = make_pool(num_blocks=64)
    l1 = L1Cache(capacity_bytes=2 * ENTRY)
    kv = KVCache(pool, BlockAllocator(64), l1, CHUNK)
    result = run(kv, list(range(3 * CHUNK)))
    assert result.evicted == []
    assert l1.num_entries == 2  # the third put is refused, nothing evicted
    assert math.isclose(l1.usage_ratio, 1.0)
