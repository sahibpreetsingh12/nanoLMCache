"""
Tests for gather/scatter — the L0 <-> flat-buffer round trip.

The thing under test is pure addressing arithmetic. gather and scatter compute
no values of their own; all they do is decide, for each token position i, which
block and slot it belongs to:

    pool    [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
                 |       |       |           |            |          |
              which    K or V  which     which token    which    the numbers
              layer            block      in block      head

So a test here is not asking "is the arithmetic fast" or "is the KV correct" --
it is asking "did every number land in the position it was supposed to land in".
That question is only answerable if every number is *distinguishable* from every
other one, which is what fill_distinct below exists to arrange.
"""

import math

import pytest
import torch

from engine import BlockAllocator, make_pool
from transfer import gather, scatter


# The offset stride. 1000 keeps the two halves of each number readable as decimal
# digits, which matters because these numbers get read by a human when a test
# fails. Guarded in _slab rather than assumed.
STRIDE = 1000


def _slab(pool: torch.Tensor, i: int) -> torch.Tensor:
    """
    The numbers token i is given -- a two-part address, (i+1)*1000 + 0..31.

    Values only. This function knows nothing about blocks, slots, or block
    tables, which is the whole reason it can be shared between fill_distinct
    (which writes these numbers into the pool) and expected_buffer (which says
    where gather must put them). Sharing the *values* is fine: they are fixture
    definition, not the thing under test. Sharing the *addressing* would not be
    -- then a wrong `//` or `%` would appear identically on both sides of the
    assert and cancel out, which is precisely the trap a round-trip test falls
    into.
    """
    num_layers, two, _, _, num_kv_heads, head_dim = pool.shape
    slab_shape = (num_layers, two, num_kv_heads, head_dim)
    slab_numel = num_layers * two * num_kv_heads * head_dim

    # If a slab ever held 1000+ numbers the two halves of the address would
    # overlap and two different positions could collide.
    assert slab_numel < STRIDE, (
        f"slab holds {slab_numel} numbers, too many for a stride of {STRIDE}"
    )
    return (i + 1) * STRIDE + torch.arange(slab_numel, dtype=pool.dtype).reshape(
        slab_shape
    )


def expected_buffer(pool: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    The buffer gather MUST return, built without calling gather or scatter.

    This is the independent expectation, and it is the only reason the test can
    catch a bug in the addressing arithmetic. It is built from one fact about
    gather's contract -- axis 2 of a buffer is sequence position:

        pool    [layers, 2, num_blocks, block_size, heads, head_dim]   6 axes
        buffer  [layers, 2,        num_tokens,      heads, head_dim]   5 axes

    gather collapses the block and slot axes into one token axis. So the buffer
    is just the slabs in sequence order, stacked at dim=2 -- no block table
    involved, because a buffer has no addresses in it at all.
    """
    return torch.stack([_slab(pool, i) for i in range(num_tokens)], dim=2)


def fill_distinct(pool: torch.Tensor, block_table: list[int], num_tokens: int) -> None:
    """
    Fill a request's slots so that every single number in the pool is unique.

    Why not just use engine.fake_prefill? Because fake_prefill writes one scalar
    into the whole of token i's slab:

        pool[:, :, block_id, slot, :, :] = tok

    The two `:` on each side cover layer, K/V, head and head_dim, so a scalar on
    the right gets broadcast into all of them. Token 7's KV becomes the number 7
    repeated 32 times. That is deliberate -- it makes the pool readable by eye --
    but it makes this file's tests unable to fail. If every number in a slab is
    identical, then a gather that swapped K with V, or transposed the layers, or
    shuffled the heads, returns a byte-identical buffer and the assert passes.
    A test whose fixture is symmetric under the bug it is hunting cannot see it.

    The numbers themselves come from _slab, where the scheme is explained. All
    this function adds is the addressing: which block, which slot.

    Note the signature takes num_tokens, not token_ids. fake_prefill needs the
    ids because it writes them; this does not care what the tokens *are*, only
    where each position lands. The arithmetic is the subject.

    Writes in place, like fake_prefill and scatter.
    """
    block_size = pool.shape[3]

    capacity = len(block_table) * block_size
    if num_tokens > capacity:
        raise ValueError(
            f"{num_tokens} tokens need more than {len(block_table)} blocks "
            f"(capacity {capacity})"
        )

    for i in range(num_tokens):
        block_id = block_table[i // block_size]
        slot = i % block_size
        pool[:, :, block_id, slot, :, :] = _slab(pool, i)


# =============================================================================
# gather
# =============================================================================


def test_gather_matches_expected():
    """
    gather lays a request's KV out in sequence order, whatever blocks it is in.

    The one test that can catch a wrong `//` or `%`, because the right-hand side
    of the assert is built from gather's contract rather than from gather. The
    `zip(token_ids, block_table)` bug -- one token per block instead of
    block_size tokens per block -- fails here on the first token of block 1.
    """
    pool = make_pool()
    block_size = pool.shape[3]

    # A token count that fills its blocks exactly; the partial-last-block case
    # is a separate concern and gets its own test.
    num_tokens = 3 * block_size
    block_table = BlockAllocator().allocate(math.ceil(num_tokens / block_size))

    # Guard the premise. The allocator shuffles on purpose, and if it ever
    # handed back consecutive IDs this test would still pass while proving
    # nothing about scattered blocks.
    assert block_table != sorted(block_table), (
        f"block table {block_table} is not scattered; test proves nothing"
    )

    fill_distinct(pool, block_table, num_tokens)
    got = gather(pool, block_table, num_tokens)

    num_layers, two, _, _, num_kv_heads, head_dim = pool.shape
    assert got.shape == (num_layers, two, num_tokens, num_kv_heads, head_dim)
    assert torch.equal(got, expected_buffer(pool, num_tokens))


def test_gather_stops_at_num_tokens():
    """
    gather reads exactly num_tokens, not everything the blocks could hold.

    A request's last block is almost always partly empty -- 14 tokens reserve
    4 blocks of 4 and leave 2 slots over -- and this is the whole reason
    num_tokens is a parameter instead of len(block_table) * block_size. The
    block table says how much space was *reserved*; only the caller knows how
    much of it holds this request's KV.

    Reading the spare slots would not crash. engine.BlockAllocator.free()
    deliberately does not zero a block's contents, so in a running system those
    two slots hold whatever the previous tenant left behind. A gather that read
    them would hand L1 a buffer with two tokens of someone else's KV glued to
    the end, stored under a key that claims to describe 14 tokens. The cache
    would then serve that as a hit. Nothing would raise; the output would just
    be quietly wrong.

    Here the pool starts zeroed and nothing has been freed into it, so those
    slots are 0.0 -- which the last assert turns into a tripwire.
    """
    pool = make_pool()
    block_size = pool.shape[3]

    num_blocks = 4
    num_tokens = num_blocks * block_size - 2  # 14 tokens in 16 slots
    block_table = BlockAllocator().allocate(num_blocks)

    fill_distinct(pool, block_table, num_tokens)

    # Premise: the last two slots of the final block really were left untouched.
    tail = pool[:, :, block_table[-1], num_tokens % block_size :]
    assert torch.equal(tail, torch.zeros_like(tail)), (
        "fill_distinct wrote past num_tokens; this test is testing the wrong thing"
    )

    got = gather(pool, block_table, num_tokens)

    assert got.shape[2] == num_tokens
    assert torch.equal(got, expected_buffer(pool, num_tokens))

    # _slab starts every token at STRIDE, so no number it writes is ever 0.
    # A zero in the buffer can therefore only have come from a slot nobody
    # wrote -- which is exactly the symptom of a gather that ignored
    # num_tokens and read the blocks to capacity.
    assert (got != 0).all(), "buffer contains an unwritten slot"
