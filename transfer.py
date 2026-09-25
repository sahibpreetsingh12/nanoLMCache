"""
Moving KV between L0 and a flat buffer.

L0 stores KV paged: scattered blocks, addressed through a block table.
A cache tier wants it flat: one contiguous buffer, addressed by content.
These two functions are the bridge, and they are inverses of each other.

    gather   L0 -> buffer    on a MISS, after prefill, to store the result
    scatter  buffer -> L0    on a HIT, to restore it into a new request

The buffer drops the block and slot axes in favour of a single token axis:

    pool    [layers, 2, num_blocks, block_size, heads, head_dim]
    buffer  [layers, 2,        num_tokens,      heads, head_dim]

Q - Why a buffer can be written to L1 and a pool region cannot:
  Not contiguous.  A request's tokens sit in blocks [10, 14, 5, 1], interleaved
                   with other requests' blocks. There is no single slice of the
                   pool that is "this request's KV" -- reading it means visiting
                   four places. A cache entry has to be one object.

  Not stable.      Block IDs are on loan. When the request finishes they go back
                   to the free list and the next request overwrites them. Storing
                   a reference to block 10 stores a location whose contents are
                   about to change.

  Not portable.    A block ID is an index into one engine's pool. It means
                   nothing to a cache daemon in another process, and nothing at
                   all once written to disk.

  Not addressable. The cache is keyed by content -- the chunk hash from
                   token_db.py. Hashing requires the bytes in a defined order,
                   which the scattered layout does not give you.

Gather fixes all four at once by throwing the addresses away. What comes out is
contiguous, self-describing (axis 2 is sequence position), detached from engine
state, and therefore hashable, storable, sendable, and restorable somewhere else.
"""

import torch


def gather(pool: torch.Tensor, block_table: list[int], num_tokens: int) -> torch.Tensor:
    """
    Read a request's KV out of the pool into one flat buffer, in sequence order.

    Walks the same arithmetic fake_prefill used to write it:
    token i lives in block_table[i // block_size] at slot i % block_size.
    """
    block_size = pool.shape[3]
    num_layers, _, _, _, num_kv_heads, head_dim = pool.shape

    buffer = torch.zeros(
        num_layers, 2, num_tokens, num_kv_heads, head_dim, dtype=pool.dtype
    )
    for i in range(num_tokens):
        buffer[:, :, i] = pool[:, :, block_table[i // block_size], i % block_size]
    return buffer


def scatter(
    pool: torch.Tensor, block_table: list[int], buffer: torch.Tensor
) -> None:
    """
    Write a flat buffer back into the pool, at this request's blocks.

    The inverse of gather, and the reason the block table is an *input* to the
    copy rather than a property of the data. The buffer holds the same tokens
    that were gathered earlier, but the request receiving them has been handed
    a completely different set of blocks -- the originals were freed and reused
    long ago. Same content, new addresses.

    Writes in place and returns nothing: the pool already exists, this fills
    part of it. Capacity is not re-derived from the caller; the buffer knows
    its own length.
    """
    block_size = pool.shape[3]
    num_tokens = buffer.shape[2]

    capacity = len(block_table) * block_size
    if num_tokens > capacity:
        raise ValueError(
            f"{num_tokens} tokens need more than {len(block_table)} blocks "
            f"(capacity {capacity})"
        )

    for i in range(num_tokens):
        pool[:, :, block_table[i // block_size], i % block_size] = buffer[:, :, i]