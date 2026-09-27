"""
L1 — the hot cache tier.

If L0 is the desk, L1 is the filing cabinet next to it. Same room, same process,
but nothing in it belongs to a live request: entries here are keyed by *content*
(the chained prefix hash from token_db.py) and survive the request that produced
them. That is the whole point -- L0 blocks go back to the free list when a
request finishes, L1 entries do not.

What it stores is a flat buffer, exactly what gather() produced:

    buffer  [layers, 2, num_tokens, heads, head_dim]

and never a block ID, a block table, or a pool. Keeping engine state out of here
is deliberate: l1.py does not import engine.py, so the day L1 moves into another
process there is nothing engine-shaped to untangle. Keys and buffers cross the
seam; addresses do not.

Underneath, this is a dict. The only thing it adds is a byte counter -- because
a cache without a size is just a memory leak with good manners. Every eviction
policy in Stage 4 is a decision about *which* key to drop; the number that says
"drop something" comes from here.

Note what this class deliberately does NOT do:

  No eviction.   put() refuses an entry that does not fit and says so. Choosing
                 a victim is a policy question, and policy lands in eviction.py
                 (Stage 4). A tier that silently evicted would make the first
                 cache hit experiment unreproducible.

  No hit stats.  Hit and miss are properties of a *lookup*, and lookups belong
                 to the orchestrator. L1 only knows whether it has a key.

  No hashing.    Keys arrive already computed. L1 never sees a token.
"""

import torch


def buffer_bytes(buffer: torch.Tensor) -> int:
    """
    Size of a buffer in bytes: element count times element width.

    Not len(), not numel() -- bytes. Capacity is a memory budget, and float32
    and float16 buffers of the same shape cost different amounts of it.
    """
    return buffer.numel() * buffer.element_size()


class L1Cache:
    """
    Content-addressed store of gathered KV buffers, with a byte budget.

    Insertion order is preserved (plain dicts do that since 3.7), which is the
    hook Stage 4's LRU will reach for. Nothing here depends on it yet.
    """

    def __init__(self, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be positive, got {capacity_bytes}")
        self.capacity_bytes = capacity_bytes
        self._store: dict[int, torch.Tensor] = {}
        self._usage_bytes = 0

    # -- accounting ----------------------------------------------------------

    @property
    def usage_bytes(self) -> int:
        """Bytes currently held. Tracked incrementally, never recomputed."""
        return self._usage_bytes

    @property
    def usage_ratio(self) -> float:
        """usage/capacity, in [0, 1]. Stage 4's watermarks compare against this."""
        return self._usage_bytes / self.capacity_bytes

    @property
    def num_entries(self) -> int:
        return len(self._store)

    def keys(self) -> list[int]:
        """Keys in insertion order -- oldest first."""
        return list(self._store)

    # -- the cache itself ----------------------------------------------------

    def contains(self, key: int) -> bool:
        """Is this chunk cached? The question cache.py asks before prefilling."""
        return key in self._store

    def put(self, key: int, buffer: torch.Tensor) -> bool:
        """
        Store a gathered buffer under its content key.

        Returns True if stored, False if it did not fit. False is a normal
        outcome, not an error: an uncacheable chunk just gets prefilled again
        next time, which is slower but still correct. Raising here would make a
        full cache break requests, which is the opposite of what a cache is for.

        The buffer is cloned. The caller's copy came out of gather() and is
        theirs to reuse or overwrite; if L1 kept a reference, a later write
        through that reference would silently rewrite a cache entry and the
        hit would return KV that no longer matches its key. Copying at the
        boundary is what makes the entry immutable in practice.

        Re-putting an existing key overwrites it, and the accounting subtracts
        the old size before adding the new one. Same key means same tokens, so
        this should be a no-op in content terms -- but getting the arithmetic
        right matters more than the case being rare, since a leak here shows up
        much later as a cache that thinks it is full while holding nothing.
        """
        nbytes = buffer_bytes(buffer)
        if nbytes > self.capacity_bytes:
            # Cannot ever fit, even in an empty cache. Worth separating from
            # "does not fit right now" -- no amount of eviction fixes it.
            return False

        old = self._store.get(key)
        projected = self._usage_bytes + nbytes - (buffer_bytes(old) if old is not None else 0)
        if projected > self.capacity_bytes:
            return False

        self._store[key] = buffer.clone()
        self._usage_bytes = projected
        return True

    def get(self, key: int) -> torch.Tensor | None:
        """
        Fetch a cached buffer, or None if it is not here.

        None rather than KeyError: a miss is the expected half of a lookup.

        The returned tensor is L1's own. scatter() only reads from it, which is
        the only thing cache.py does with it -- callers must not write through
        it. Handing back a clone instead would be safer and would also copy the
        whole buffer on every hit, which is precisely the cost the cache exists
        to avoid.
        """
        return self._store.get(key)

    def delete(self, key: int) -> bool:
        """
        Drop an entry and reclaim its bytes. True if something was removed.

        Stage 4's eviction loop is a sequence of these; Stage 5 will copy the
        buffer down to L2 first and then call this.
        """
        buffer = self._store.pop(key, None)
        if buffer is None:
            return False
        self._usage_bytes -= buffer_bytes(buffer)
        return True

    def clear(self) -> None:
        self._store.clear()
        self._usage_bytes = 0

    def __repr__(self) -> str:
        return (
            f"L1Cache({self.num_entries} entries, "
            f"{self._usage_bytes}/{self.capacity_bytes} bytes, "
            f"{self.usage_ratio:.0%} full)"
        )
