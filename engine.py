"""
L0 — the engine's own KV cache.

One fixed-size tensor, carved into blocks, handed out by an allocator. This is
the memory the engine attends over; nothing here survives a request finishing.

The pool is allocated once and never resized. A real engine claims whatever GPU
memory is left after the model weights and keeps it for the process lifetime, so
it never has to allocate mid-request and never OOMs halfway through generation.

Shape:

    [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
         |       |       |           |            |          |
      which    K or V  which     which token    which    the numbers
      layer            block      in block      head

The order matters. The rightmost dimensions vary fastest, so pool[layer, kv, b]
is one contiguous slab -- which is what makes a block the unit of copying.

Note a block ID names a *position*, not a single slab: block 7 exists in every
layer and for both K and V. With 2 layers that is 4 slabs per block.
"""

import random

import torch

# Toy dimensions: small enough to print the whole pool and check by hand.
# Real models are ~12 layers, 12 heads, head_dim 64, block_size 16.
NUM_LAYERS = 2
NUM_BLOCKS = 16
BLOCK_SIZE = 4
NUM_KV_HEADS = 2
HEAD_DIM = 4


def make_pool(
    num_layers: int = NUM_LAYERS,
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Allocate the KV pool. Called once; the result is L0."""
    return torch.zeros(
        num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype
    )


class OutOfBlocks(Exception):
    """Raised when a request asks for more blocks than are free."""


class BlockAllocator:
    """
    Tracks which blocks are in use and hands out block IDs.

    The list it returns *is* the block table: the ordered mapping from position
    in the sequence to physical block. Everything downstream reads it.

    IDs are handed out in shuffled order on purpose. If this returned
    [0, 1, 2, 3], code that assumed blocks were contiguous would pass its tests
    by accident and fail against a real engine.
    """

    def __init__(self, num_blocks: int = NUM_BLOCKS, seed: int = 0):
        # Shuffled once, then popped from the front. Seeded so runs repeat.
        self._free: list[int] = list(range(num_blocks))
        random.Random(seed).shuffle(self._free)
        self._used: set[int] = set()
        self.num_blocks = num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        """
        Take n blocks out of the free list and return them as a block table.

        Raises rather than returning a short list: a real engine would queue or
        preempt the request, but silently handing back fewer blocks than asked
        for would let the caller index past the end of its own table.
        """
        if n > len(self._free):
            raise OutOfBlocks(f"requested {n} blocks, only {len(self._free)} free")
        block_ids = [self._free.pop(0) for _ in range(n)]
        self._used.update(block_ids)
        return block_ids

    def free(self, block_ids: list[int]) -> None:
        """
        Return blocks to the free list.

        Validated because the failure mode is silent: freeing a block twice
        would put it in the list twice, and two live requests would then be
        handed the same block and overwrite each other's KV.

        Contents are deliberately NOT zeroed. The next writer overwrites them,
        which is what real engines do -- and it means forgetting to write before
        reading gives you someone else's stale KV rather than an obvious crash.
        """
        for block_id in block_ids:
            if block_id not in self._used:
                raise ValueError(f"block {block_id} is not allocated")
            self._used.remove(block_id)
            self._free.append(block_id)
