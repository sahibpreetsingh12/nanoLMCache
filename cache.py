"""
The orchestrator — where a prompt meets the cache.

Everything else in this project does one job and knows nothing about the others.
This file is the one that knows the order they go in:

    tokens -> keys          token_db.process_tokens
    key in L1?              l1.contains
      HIT   fetch, scatter into this request's blocks, skip prefill
      MISS  prefill, gather, store under the key

That branch is the entire point of a KV cache. Everything above it exists to
make the question answerable; everything below it exists to make the answer
cheap to keep.

Two things stay separate on purpose:

  Keys are content, blocks are addresses. A chunk's key is the same every time
  those tokens appear. The blocks holding it are whatever the allocator had
  free this second. The cache is keyed by the first and knows nothing of the
  second -- which is why a hit can be restored into blocks that did not exist
  when it was stored.

  The engine allocates for the whole prompt; the cache only handles complete
  chunks. A trailing partial chunk still gets prefilled and still occupies L0.
  It simply never gets a key, so it is recomputed on every request.
"""

import math
from dataclasses import dataclass, field

import torch

from engine import fake_prefill
from l1 import L1Cache
from token_db import CHUNK_SIZE, process_tokens
from transfer import gather, scatter


@dataclass
class ChunkResult:
    """What happened to one chunk of one request."""

    start: int
    end: int
    key: int
    hit: bool
    blocks: list[int]


@dataclass
class RequestResult:
    """What happened to one request."""

    num_tokens: int
    block_table: list[int]
    """ so we can have a request with necessary blocks like num_tokens, 
    block table but chunks can be empty if the request is not processed yet. """
    chunks: list[ChunkResult] = field(default_factory=list)

    @property
    def hits(self) -> int:
        return sum(1 for c in self.chunks if c.hit)

    @property
    def misses(self) -> int:
        return len(self.chunks) - self.hits

    @property
    def tokens_reused(self) -> int:
        """Tokens whose KV came from the cache instead of being computed."""
        return sum(c.end - c.start for c in self.chunks if c.hit)

    def __repr__(self) -> str:
        return (
            f"RequestResult({self.num_tokens} tokens, "
            f"{self.hits} hit / {self.misses} miss, "
            f"{self.tokens_reused} tokens reused)"
        )


class KVCache:
    """
    Wires token_db, engine, transfer and l1 into one request path.

    Holds no KV of its own -- the pool belongs to the engine and the buffers
    belong to L1. This object only knows the sequence of calls.
    """

    def __init__(
        self,
        pool: torch.Tensor,
        allocator,
        l1: L1Cache,
        chunk_size: int = CHUNK_SIZE,
    ):
        self.pool = pool
        self.allocator = allocator
        self.l1 = l1
        self.chunk_size = chunk_size
        self.block_size = pool.shape[3]

        if chunk_size % self.block_size != 0:
            # A chunk that ended mid-block would share its last block with the
            # next chunk, so restoring one would half-overwrite the other.
            raise ValueError(
                f"chunk_size {chunk_size} must be a multiple of "
                f"block_size {self.block_size}"
            )

    def _blocks_for(self, block_table: list[int], start: int, end: int) -> list[int]:
        """The slice of the block table covering token positions [start, end)."""
        first = start // self.block_size
        last = math.ceil(end / self.block_size)
        return block_table[first:last]

    def process(self, token_ids: list[int]) -> RequestResult:
        """
        Run one request: allocate, resolve every chunk, return what happened.

        Blocks are allocated for *all* tokens, including any trailing remainder
        too short to be a chunk -- the engine needs that KV to generate even
        though the cache will never hold it.
        """
        num_blocks = math.ceil(len(token_ids) / self.block_size)
        block_table = self.allocator.allocate(num_blocks)
        result = RequestResult(num_tokens=len(token_ids), block_table=block_table)

        cached_upto = 0
        for start, end, key in process_tokens(token_ids, self.chunk_size):
            blocks = self._blocks_for(block_table, start, end)

            if self.l1.contains(key):
                # HIT: the KV for these tokens already exists. Write it into
                # this request's blocks and never run the forward pass.
                scatter(self.pool, blocks, self.l1.get(key))
                hit = True
            else:
                # MISS: compute it, then flatten it out of the pool and keep it.
                fake_prefill(self.pool, blocks, token_ids[start:end])
                self.l1.put(key, gather(self.pool, blocks, end - start))
                hit = False

            result.chunks.append(ChunkResult(start, end, key, hit, blocks))
            cached_upto = end

        # The tail: tokens past the last complete chunk. Prefilled like any
        # other, but never stored -- there is no key for a partial chunk.
        if cached_upto < len(token_ids):
            tail_blocks = self._blocks_for(block_table, cached_upto, len(token_ids))
            fake_prefill(self.pool, tail_blocks, token_ids[cached_upto:])

        return result

    def release(self, result: RequestResult) -> None:
        """
        Finish a request: hand its blocks back to the engine.

        The KV in those blocks is now unreachable -- the next allocation will
        overwrite it. Anything worth keeping was copied into L1 during
        process(); this is the moment that makes the cache worth having.
        """
        self.allocator.free(result.block_table)
