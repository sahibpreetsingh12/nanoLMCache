"""
Interactive driver.

Ask for N turns, read N prompts, run each through the cache, print what
happened. Prompts accumulate the way a chat does -- each turn resends the whole
conversation so far -- which is what produces a shared prefix worth caching.
Pass --no-accumulate to treat every prompt as independent instead.

Usage:
    python run.py                 chunk_size 256, block_size 16
    python run.py --chunk 64      smaller chunks, so short prompts still cache
    python run.py --no-accumulate each prompt stands alone
    python run.py --chunk 8 --block 4 --l1-kb 5   tiny L1, so eviction happens
"""

import argparse
import math

from cache import KVCache
from controller import EvictionController
from engine import BlockAllocator, make_pool
from eviction import LRUPolicy
from l1 import L1Cache
from tokenizer import encode

NUM_BLOCKS = 1024
L1_CAPACITY_BYTES = 50 * 1024 * 1024  # 50 MB, binary to match the KB/MB display


def parse_args():
    p = argparse.ArgumentParser(description="Run prompts through nano-lmcache.")
    p.add_argument("--chunk", type=int, default=256, help="chunk_size (default 256)")
    p.add_argument("--block", type=int, default=16, help="block_size (default 16)")
    p.add_argument("--l1-kb", type=float, default=L1_CAPACITY_BYTES / 1024,
                   help="L1 capacity in KB (default 50 MB worth)")
    p.add_argument("--watermark", type=float, default=0.9,
                   help="evict when L1 usage reaches this fraction (default 0.9)")
    p.add_argument("--ratio", type=float, default=0.2,
                   help="each eviction frees this fraction of capacity (default 0.2)")
    p.add_argument(
        "--no-accumulate",
        action="store_true",
        help="treat each prompt as independent instead of appending to the chat",
    )
    return p.parse_args()


def fmt_bytes(n: float) -> str:
    """Bytes as B / KB / MB, so a 5 KB and a 50 MB L1 both read naturally."""
    for unit in ("B", "KB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} MB"


def ask_int(prompt: str) -> int:
    while True:
        try:
            n = int(input(prompt).strip())
            if n > 0:
                return n
            print("  needs to be a positive number")
        except ValueError:
            print("  needs to be a number")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\naborted")


def main():
    args = parse_args()

    pool = make_pool(num_blocks=NUM_BLOCKS, block_size=args.block)
    allocator = BlockAllocator(num_blocks=NUM_BLOCKS)
    l1 = L1Cache(capacity_bytes=int(args.l1_kb * 1024))
    policy = LRUPolicy()
    controller = EvictionController(l1, policy, args.watermark, args.ratio)
    kv = KVCache(pool, allocator, l1, chunk_size=args.chunk,
                 policy=policy, controller=controller)

    pool_bytes = pool.numel() * pool.element_size()
    print(f"\nnano-lmcache")
    print(f"  chunk_size {args.chunk}, block_size {args.block} "
          f"-> {args.chunk // args.block} blocks per chunk")
    print(f"  L0 pool    {NUM_BLOCKS} blocks, {fmt_bytes(pool_bytes)}, "
          f"{NUM_BLOCKS * args.block} token slots")
    print(f"  L1 budget  {fmt_bytes(l1.capacity_bytes)}, evict at "
          f"{args.watermark:.0%}, free {args.ratio:.0%} each time")
    print(f"  mode       {'independent prompts' if args.no_accumulate else 'accumulating chat'}\n")

    n = ask_int("How many turns? ")

    conversation = ""
    turns = []

    for i in range(1, n + 1):
        try:
            text = input(f"\nturn {i}/{n} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\naborted")
            break
        if not text:
            print("  empty, skipping")
            continue

        conversation = text if args.no_accumulate else (conversation + " " + text).strip()
        tokens = encode(conversation)

        needed = math.ceil(len(tokens) / args.block)
        if needed > allocator.num_free:
            print(f"  out of blocks: need {needed}, {allocator.num_free} free")
            break

        result = kv.process(tokens)
        turns.append(result)

        for c in result.chunks:
            print(f"   [{c.start:5d}:{c.end:5d}]  {'HIT ' if c.hit else 'MISS'}  "
                  f"key {c.key:#018x}")
        tail = len(tokens) - (result.chunks[-1].end if result.chunks else 0)
        if not result.chunks:
            print(f"   nothing cacheable: {len(tokens)} tokens < chunk_size {args.chunk}")
        print(f"   {len(tokens)} tokens · {result.hits} hit / {result.misses} miss "
              f"· {result.tokens_reused} reused · {tail} trailing uncached")
        for key in result.evicted:
            print(f"   evicted               key {key:#018x}")
        if result.evicted:
            print(f"   L1 now {kv.l1.usage_ratio:.0%} full after evicting "
                  f"{len(result.evicted)} entries")

        kv.release(result)

    report(turns, kv, allocator, pool_bytes, args)


def report(turns, kv, allocator, pool_bytes, args):
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)

    if not turns:
        print("  no turns run")
        return

    chunks = sum(len(t.chunks) for t in turns)
    hits = sum(t.hits for t in turns)
    misses = sum(t.misses for t in turns)
    tokens = sum(t.num_tokens for t in turns)
    reused = sum(t.tokens_reused for t in turns)
    uncached = tokens - sum(
        t.chunks[-1].end if t.chunks else 0 for t in turns
    )

    print(f"\n  turns           {len(turns)}")
    print(f"  tokens sent     {tokens}")
    print(f"  chunks looked up{chunks:>6}")
    print(f"    hits          {hits:>6}" + (f"  ({hits/chunks:.0%})" if chunks else ""))
    print(f"    misses        {misses:>6}" + (f"  ({misses/chunks:.0%})" if chunks else ""))

    print(f"\n  prefill avoided {reused} tokens" + (f"  ({reused/tokens:.0%} of all tokens sent)" if tokens else ""))
    print(f"  never cacheable {uncached} tokens  (trailing partial chunks)")

    print(f"\n  L1  {kv.l1.num_entries} entries, "
          f"{fmt_bytes(kv.l1.usage_bytes)} of "
          f"{fmt_bytes(kv.l1.capacity_bytes)}  "
          f"({kv.l1.usage_ratio:.2%} full)")

    print(f"      evicted {kv.controller.num_evictions} entries, "
          f"{fmt_bytes(kv.controller.bytes_evicted)} in total")

    used_blocks = NUM_BLOCKS - allocator.num_free
    print(f"  L0  {allocator.num_free} of {NUM_BLOCKS} blocks free "
          f"({used_blocks} still held), pool is {fmt_bytes(pool_bytes)}")

    if chunks and hits == 0:
        print("\n  no hits: turns shared no complete chunk. Try a longer prompt,")
        print(f"  or --chunk {max(8, args.chunk // 8)} so shorter text still chunks.")


if __name__ == "__main__":
    main()
