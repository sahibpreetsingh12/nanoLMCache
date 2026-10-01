# nanoLMCache

A minimal KV cache for LLM inference, built from scratch to understand how
[LMCache](https://github.com/LMCache/LMCache) works.

Not a clone and not a library — a small, complete implementation of the ideas:
prefix-based chunking, paged KV storage, gather/scatter between tiers, and
(soon) eviction. CPU-only, no GPU, no real model weights.

---

## Where I am

**Stage 3 of 5 — working cache.** A prompt goes in, chunks that were seen
before are restored from cache, the rest is prefilled and stored.

- [x] **Stage 1** — prompt → tokens → chunks → chained prefix keys
- [x] **Stage 2** — paged L0 pool, gather/scatter round trip
- [x] **Stage 3** — L1 store + the hit/miss branch
- [ ] **Stage 4** — eviction: watermark, ratio, LRU
- [ ] **Stage 5** — L2 disk tier


**Next:** `eviction.py` — LRU victim selection, and a watermark loop that
decides when to run it.

---

## Try it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers pytest

python nano-lmcache-run.py --chunk 64
```

It asks how many turns, then reads that many prompts. Each turn appends to the
conversation, so later turns share a prefix with earlier ones — which is where
the reuse comes from.

```
turn 2/3 > What does the block table describe?
   [    0:   64]  HIT   key 0xd2ad18ea5e2df168
   102 tokens · 1 hit / 0 miss · 64 reused · 38 trailing uncached

==============================================================
SUMMARY
==============================================================
  chunks looked up     3
    hits               2  (67%)
    misses             1  (33%)

  prefill avoided 128 tokens  (42% of all tokens sent)
  L1  1 entries, 8.0 KB of 50 MB  (0.02% full)
  L0  1024 of 1024 blocks free, pool is 2048 KB
```

Tests: `pytest tests/ -q` — 27 passing, offline, no network.

---

## The pipeline

```
"Explain KV caching"                       prompt
        │  tokenizer.py
        ▼
[1023, 44, 8871, 502, ...]                 token IDs
        │  token_db.py: chunk(256) + chained prefix hash
        ▼
[(0,256,h1), (256,512,h2)]                 (start, end, key)
        │  cache.py: l1.contains(key)?
        ▼
    ┌── HIT ──► l1.get ──► scatter into L0 blocks ──► skip prefill
    │
    └── MISS ─► prefill ─► KV lands in L0, scattered blocks
                        │  gather(block_table)
                        ▼
                 contiguous buffer ──► l1.put(key, buffer)
```

**L0** is the engine's own KV pool — small, fixed, freed when a request ends.
**L1** is the cache's memory — holds KV from finished requests.
**L2** (Stage 5) is disk.

---

## Modules

| File | What it does |
|---|---|
| `tokenizer.py` | prompt ↔ token IDs (real HF tokenizer, `facebook/opt-125m`) |
| `token_db.py` | chunking + chained prefix hashing → cache keys |
| `engine.py` | the L0 paged pool, block allocator, fake prefill |
| `transfer.py` | `gather` (blocks → flat buffer), `scatter` (buffer → blocks) |
| `l1.py` | content-addressed buffer store with a byte budget |
| `cache.py` | the orchestrator — the hit/miss branch |
| `nano-lmcache-run.py` | interactive driver with stats |

---

## Three ideas worth the whole project

**Keys are chained.** A chunk's key is hashed together with the key of
everything before it, so two prompts sharing their first N chunks produce the
same first N keys — and diverge from N+1 onward. That single fact is why cache
reuse is always a *prefix*, never a middle match.

**A cache entry cannot contain an address.** KV lives in scattered blocks whose
IDs are on loan from the allocator. `gather` throws those addresses away and
produces a flat buffer, which is what makes the entry storable, hashable, and
restorable into completely different blocks later.

**Complete chunks only.** A trailing partial chunk is dropped — it still gets
prefilled and still occupies L0, it just never gets a key. With `chunk_size=256`
that can waste up to 255 tokens of recomputation on every request.

---

## Scope

Deliberately out: real model weights, GPU/CUDA, attention, the two-process
split, ZMQ/shared memory, compression, distributed anything.

Built alongside contributing to LMCache and SGLang. The toy exists to make the
real thing legible.
