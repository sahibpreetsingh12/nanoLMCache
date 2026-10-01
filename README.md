# nanoLMCache

A minimal KV cache for LLM inference, built from scratch to understand how
[LMCache](https://github.com/LMCache/LMCache) works.

Not a clone and not a library — a small, complete implementation of the ideas:
prefix-based chunking, paged KV storage, gather/scatter between tiers, and
(soon) eviction. CPU-only, no GPU, no real model weights.

---

## Watch it

<!--
  Replace the line below with a GitHub attachment URL, on its own line:

    1. Open a new issue on this repo (you will NOT submit it)
    2. Drag demos/out/kv_cache_dark.mp4 into the comment box and wait for upload
    3. Copy the https://github.com/user-attachments/assets/... URL it inserts
    4. Paste it below as a BARE line — no markdown link, no image syntax
    5. Close the issue tab without submitting

  A bare user-attachments URL renders as an inline player. A ![](...) or a
  YouTube link will not — GitHub README markdown cannot embed an iframe.
  Attachments are size-capped (~10MB for video), so render at -qm rather than
  -qh if the file is too large.
-->

_(video goes here — see the comment in this file's source)_


---

## Three Key ideas 
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

## Try Yourself

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

Tests: `pytest tests/ -q` — 29 passing, offline, no network.

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
| `demos/` | the animation, and the chunking walkthrough |

---

## Where I am

**Stage 3 of 5 — working cache.** A prompt goes in, chunks that were seen
before are restored from cache, the rest is prefilled and stored.

- [x] **Stage 1** — prompt → tokens → chunks → chained prefix keys
- [x] **Stage 2** — paged L0 pool, gather/scatter round trip
- [x] **Stage 3** — L1 store + the hit/miss branch
- [ ] **Stage 4** — eviction: watermark, ratio, LRU
- [ ] **Stage 5** — L2 disk tier

**Next:** Working on `tests/test_transfer.py` (2 of 4 tests written), then
`eviction.py` — LRU victim selection, and a watermark loop that decides when to
run it.

Until Stage 4 lands there is no notion of hot or cold here: a full `L1Cache.put`
refuses the new entry rather than evicting an old one, so the cache never has to
ask which entry is coldest. LRU is what creates that answer; the L2 tier is what
makes "cold" mean *demoted* rather than *deleted*.

---


