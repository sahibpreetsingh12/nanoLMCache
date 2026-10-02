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

## The three tiers

Everything here is named after *where the KV lives*. These names come from
LMCache and are not universal, so they are worth pinning down before reading any
code.

| | What it is | Lives in | Addressed by | Survives the request? |
|---|---|---|---|---|
| **L0** | the engine's own KV pool, carved into fixed-size blocks | GPU memory (here: one torch tensor) | **block ID** — a position in the pool | **No.** Blocks go back on the free list |
| **L1** | the cache's own memory | CPU RAM (here: a dict) | **content key** — a hash of the tokens | **Yes** |
| **L2** | the cold tier *(Stage 5, not built yet)* | disk | content key | Yes, and across restarts |

The step from L0 to L1 is the one that matters, and it is a change of
**addressing scheme**, not just a copy. L0 answers *"which block?"*. L1 answers
*"which tokens?"*. `gather` and `scatter` are the translation between the two —
which is the second of the three ideas below.

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

python run.py --chunk 64
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

## How to read this repo

GitHub lists files alphabetically, which here is almost exactly backwards:
`cache.py` depends on everything, `tokenizer.py` depends on nothing. Read them
in this order instead.

Every file opens with a long module docstring explaining *why* it is shaped the
way it is. Those docstrings are the real writing in this repo — the code itself
is short.

| # | File | What to look for |
|---|---|---|
| 1 | [`tokenizer.py`](tokenizer.py) | The smallest file. A real HF tokenizer, deliberately not a toy — word count ≠ token count, and that is what makes chunk boundaries land in unintuitive places. |
| 2 | [`token_db.py`](token_db.py) | Chunking, and the **chained** prefix hash. Idea #1. Convince yourself why chunk 3's key depends on chunks 1 and 2. |
| 3 | [`engine.py`](engine.py) | **L0.** The 6-axis pool shape, and why a *block* is the unit of copying. A block ID is a position, not a slab: block 7 exists in every layer and for both K and V. |
| 4 | [`transfer.py`](transfer.py) | `gather` / `scatter`. The best docstring here — four reasons a cache entry cannot hold an address. Idea #2. |
| 5 | [`l1.py`](l1.py) | **L1.** A dict with a byte budget. Note everything it deliberately does *not* do: no eviction, no hit stats, no hashing. |
| 6 | [`cache.py`](cache.py) | The orchestrator — the only file that knows what order the others go in, and the only place the HIT/MISS branch appears. **Read it last.** |
| 7 | [`run.py`](run.py) | Drive it yourself: multi-turn, with hit/miss stats. |
| 8 | [`tests/`](tests/) | What each piece has to guarantee. `test_transfer.py` is worth reading for *why* its fixture exists at all. |

Also here: [`demos/`](demos/) — the animation above, plus a chunking
walkthrough. [`extras/`](extras/) — early design notes, kept for the record and
not part of the project.

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


