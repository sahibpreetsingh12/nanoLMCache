"""
Stage 1 demo: real text -> chunks -> keys.

Models the workload LMCache actually targets: a long shared context (system
prompt + document) followed by a short, varying question. The context is
identical across turns, so its chunk keys are identical too — which is exactly
what makes the KV for it reusable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import encode, decode
from token_db import process_tokens, CHUNK_SIZE

SYSTEM_PROMPT = """You are a careful systems engineer. Answer using only the
document provided. If the document does not contain the answer, say so plainly
rather than guessing. Prefer concrete mechanisms over general description."""

DOCUMENT = """
Paged attention changes how an inference engine stores the key-value cache. A
naive implementation reserves one contiguous buffer per sequence, sized for the
longest output the sequence might produce. That wastes an enormous amount of
memory, because most sequences finish long before they reach the maximum, and
the reserved tail sits unused for the lifetime of the request.

The paged approach borrows an idea from virtual memory. The cache is carved into
fixed-size blocks, each holding the keys and values for a small number of tokens
- sixteen is a common choice. A sequence is then described by a block table: an
ordered list of block identifiers naming which physical blocks hold its tokens.
The blocks themselves need not be adjacent. They are handed out from a free pool
as the sequence grows, and returned when it finishes.

This buys three things. Memory is allocated only as tokens are produced, so
nothing is reserved speculatively. Fragmentation drops to at most one partly
filled block per sequence. And because a block is just an entry in a table, two
sequences that share a prefix can point at the same physical blocks rather than
holding duplicate copies, which is the foundation of prefix caching.

The cost is indirection. Every attention kernel must now consult the block table
to find where a token's keys and values actually live, and the blocks for a
single sequence are scattered across the pool rather than laid out in order.
Reading a contiguous span of tokens means gathering from many places at once.

That scattering is what makes exporting the cache interesting. An external cache
tier wants a flat, contiguous buffer it can hash, store, and later hand back. The
engine holds the same data spread across blocks that were allocated in whatever
order the pool happened to have free. Bridging the two requires walking the block
table, copying each block's slice into the right offset of a flat buffer, and
repeating that for every layer of the model.

The reverse direction is subtler. When a cached entry is restored, the sequence
receiving it has been allocated a completely different set of blocks. The token
content is identical, but the physical addresses are not. The restore path has to
scatter the flat buffer back into whatever blocks the new request was given,
which means the block table is an input to the copy, not a property of the data.

None of this changes the arithmetic of attention. It changes only where the
operands live, and how much bookkeeping stands between a token index and the
bytes that represent it.
"""

QUESTION_A = "\n\nQuestion: Why does the paged approach reduce fragmentation?"
QUESTION_B = "\n\nQuestion: What has to happen when a cached entry is restored?"


def show(label: str, prompt: str) -> list[int]:
    tokens = encode(prompt)
    words = len(prompt.split())
    print(f"\n{label}")
    print(f"  {words} words -> {len(tokens)} tokens  (ratio {len(tokens)/words:.2f})")
    keys = []
    for start, end, key in process_tokens(tokens):
        preview = decode(tokens[start:start + 8]).replace("\n", " ").strip()
        print(f"  [{start:4d}:{end:4d}]  {key:#018x}  {preview[:40]!r}...")
        keys.append(key)
    dropped = len(tokens) - (len(tokens) - len(tokens) % CHUNK_SIZE)
    print(f"  {len(keys)} complete chunks, {dropped} trailing tokens dropped")
    return keys


if __name__ == "__main__":
    context = SYSTEM_PROMPT + DOCUMENT

    a = show("TURN A — context + question A", context + QUESTION_A)
    b = show("TURN B — context + question B", context + QUESTION_B)

    print("\nCHUNK KEY COMPARISON")
    for i in range(max(len(a), len(b))):
        ka = a[i] if i < len(a) else None
        kb = b[i] if i < len(b) else None
        mark = "HIT " if ka == kb else "MISS"
        print(f"  chunk {i}: {mark}  {ka:#018x}  vs  {kb:#018x}")

    shared = sum(1 for x, y in zip(a, b) if x == y)
    print(f"\n  {shared}/{len(a)} chunks reusable across the two turns")
