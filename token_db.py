"""
Chunking + prefix hashing.

Turns a flat list of token IDs into cache keys. Two rules decide everything:

  1. Tokens are cut into fixed-size chunks (256 by default).
  2. Each chunk's key is hashed together with the key of everything before it.

Rule 2 is why cache reuse is prefix-based: two prompts that share their first
N chunks produce the same first N keys, and diverge from chunk N+1 onward.

Mirrors LMCache's ChunkedTokenDatabase (lmcache/v1/token_database.py:298-450).
"""

import hashlib
import struct
from typing import Iterator

CHUNK_SIZE = 256

# Fixed starting point for the hash chain. Any constant works — it just has to
# be the same on every run, so the first chunk always chains from the same place.
INIT_HASH = 0

# Keys are truncated to 64 bits so they stay readable in logs. Collisions are
# astronomically unlikely at this scale; LMCache makes the same tradeoff.
HASH_BITS = 64
HASH_BYTES = HASH_BITS // 8


def _get_init_hash() -> int:
    return INIT_HASH


def hash_tokens(token_chunk: list[int], prefix_hash: int) -> int:
    """
    Combine one chunk with the hash of everything before it.

    Both inputs matter. Hashing only the chunk would make each key independent,
    and prefix reuse would stop working.

    Deliberately NOT Python's builtin hash(): PYTHONHASHSEED randomises it per
    process, so keys would differ between runs and nothing would ever hit.
    """
    h = hashlib.sha256()
    h.update(prefix_hash.to_bytes(HASH_BYTES, "big"))
    # hashlib needs bytes; tokens are ints. 4-byte signed ints, little-endian.
    h.update(struct.pack(f"<{len(token_chunk)}i", *token_chunk))
    return int.from_bytes(h.digest()[:HASH_BYTES], "big")


def chunk_tokens(
    tokens: list[int],
    chunk_size: int = CHUNK_SIZE,
    save_unfull_chunk: bool = False,
) -> Iterator[list[int]]:
    """
    Cut tokens into chunks of chunk_size.

    When save_unfull_chunk is False, a trailing partial chunk is dropped.
    This is why 641 tokens produce 2 chunks (512 tokens) and not 3 — the last
    129 tokens are never cached.

    Dropping it is the safe default: a partial chunk hashed as if it were
    complete would collide with the same token range in a longer prompt.
    """
    end = len(tokens) if save_unfull_chunk else len(tokens) - len(tokens) % chunk_size
    for i in range(0, end, chunk_size):
        yield tokens[i : i + chunk_size]


def prefix_hash(token_chunks: Iterator[list[int]]) -> Iterator[int]:
    """Chain a hash across chunks, yielding one key per chunk."""
    h = _get_init_hash()
    for token_chunk in token_chunks:
        h = hash_tokens(token_chunk, h)
        yield h


def process_tokens(
    tokens: list[int],
    chunk_size: int = CHUNK_SIZE,
    save_unfull_chunk: bool = False,
) -> Iterator[tuple[int, int, int]]:
    """
    The public entry point: token IDs in, (start, end, key) out.

    start/end are indices into `tokens`, so the caller knows which slice of the
    sequence each key covers — needed later to gather the matching KV.
    """
    chunks = list(chunk_tokens(tokens, chunk_size, save_unfull_chunk))
    for chunk_id, key in enumerate(prefix_hash(iter(chunks))):
        start = chunk_id * chunk_size
        end = start + len(chunks[chunk_id])
        yield start, end, key


if __name__ == "__main__":
    # Smoke tests with synthetic tokens — no tokenizer, no network.

    def show(label, tokens, **kw):
        print(label)
        keys = []
        for start, end, key in process_tokens(tokens, **kw):
            print(f"  [{start:4d}:{end:4d}]  {key:#018x}")
            keys.append(key)
        if not keys:
            print("  (no complete chunks)")
        return keys

    # 1. Truncation: 641 tokens -> 2 chunks, trailing 129 dropped.
    a = show("1. 641 tokens, save_unfull_chunk=False", list(range(641)))

    # 2. Determinism: same tokens, same keys. Re-run the process to prove
    #    these survive PYTHONHASHSEED randomisation.
    b = show("\n2. same 641 tokens again", list(range(641)))
    print(f"   identical: {a == b}")

    # 3. Shared prefix, diverging tail — divergence placed inside a COMPLETE
    #    chunk (token 600 lives in chunk 2, which spans 512:768).
    base = list(range(768))
    forked = base[:600] + list(range(9000, 9168))
    c = show("\n3a. base, 768 tokens", base)
    d = show("3b. same first 600 tokens, different after", forked)
    print(f"   chunk 0 match: {c[0] == d[0]}")
    print(f"   chunk 1 match: {c[1] == d[1]}")
    print(f"   chunk 2 match: {c[2] == d[2]}   <- divergence lands here")

    # 4. Under one chunk: nothing cacheable.
    show("\n4. 200 tokens (under chunk_size)", list(range(200)))
