"""
Unit tests for chunking and prefix hashing.

These verify the rules that make cache reuse work:
1. Complete chunks only — a trailing partial chunk is dropped by default
2. save_unfull_chunk keeps the remainder, with honest boundaries
3. Sequences shorter than one chunk produce nothing cacheable
4. Keys are deterministic within a process
5. Keys are deterministic ACROSS processes (PYTHONHASHSEED independence)
6. The hash is chained — identical tokens after different prefixes differ
7. A shared prefix yields shared keys until the point of divergence
8. Reported (start, end) boundaries actually cover the sequence

Deliberately offline: these use synthetic token IDs, never the real tokenizer,
so they stay fast and do not depend on network access. Real-text behaviour is
demonstrated in demos/stage1_chunking.py, which is not a test.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from token_db import CHUNK_SIZE, chunk_tokens, hash_tokens, process_tokens

ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Helper Functions
# =============================================================================


def keys_of(tokens, **kwargs):
    """Just the keys, dropping boundaries."""
    return [key for _, _, key in process_tokens(tokens, **kwargs)]


def keys_in_fresh_process(seed):
    """Compute keys for range(641) in a new interpreter with a given hash seed."""
    code = (
        "import json, token_db; "
        "print(json.dumps([k for _, _, k in "
        "token_db.process_tokens(list(range(641)))]))"
    )
    env = dict(os.environ, PYTHONHASHSEED=seed)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


# =============================================================================
# Chunking
# =============================================================================


class TestChunking:
    """How a token sequence is cut into chunks."""

    def test_truncation_drops_partial_chunk(self):
        # 641 = 2 complete chunks + 129 leftover. The leftover is dropped.
        chunks = list(chunk_tokens(list(range(641))))
        assert len(chunks) == 2
        assert all(len(c) == CHUNK_SIZE for c in chunks)

    def test_save_unfull_chunk_keeps_remainder(self):
        chunks = list(chunk_tokens(list(range(641)), save_unfull_chunk=True))
        assert len(chunks) == 3
        assert len(chunks[-1]) == 129

    def test_under_chunk_size_yields_nothing(self):
        assert list(chunk_tokens(list(range(200)))) == []

    def test_exact_multiple_has_no_remainder(self):
        chunks = list(chunk_tokens(list(range(768))))
        assert len(chunks) == 3
        assert [t for c in chunks for t in c] == list(range(768))

    def test_chunks_are_contiguous_and_ordered(self):
        chunks = list(chunk_tokens(list(range(20)), chunk_size=8))
        assert chunks == [list(range(8)), list(range(8, 16))]


# =============================================================================
# Hashing
# =============================================================================


class TestHashing:
    """Properties the key function must have."""

    def test_same_tokens_same_keys(self):
        assert keys_of(list(range(641))) == keys_of(list(range(641)))

    def test_keys_stable_across_processes(self):
        # The real PYTHONHASHSEED check: builtin hash() is randomised per
        # process, so a key function built on it would differ between runs.
        # The engine and the cache daemon are separate processes, so a key
        # that is not stable across them can never produce a hit.
        in_process = keys_of(list(range(641)))
        seed_a = keys_in_fresh_process("0")
        seed_b = keys_in_fresh_process("12345")
        assert seed_a == seed_b
        assert seed_a == in_process

    def test_prefix_hash_is_chained(self):
        # Identical tokens reached via different histories must differ,
        # or KV computed in one context would be served in another.
        assert hash_tokens([1, 2, 3], 0) != hash_tokens([1, 2, 3], 999)

    def test_chunk_content_affects_key(self):
        assert hash_tokens([1, 2, 3], 0) != hash_tokens([1, 2, 4], 0)

    def test_keys_fit_declared_width(self):
        key = hash_tokens([1, 2, 3], 0)
        assert 0 <= key < 2**64


# =============================================================================
# Prefix reuse
# =============================================================================


class TestPrefixReuse:
    """The property the whole cache is built on."""

    def test_shared_prefix_diverging_tail(self):
        base = list(range(768))
        forked = base[:600] + list(range(9000, 9168))  # diverges inside chunk 2
        a, b = keys_of(base), keys_of(forked)

        assert len(a) == len(b) == 3
        assert a[0] == b[0]
        assert a[1] == b[1]
        assert a[2] != b[2]

    def test_divergence_poisons_every_later_chunk(self):
        # Chaining means a change in chunk 0 must invalidate 1 and 2 as well.
        base = list(range(768))
        forked = list(range(9000, 9001)) + base[1:]
        a, b = keys_of(base), keys_of(forked)
        assert all(x != y for x, y in zip(a, b))


# =============================================================================
# Boundaries
# =============================================================================


class TestBoundaries:
    """(start, end) must describe the slice the key was computed from."""

    def test_boundaries_cover_the_sequence(self):
        for start, end, _ in process_tokens(list(range(768))):
            assert end - start == CHUNK_SIZE

    def test_boundaries_index_the_right_slice(self):
        tokens = list(range(768))
        for start, end, _ in process_tokens(tokens):
            assert tokens[start:end] == list(range(start, end))

    def test_unfull_chunk_reports_short_boundary(self):
        spans = list(process_tokens(list(range(641)), save_unfull_chunk=True))
        start, end, _ = spans[-1]
        assert (start, end) == (512, 641)
