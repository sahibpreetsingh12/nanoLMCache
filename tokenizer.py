"""
Text <-> token IDs.

Thin wrapper over a real HuggingFace tokenizer. Deliberately real rather than a
toy: real tokenization is where word count stops equalling token count, and that
mismatch is what makes chunk boundaries land in unintuitive places.
"""

from transformers import AutoTokenizer

DEFAULT_MODEL = "facebook/opt-125m"

_cache: dict[str, object] = {}


def get_tokenizer(name: str = DEFAULT_MODEL):
    """
    Load a tokenizer. Downloads to ~/.cache/huggingface on first call, then
    reads from disk. Memoised so repeated calls are free.
    """
    if name not in _cache:
        _cache[name] = AutoTokenizer.from_pretrained(name)
    return _cache[name]


def encode(prompt: str, name: str = DEFAULT_MODEL) -> list[int]:
    """Text -> token IDs."""
    return get_tokenizer(name).encode(prompt)


def decode(token_ids: list[int], name: str = DEFAULT_MODEL) -> str:
    """Token IDs -> text. For eyeballing what a chunk actually contains."""
    return get_tokenizer(name).decode(token_ids)
