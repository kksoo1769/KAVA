"""한국어와 영어에 특화된 EXAONE-4.0 tokenizer."""

from __future__ import annotations

from transformers import AutoTokenizer

TOKENIZER_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"

def load_tokenizer(name: str = TOKENIZER_NAME):
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer
