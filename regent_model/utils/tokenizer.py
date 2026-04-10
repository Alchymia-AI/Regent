"""
Tokenizer wrapper for Regent model.

Uses SentencePiece BPE tokenizer. For the prototype, we can use an
existing tokenizer (e.g., from Llama) since tokenizer training requires
the full pre-training corpus.

Special tokens:
    [PAD]    = 0
    [BOS]    = 1
    [EOS]    = 2
    [GROUND] = 3  — inserted by Ver Head when grounding drops below threshold
    [EPG]    = 4  — marks start of EPG prefix region
    [META]   = 5  — marks the ---REGENT_META--- boundary
"""

from pathlib import Path

import sentencepiece as spm


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "[GROUND]": 3,
    "[EPG]": 4,
    "[META]": 5,
}


class RegentTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.special = SPECIAL_TOKENS

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def pad_id(self) -> int:
        return self.special["[PAD]"]

    @property
    def bos_id(self) -> int:
        return self.special["[BOS]"]

    @property
    def eos_id(self) -> int:
        return self.special["[EOS]"]

    @property
    def ground_id(self) -> int:
        return self.special["[GROUND]"]

    @property
    def meta_id(self) -> int:
        return self.special["[META]"]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = self.sp.Encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        # Filter out special tokens before decoding
        filtered = [i for i in ids if i >= len(self.special)]
        return self.sp.Decode(filtered)

    def encode_epg_node(self, key: str, value: str, max_tokens: int = 32) -> list[int]:
        """Encode an EPG node's key+value pair, truncated to max_tokens."""
        text = f"{key}: {value}"
        ids = self.sp.Encode(text)
        return ids[:max_tokens]
