"""
HuggingFace PreTrainedTokenizer wrapper for the Regent SentencePiece tokenizer.

Enables:
    tok = AutoTokenizer.from_pretrained("path/to/export", trust_remote_code=True)
    ids = tok("Hello, Regent!", return_tensors="pt")
    txt = tok.decode(output_ids[0])
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


VOCAB_FILES = {"vocab_file": "regent.model"}

SPECIAL_TOKENS: Dict[str, int] = {
    "[PAD]":    0,
    "[BOS]":    1,
    "[EOS]":    2,
    "[GROUND]": 3,
    "[EPG]":    4,
    "[META]":   5,
}


class RegentTokenizer(PreTrainedTokenizer):
    """
    SentencePiece BPE tokenizer for Regent.

    Special tokens:
        [PAD] = 0   [BOS] = 1   [EOS] = 2
        [GROUND] = 3   [EPG] = 4   [META] = 5
    """

    vocab_files_names       = VOCAB_FILES
    model_input_names       = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file:    str,
        bos_token:     str = "[BOS]",
        eos_token:     str = "[EOS]",
        unk_token:     str = "[UNK]",
        pad_token:     str = "[PAD]",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        **kwargs,
    ):
        self.vocab_file    = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(vocab_file)

        self._special = SPECIAL_TOKENS

        super().__init__(
            bos_token      = bos_token,
            eos_token      = eos_token,
            unk_token      = unk_token,
            pad_token      = pad_token,
            add_bos_token  = add_bos_token,
            add_eos_token  = add_eos_token,
            **kwargs,
        )

    # ------------------------------------------------------------------ vocab

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.sp.IdToPiece(i): i for i in range(self.sp.GetPieceSize())}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # ------------------------------------------------------------------ core tokenizer interface

    def _tokenize(self, text: str) -> List[str]:
        return self.sp.EncodeAsPieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._special:
            return self._special[token]
        return self.sp.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        # Check special tokens first
        for tok, idx in self._special.items():
            if idx == index:
                return tok
        return self.sp.IdToPiece(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Filter special tokens before decoding
        filtered = [t for t in tokens if t not in self._special]
        return self.sp.DecodePieces(filtered)

    # ------------------------------------------------------------------ build_inputs_with_special_tokens

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )
        bos = [1] if self.add_bos_token else [0]
        eos = [1] if self.add_eos_token else [0]
        if token_ids_1 is None:
            return bos + ([0] * len(token_ids_0)) + eos
        return bos + ([0] * len(token_ids_0)) + eos + ([0] * len(token_ids_1)) + eos

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos = [0] if self.add_bos_token else []
        eos = [0] if self.add_eos_token else []
        if token_ids_1 is None:
            return bos + ([0] * len(token_ids_0)) + eos
        return bos + ([0] * len(token_ids_0)) + eos + ([0] * len(token_ids_1)) + eos

    # ------------------------------------------------------------------ save / load

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path must be a directory: {save_directory}")
        name   = (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES["vocab_file"]
        out    = os.path.join(save_directory, name)
        import shutil
        if os.path.abspath(self.vocab_file) != os.path.abspath(out):
            shutil.copy2(self.vocab_file, out)
        return (out,)
