from typing import Optional

class EncodingPair:
    """Result of encoding a pair of texts (e.g. for cross-encoder models)."""

    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class Tokenizer:
    """Fast, correct tokenizer. Supports BPE, WordPiece, and Unigram."""

    @staticmethod
    def from_json(path: str) -> "Tokenizer":
        """Load a tokenizer from a HuggingFace tokenizer.json file."""
        ...
    @staticmethod
    def from_file(path: str) -> "Tokenizer":
        """Load a tokenizer from a .tkz binary file."""
        ...
    @staticmethod
    def from_pretrained(repo_id: str) -> "Tokenizer":
        """Download and load a tokenizer from the HuggingFace Hub."""
        ...
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text into token IDs."""
        ...
    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        """Encode multiple texts in parallel."""
        ...
    def encode_pair(
        self, text_a: str, text_b: str, add_special_tokens: bool = True
    ) -> EncodingPair:
        """Encode a pair of texts (e.g. for cross-encoder models)."""
        ...
    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode raw bytes into token IDs."""
        ...
    def decode(self, tokens: list[int]) -> Optional[str]:
        """Decode token IDs back to a string. Returns None if not valid UTF-8."""
        ...
    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode token IDs back to raw bytes."""
        ...
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        ...
    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts in parallel."""
        ...
    def save(self, path: str) -> None:
        """Save the tokenizer to a .tkz binary file."""
        ...
    @property
    def vocab_size(self) -> int:
        """The vocabulary size."""
        ...
    def __repr__(self) -> str: ...

class TokieError(Exception):
    """Error raised by tokie operations."""
    ...
