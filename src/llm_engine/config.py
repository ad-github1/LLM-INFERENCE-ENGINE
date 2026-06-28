from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Serving limits for the continuous batching engine."""

    max_batch_size: int = 8
    max_prefill_tokens_per_step: int = 32
    max_decode_tokens_per_step: int | None = None
    cache_block_size: int = 16
    cache_num_blocks: int = 256
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_prefill_tokens_per_step <= 0:
            raise ValueError("max_prefill_tokens_per_step must be positive")
        if self.max_decode_tokens_per_step is not None and self.max_decode_tokens_per_step <= 0:
            raise ValueError("max_decode_tokens_per_step must be positive when set")
        if self.cache_block_size <= 0:
            raise ValueError("cache_block_size must be positive")
        if self.cache_num_blocks <= 0:
            raise ValueError("cache_num_blocks must be positive")
