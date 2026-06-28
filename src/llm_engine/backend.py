from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_engine.types import DecodeWork, PrefillWork


@dataclass(frozen=True, slots=True)
class ModelOutput:
    request_id: str
    next_token: int


class ModelBackend(ABC):
    """Backend boundary used by the scheduler/engine.

    A real implementation would map work items to tensor batches and use each
    request's KV block table to read/write cache pages.
    """

    @abstractmethod
    def prefill(self, work: tuple[PrefillWork, ...]) -> tuple[ModelOutput, ...]:
        """Run final prefill chunks and return first sampled decode tokens."""

    @abstractmethod
    def decode(self, work: tuple[DecodeWork, ...]) -> tuple[ModelOutput, ...]:
        """Run one-token decode for each active request."""


class FakeModelBackend(ModelBackend):
    """Deterministic token generator for tests and local demos."""

    def __init__(self, *, vocab_size: int = 32000, eos_token_id: int | None = None) -> None:
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def prefill(self, work: tuple[PrefillWork, ...]) -> tuple[ModelOutput, ...]:
        outputs: list[ModelOutput] = []
        for item in work:
            if not item.is_final_chunk:
                continue
            seed = item.tokens[-1] if item.tokens else 0
            outputs.append(
                ModelOutput(
                    request_id=item.request_id,
                    next_token=self._next(seed=seed, position=item.token_start + len(item.tokens)),
                )
            )
        return tuple(outputs)

    def decode(self, work: tuple[DecodeWork, ...]) -> tuple[ModelOutput, ...]:
        return tuple(
            ModelOutput(
                request_id=item.request_id,
                next_token=self._next(seed=item.token_id, position=item.position),
            )
            for item in work
        )

    def _next(self, *, seed: int, position: int) -> int:
        token = (seed * 1103515245 + position * 12345 + 17) % self.vocab_size
        if self.eos_token_id is not None and position > 0 and position % 17 == 0:
            return self.eos_token_id
        return token
