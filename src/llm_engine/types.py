from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(slots=True)
class Request:
    request_id: str
    prompt_tokens: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None
    status: RequestStatus = RequestStatus.WAITING
    prompt_cursor: int = 0
    generated_tokens: list[int] = field(default_factory=list)
    submitted_step: int = 0
    first_token_step: int | None = None
    finished_step: int | None = None
    submitted_at_ns: int = 0
    first_token_at_ns: int | None = None
    finished_at_ns: int | None = None
    error: str | None = None

    @property
    def prompt_done(self) -> bool:
        return self.prompt_cursor >= len(self.prompt_tokens)

    @property
    def is_terminal(self) -> bool:
        return self.status in {RequestStatus.FINISHED, RequestStatus.FAILED}

    @property
    def needs_decode(self) -> bool:
        return self.status == RequestStatus.DECODE and not self.is_terminal

    @property
    def last_generated_token(self) -> int:
        if not self.generated_tokens:
            raise ValueError("request has not generated a token yet")
        return self.generated_tokens[-1]

    def remaining_prompt_tokens(self) -> int:
        return max(0, len(self.prompt_tokens) - self.prompt_cursor)

    def should_stop(self) -> bool:
        if len(self.generated_tokens) >= self.max_new_tokens:
            return True
        return self.eos_token_id is not None and self.generated_tokens[-1:] == [self.eos_token_id]


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    token_id: int
    step: int
    source: str
    finished: bool = False


@dataclass(frozen=True, slots=True)
class PrefillWork:
    request_id: str
    token_start: int
    tokens: tuple[int, ...]
    is_final_chunk: bool


@dataclass(frozen=True, slots=True)
class DecodeWork:
    request_id: str
    token_id: int
    position: int


@dataclass(frozen=True, slots=True)
class StepResult:
    step: int
    prefill: tuple[PrefillWork, ...] = ()
    decode: tuple[DecodeWork, ...] = ()
    token_events: tuple[TokenEvent, ...] = ()
    finished_request_ids: tuple[str, ...] = ()
    cache_used_blocks: int = 0
    cache_free_blocks: int = 0
    cache_memory_utilization: float = 0.0
    cache_internal_fragmentation_ratio: float = 0.0
    cache_external_fragmentation_ratio: float = 0.0
    step_duration_ms: float = 0.0

    @property
    def emitted_tokens(self) -> int:
        return len(self.token_events)
