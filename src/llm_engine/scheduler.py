from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from llm_engine.config import EngineConfig
from llm_engine.kv_cache import PagedKVCacheManager
from llm_engine.types import DecodeWork, PrefillWork, Request, RequestStatus


@dataclass(frozen=True, slots=True)
class Schedule:
    prefill: tuple[PrefillWork, ...]
    decode: tuple[DecodeWork, ...]


class _CacheProjection:
    """Tracks per-step KV reservations before the engine mutates the cache."""

    def __init__(self, cache: PagedKVCacheManager) -> None:
        self.cache = cache
        self.free_blocks = cache.free_blocks
        self.tokens: dict[str, int] = {}
        self.blocks: dict[str, int] = {}

    def can_reserve(self, request_id: str, token_count: int) -> bool:
        return self._additional_blocks(request_id, token_count) <= self.free_blocks

    def reserve(self, request_id: str, token_count: int) -> bool:
        additional_blocks = self._additional_blocks(request_id, token_count)
        if additional_blocks > self.free_blocks:
            return False

        current_tokens = self._tokens_for(request_id)
        current_blocks = self._blocks_for(request_id)
        self.free_blocks -= additional_blocks
        self.tokens[request_id] = current_tokens + token_count
        self.blocks[request_id] = current_blocks + additional_blocks
        return True

    def _additional_blocks(self, request_id: str, token_count: int) -> int:
        current_tokens = self._tokens_for(request_id)
        current_blocks = self._blocks_for(request_id)
        needed_blocks = self._blocks_needed(current_tokens + token_count)
        return needed_blocks - current_blocks

    def _tokens_for(self, request_id: str) -> int:
        if request_id in self.tokens:
            return self.tokens[request_id]
        if not self.cache.has_request(request_id):
            return 0
        return self.cache.token_count(request_id)

    def _blocks_for(self, request_id: str) -> int:
        if request_id in self.blocks:
            return self.blocks[request_id]
        if not self.cache.has_request(request_id):
            return 0
        return len(self.cache.block_table(request_id))

    def _blocks_needed(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return ceil(token_count / self.cache.block_size)


class ContinuousBatchScheduler:
    """Builds each serving step from live requests and resource budgets."""

    def __init__(self, config: EngineConfig, cache: PagedKVCacheManager) -> None:
        self.config = config
        self.cache = cache

    def schedule(self, requests: dict[str, Request]) -> Schedule:
        projection = _CacheProjection(self.cache)
        decode = self._schedule_decode(requests, projection)
        prefill = self._schedule_prefill(
            requests,
            reserved_decode_slots=len(decode),
            projection=projection,
        )
        return Schedule(prefill=prefill, decode=decode)

    def _schedule_decode(
        self,
        requests: dict[str, Request],
        projection: _CacheProjection,
    ) -> tuple[DecodeWork, ...]:
        decode_limit = self.config.max_decode_tokens_per_step or self.config.max_batch_size
        decode_limit = min(decode_limit, self.config.max_batch_size)

        work: list[DecodeWork] = []
        for request in requests.values():
            if len(work) >= decode_limit:
                break
            if request.status != RequestStatus.DECODE:
                continue
            if not projection.reserve(request.request_id, 1):
                continue
            work.append(
                DecodeWork(
                    request_id=request.request_id,
                    token_id=request.last_generated_token,
                    position=len(request.prompt_tokens) + len(request.generated_tokens),
                )
            )
        return tuple(work)

    def _schedule_prefill(
        self,
        requests: dict[str, Request],
        *,
        reserved_decode_slots: int,
        projection: _CacheProjection,
    ) -> tuple[PrefillWork, ...]:
        available_batch_slots = max(0, self.config.max_batch_size - reserved_decode_slots)
        if available_batch_slots == 0:
            return ()

        token_budget = self.config.max_prefill_tokens_per_step
        work: list[PrefillWork] = []

        for request in requests.values():
            if len(work) >= available_batch_slots or token_budget <= 0:
                break
            if request.status not in {RequestStatus.WAITING, RequestStatus.PREFILL}:
                continue
            remaining = request.remaining_prompt_tokens()
            if remaining <= 0:
                continue

            chunk_len = min(remaining, token_budget)
            while chunk_len > 0 and not projection.can_reserve(request.request_id, chunk_len):
                chunk_len -= 1
            if chunk_len == 0:
                continue
            projection.reserve(request.request_id, chunk_len)

            token_start = request.prompt_cursor
            tokens = tuple(request.prompt_tokens[token_start : token_start + chunk_len])
            work.append(
                PrefillWork(
                    request_id=request.request_id,
                    token_start=token_start,
                    tokens=tokens,
                    is_final_chunk=chunk_len == remaining,
                )
            )
            token_budget -= chunk_len

        return tuple(work)
