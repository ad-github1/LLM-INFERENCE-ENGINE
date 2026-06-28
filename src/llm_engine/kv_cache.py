from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil


class KVCacheError(RuntimeError):
    """Raised when KV-cache capacity or ownership rules are violated."""


@dataclass(slots=True)
class CacheBlock:
    block_id: int
    ref_count: int = 0

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


@dataclass(slots=True)
class RequestCacheState:
    request_id: str
    block_ids: list[int]
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class CacheStats:
    block_size: int
    total_blocks: int
    used_blocks: int
    free_blocks: int
    live_requests: int
    allocated_tokens: int
    capacity_tokens: int
    allocated_token_slots: int
    internal_fragmentation_tokens: int
    internal_fragmentation_ratio: float
    memory_utilization: float
    free_block_runs: int
    largest_free_run: int
    external_fragmentation_ratio: float


class PagedKVCacheManager:
    """Fixed-size block allocator for per-request KV cache.

    The manager stores logical request page tables. It does not hold tensor
    memory; a real backend would use these block IDs to index physical KV pages.
    """

    def __init__(self, *, block_size: int, num_blocks: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.block_size = block_size
        self.num_blocks = num_blocks
        self._blocks = [CacheBlock(block_id=i) for i in range(num_blocks)]
        self._free_block_ids: deque[int] = deque(range(num_blocks))
        self._requests: dict[str, RequestCacheState] = {}

    @property
    def free_blocks(self) -> int:
        return len(self._free_block_ids)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    def has_request(self, request_id: str) -> bool:
        return request_id in self._requests

    def block_table(self, request_id: str) -> tuple[int, ...]:
        state = self._require_request(request_id)
        return tuple(state.block_ids)

    def token_count(self, request_id: str) -> int:
        return self._require_request(request_id).token_count

    def can_allocate_tokens(self, request_id: str, additional_tokens: int) -> bool:
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        state = self._requests.get(request_id)
        current_tokens = 0 if state is None else state.token_count
        current_blocks = 0 if state is None else len(state.block_ids)
        needed_blocks = self._blocks_needed(current_tokens + additional_tokens) - current_blocks
        return needed_blocks <= self.free_blocks

    def create_request(self, request_id: str) -> None:
        if request_id in self._requests:
            raise KVCacheError(f"request {request_id!r} already has a KV cache")
        self._requests[request_id] = RequestCacheState(request_id=request_id, block_ids=[])

    def append_tokens(self, request_id: str, token_count: int) -> tuple[int, ...]:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        if token_count == 0:
            return ()

        state = self._requests.get(request_id)
        current_tokens = 0 if state is None else state.token_count
        current_blocks = 0 if state is None else len(state.block_ids)
        needed_total_blocks = self._blocks_needed(current_tokens + token_count)
        additional_blocks = needed_total_blocks - current_blocks
        if additional_blocks > self.free_blocks:
            raise KVCacheError(
                f"KV cache OOM for request {request_id!r}: need {additional_blocks} blocks, "
                f"only {self.free_blocks} free"
            )

        if state is None:
            self.create_request(request_id)
            state = self._requests[request_id]

        allocated: list[int] = []
        for _ in range(additional_blocks):
            block_id = self._free_block_ids.popleft()
            self._blocks[block_id].ref_count = 1
            state.block_ids.append(block_id)
            allocated.append(block_id)

        state.token_count += token_count
        return tuple(allocated)

    def free_request(self, request_id: str) -> tuple[int, ...]:
        state = self._requests.pop(request_id, None)
        if state is None:
            return ()

        freed: list[int] = []
        for block_id in state.block_ids:
            block = self._blocks[block_id]
            if block.ref_count <= 0:
                raise KVCacheError(f"block {block_id} has invalid ref_count {block.ref_count}")
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_block_ids.append(block_id)
                freed.append(block_id)
        return tuple(freed)

    def stats(self) -> CacheStats:
        allocated_tokens = sum(state.token_count for state in self._requests.values())
        allocated_token_slots = self.used_blocks * self.block_size
        internal_fragmentation_tokens = max(0, allocated_token_slots - allocated_tokens)
        internal_fragmentation_ratio = (
            internal_fragmentation_tokens / allocated_token_slots if allocated_token_slots else 0.0
        )
        capacity_tokens = self.block_size * self.num_blocks
        memory_utilization = allocated_tokens / capacity_tokens if capacity_tokens else 0.0
        free_block_runs, largest_free_run = self._free_run_stats()
        external_fragmentation_ratio = (
            1.0 - (largest_free_run / self.free_blocks) if self.free_blocks else 0.0
        )
        return CacheStats(
            block_size=self.block_size,
            total_blocks=self.num_blocks,
            used_blocks=self.used_blocks,
            free_blocks=self.free_blocks,
            live_requests=len(self._requests),
            allocated_tokens=allocated_tokens,
            capacity_tokens=capacity_tokens,
            allocated_token_slots=allocated_token_slots,
            internal_fragmentation_tokens=internal_fragmentation_tokens,
            internal_fragmentation_ratio=internal_fragmentation_ratio,
            memory_utilization=memory_utilization,
            free_block_runs=free_block_runs,
            largest_free_run=largest_free_run,
            external_fragmentation_ratio=external_fragmentation_ratio,
        )

    def _blocks_needed(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return ceil(token_count / self.block_size)

    def _free_run_stats(self) -> tuple[int, int]:
        if not self._free_block_ids:
            return 0, 0

        free_ids = sorted(self._free_block_ids)
        run_count = 1
        current_run = 1
        largest_run = 1
        previous = free_ids[0]

        for block_id in free_ids[1:]:
            if block_id == previous + 1:
                current_run += 1
            else:
                run_count += 1
                largest_run = max(largest_run, current_run)
                current_run = 1
            previous = block_id

        largest_run = max(largest_run, current_run)
        return run_count, largest_run

    def _require_request(self, request_id: str) -> RequestCacheState:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KVCacheError(f"request {request_id!r} does not have a KV cache") from exc
