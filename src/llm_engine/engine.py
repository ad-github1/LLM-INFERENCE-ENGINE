from __future__ import annotations

from collections import OrderedDict
from itertools import count
from time import perf_counter_ns
from typing import Iterable

from llm_engine.backend import FakeModelBackend, ModelBackend, ModelOutput
from llm_engine.config import EngineConfig
from llm_engine.kv_cache import KVCacheError, PagedKVCacheManager
from llm_engine.metrics import MetricsCollector, MetricsSnapshot
from llm_engine.scheduler import ContinuousBatchScheduler
from llm_engine.types import (
    DecodeWork,
    PrefillWork,
    Request,
    RequestStatus,
    StepResult,
    TokenEvent,
)


class InferenceEngine:
    """Single-process LLM serving loop with continuous batching."""

    def __init__(self, config: EngineConfig | None = None, backend: ModelBackend | None = None) -> None:
        self.config = config or EngineConfig()
        self.backend = backend or FakeModelBackend(eos_token_id=self.config.eos_token_id)
        self.cache = PagedKVCacheManager(
            block_size=self.config.cache_block_size,
            num_blocks=self.config.cache_num_blocks,
        )
        self.scheduler = ContinuousBatchScheduler(self.config, self.cache)
        self.metrics = MetricsCollector()
        self._requests: OrderedDict[str, Request] = OrderedDict()
        self._completed: OrderedDict[str, Request] = OrderedDict()
        self._id_counter = count(1)
        self._step = 0

    @property
    def has_pending_work(self) -> bool:
        return any(not request.is_terminal for request in self._requests.values())

    @property
    def live_requests(self) -> tuple[Request, ...]:
        return tuple(request for request in self._requests.values() if not request.is_terminal)

    @property
    def completed_requests(self) -> tuple[Request, ...]:
        return tuple(self._completed.values())

    def submit(
        self,
        prompt_tokens: Iterable[int],
        *,
        max_new_tokens: int,
        request_id: str | None = None,
        eos_token_id: int | None = None,
    ) -> str:
        prompt = list(prompt_tokens)
        if not prompt:
            raise ValueError("prompt_tokens must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        request_id = request_id or f"req-{next(self._id_counter)}"
        if request_id in self._requests:
            raise ValueError(f"request_id {request_id!r} already exists")

        request = Request(
            request_id=request_id,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.config.eos_token_id if eos_token_id is None else eos_token_id,
            submitted_step=self._step,
            submitted_at_ns=perf_counter_ns(),
        )
        self._requests[request_id] = request
        return request_id

    def step(self) -> StepResult:
        step_start_ns = perf_counter_ns()
        self._step += 1
        schedule = self.scheduler.schedule(self._requests)
        token_events: list[TokenEvent] = []
        finished_ids: list[str] = []

        self._apply_prefill(schedule.prefill, event_time_ns=perf_counter_ns())
        prefill_outputs = self.backend.prefill(schedule.prefill)
        token_events.extend(
            self._apply_model_outputs(
                prefill_outputs,
                source="prefill",
                event_time_ns=perf_counter_ns(),
            )
        )

        self._apply_decode_cache_growth(schedule.decode, event_time_ns=perf_counter_ns())
        decode_outputs = self.backend.decode(schedule.decode)
        token_events.extend(
            self._apply_model_outputs(
                decode_outputs,
                source="decode",
                event_time_ns=perf_counter_ns(),
            )
        )

        for event in token_events:
            if event.finished:
                finished_ids.append(event.request_id)

        self._retire_finished()
        stats = self.cache.stats()
        step_duration_ms = (perf_counter_ns() - step_start_ns) / 1_000_000
        prefill_tokens = sum(len(item.tokens) for item in schedule.prefill)
        decode_tokens = len(schedule.decode)
        batch_size = len(schedule.prefill) + len(schedule.decode)
        self.metrics.record_step(
            step=self._step,
            duration_ms=step_duration_ms,
            batch_size=batch_size,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            emitted_tokens=len(token_events),
            cache_stats=stats,
        )
        return StepResult(
            step=self._step,
            prefill=schedule.prefill,
            decode=schedule.decode,
            token_events=tuple(token_events),
            finished_request_ids=tuple(finished_ids),
            cache_used_blocks=stats.used_blocks,
            cache_free_blocks=stats.free_blocks,
            cache_memory_utilization=stats.memory_utilization,
            cache_internal_fragmentation_ratio=stats.internal_fragmentation_ratio,
            cache_external_fragmentation_ratio=stats.external_fragmentation_ratio,
            step_duration_ms=step_duration_ms,
        )

    def run_until_complete(self, *, max_steps: int = 10_000) -> tuple[StepResult, ...]:
        results: list[StepResult] = []
        while self.has_pending_work:
            if len(results) >= max_steps:
                raise TimeoutError(f"engine still has pending work after {max_steps} steps")
            results.append(self.step())
        return tuple(results)

    def request(self, request_id: str) -> Request:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request_id {request_id!r}") from exc

    def metrics_snapshot(self) -> MetricsSnapshot:
        return self.metrics.snapshot(tuple(self._requests.values()))

    def _apply_prefill(self, work: tuple[PrefillWork, ...], *, event_time_ns: int) -> None:
        for item in work:
            request = self._requests[item.request_id]
            request.status = RequestStatus.PREFILL
            try:
                self.cache.append_tokens(item.request_id, len(item.tokens))
            except KVCacheError as exc:
                request.status = RequestStatus.FAILED
                request.error = str(exc)
                request.finished_step = self._step
                request.finished_at_ns = event_time_ns
                continue
            request.prompt_cursor += len(item.tokens)

    def _apply_decode_cache_growth(self, work: tuple[DecodeWork, ...], *, event_time_ns: int) -> None:
        for item in work:
            request = self._requests[item.request_id]
            try:
                self.cache.append_tokens(item.request_id, 1)
            except KVCacheError as exc:
                request.status = RequestStatus.FAILED
                request.error = str(exc)
                request.finished_step = self._step
                request.finished_at_ns = event_time_ns

    def _apply_model_outputs(
        self,
        outputs: tuple[ModelOutput, ...],
        *,
        source: str,
        event_time_ns: int,
    ) -> tuple[TokenEvent, ...]:
        events: list[TokenEvent] = []
        for output in outputs:
            request = self._requests[output.request_id]
            if request.is_terminal:
                continue

            request.generated_tokens.append(output.next_token)
            if request.first_token_step is None:
                request.first_token_step = self._step
                request.first_token_at_ns = event_time_ns

            if request.should_stop():
                request.status = RequestStatus.FINISHED
                request.finished_step = self._step
                request.finished_at_ns = event_time_ns
                finished = True
            else:
                request.status = RequestStatus.DECODE
                finished = False

            events.append(
                TokenEvent(
                    request_id=output.request_id,
                    token_id=output.next_token,
                    step=self._step,
                    source=source,
                    finished=finished,
                )
            )
        return tuple(events)

    def _retire_finished(self) -> None:
        for request_id, request in list(self._requests.items()):
            if not request.is_terminal:
                continue
            if request_id in self._completed:
                continue
            self.cache.free_request(request_id)
            self._completed[request_id] = request
