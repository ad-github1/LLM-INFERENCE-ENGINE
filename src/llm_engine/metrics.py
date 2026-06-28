from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter_ns

from llm_engine.kv_cache import CacheStats
from llm_engine.types import Request, RequestStatus


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    duration_ms: float
    batch_size: int
    prefill_tokens: int
    decode_tokens: int
    emitted_tokens: int
    memory_utilization: float
    internal_fragmentation_ratio: float
    external_fragmentation_ratio: float


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    submitted_requests: int
    completed_requests: int
    failed_requests: int
    total_generated_tokens: int
    total_prefill_tokens: int
    total_decode_tokens: int
    step_count: int
    elapsed_ms: float
    request_throughput_rps: float
    token_throughput_tps: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p50_queue_wait_ms: float
    p95_queue_wait_ms: float
    p99_queue_wait_ms: float
    avg_batch_size: float
    max_batch_size_seen: int
    avg_step_duration_ms: float
    p99_step_duration_ms: float
    avg_memory_utilization: float
    peak_memory_utilization: float
    avg_internal_fragmentation_ratio: float
    peak_internal_fragmentation_ratio: float
    avg_external_fragmentation_ratio: float
    peak_external_fragmentation_ratio: float


@dataclass(slots=True)
class MetricsCollector:
    started_at_ns: int = field(default_factory=perf_counter_ns)
    steps: list[StepMetrics] = field(default_factory=list)

    def record_step(
        self,
        *,
        step: int,
        duration_ms: float,
        batch_size: int,
        prefill_tokens: int,
        decode_tokens: int,
        emitted_tokens: int,
        cache_stats: CacheStats,
    ) -> None:
        self.steps.append(
            StepMetrics(
                step=step,
                duration_ms=duration_ms,
                batch_size=batch_size,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                emitted_tokens=emitted_tokens,
                memory_utilization=cache_stats.memory_utilization,
                internal_fragmentation_ratio=cache_stats.internal_fragmentation_ratio,
                external_fragmentation_ratio=cache_stats.external_fragmentation_ratio,
            )
        )

    def snapshot(self, requests: tuple[Request, ...]) -> MetricsSnapshot:
        now_ns = perf_counter_ns()
        elapsed_ms = max((now_ns - self.started_at_ns) / 1_000_000, 0.001)
        completed = [request for request in requests if request.status == RequestStatus.FINISHED]
        failed = [request for request in requests if request.status == RequestStatus.FAILED]

        latency_ms = [
            (request.finished_at_ns - request.submitted_at_ns) / 1_000_000
            for request in completed
            if request.finished_at_ns is not None and request.submitted_at_ns > 0
        ]
        queue_wait_ms = [
            (request.first_token_at_ns - request.submitted_at_ns) / 1_000_000
            for request in completed
            if request.first_token_at_ns is not None and request.submitted_at_ns > 0
        ]

        step_durations = [step.duration_ms for step in self.steps]
        batch_sizes = [step.batch_size for step in self.steps]
        memory = [step.memory_utilization for step in self.steps]
        internal_frag = [step.internal_fragmentation_ratio for step in self.steps]
        external_frag = [step.external_fragmentation_ratio for step in self.steps]
        total_generated_tokens = sum(len(request.generated_tokens) for request in completed)
        elapsed_seconds = elapsed_ms / 1000

        return MetricsSnapshot(
            submitted_requests=len(requests),
            completed_requests=len(completed),
            failed_requests=len(failed),
            total_generated_tokens=total_generated_tokens,
            total_prefill_tokens=sum(step.prefill_tokens for step in self.steps),
            total_decode_tokens=sum(step.decode_tokens for step in self.steps),
            step_count=len(self.steps),
            elapsed_ms=elapsed_ms,
            request_throughput_rps=len(completed) / elapsed_seconds,
            token_throughput_tps=total_generated_tokens / elapsed_seconds,
            p50_latency_ms=percentile(latency_ms, 50),
            p95_latency_ms=percentile(latency_ms, 95),
            p99_latency_ms=percentile(latency_ms, 99),
            p50_queue_wait_ms=percentile(queue_wait_ms, 50),
            p95_queue_wait_ms=percentile(queue_wait_ms, 95),
            p99_queue_wait_ms=percentile(queue_wait_ms, 99),
            avg_batch_size=mean(batch_sizes) if batch_sizes else 0.0,
            max_batch_size_seen=max(batch_sizes, default=0),
            avg_step_duration_ms=mean(step_durations) if step_durations else 0.0,
            p99_step_duration_ms=percentile(step_durations, 99),
            avg_memory_utilization=mean(memory) if memory else 0.0,
            peak_memory_utilization=max(memory, default=0.0),
            avg_internal_fragmentation_ratio=mean(internal_frag) if internal_frag else 0.0,
            peak_internal_fragmentation_ratio=max(internal_frag, default=0.0),
            avg_external_fragmentation_ratio=mean(external_frag) if external_frag else 0.0,
            peak_external_fragmentation_ratio=max(external_frag, default=0.0),
        )


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if percentile_value <= 0:
        return min(values)
    if percentile_value >= 100:
        return max(values)

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (percentile_value / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
