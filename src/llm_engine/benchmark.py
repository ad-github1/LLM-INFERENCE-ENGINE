from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

from llm_engine.config import EngineConfig
from llm_engine.engine import InferenceEngine
from llm_engine.metrics import MetricsSnapshot


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    request_count: int = 1000
    seed: int = 7
    min_prompt_tokens: int = 4
    max_prompt_tokens: int = 96
    min_new_tokens: int = 4
    max_new_tokens: int = 24
    max_batch_size: int = 32
    prefill_budget: int = 512
    cache_block_size: int = 16
    cache_num_blocks: int = 4096


DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()


def run_benchmark(config: BenchmarkConfig) -> MetricsSnapshot:
    rng = random.Random(config.seed)
    engine = InferenceEngine(
        EngineConfig(
            max_batch_size=config.max_batch_size,
            max_prefill_tokens_per_step=config.prefill_budget,
            cache_block_size=config.cache_block_size,
            cache_num_blocks=config.cache_num_blocks,
        )
    )

    for index in range(config.request_count):
        prompt_len = rng.randint(config.min_prompt_tokens, config.max_prompt_tokens)
        max_new_tokens = rng.randint(config.min_new_tokens, config.max_new_tokens)
        prompt = [rng.randrange(1, 32_000) for _ in range(prompt_len)]
        engine.submit(prompt, max_new_tokens=max_new_tokens, request_id=f"bench-{index:05d}")

    engine.run_until_complete(max_steps=max(10_000, config.request_count * config.max_new_tokens))
    return engine.metrics_snapshot()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic continuous batching benchmark.")
    default = DEFAULT_BENCHMARK_CONFIG
    parser.add_argument("--requests", type=int, default=default.request_count)
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--min-prompt", type=int, default=default.min_prompt_tokens)
    parser.add_argument("--max-prompt", type=int, default=default.max_prompt_tokens)
    parser.add_argument("--min-new", type=int, default=default.min_new_tokens)
    parser.add_argument("--max-new", type=int, default=default.max_new_tokens)
    parser.add_argument("--max-batch-size", type=int, default=default.max_batch_size)
    parser.add_argument("--prefill-budget", type=int, default=default.prefill_budget)
    parser.add_argument("--block-size", type=int, default=default.cache_block_size)
    parser.add_argument("--cache-blocks", type=int, default=default.cache_num_blocks)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        request_count=args.requests,
        seed=args.seed,
        min_prompt_tokens=args.min_prompt,
        max_prompt_tokens=args.max_prompt,
        min_new_tokens=args.min_new,
        max_new_tokens=args.max_new,
        max_batch_size=args.max_batch_size,
        prefill_budget=args.prefill_budget,
        cache_block_size=args.block_size,
        cache_num_blocks=args.cache_blocks,
    )
    snapshot = run_benchmark(config)
    if args.json:
        print(json.dumps(asdict(snapshot), indent=2, sort_keys=True))
        return

    print(f"requests:        {snapshot.completed_requests}/{snapshot.submitted_requests}")
    print(f"steps:           {snapshot.step_count}")
    print(f"tokens:          {snapshot.total_generated_tokens}")
    print(f"throughput:      {snapshot.request_throughput_rps:,.2f} req/s")
    print(f"token rate:      {snapshot.token_throughput_tps:,.2f} tok/s")
    print(
        "latency ms:      "
        f"p50={snapshot.p50_latency_ms:.3f} "
        f"p95={snapshot.p95_latency_ms:.3f} "
        f"p99={snapshot.p99_latency_ms:.3f}"
    )
    print(
        "queue wait ms:   "
        f"p50={snapshot.p50_queue_wait_ms:.3f} "
        f"p95={snapshot.p95_queue_wait_ms:.3f} "
        f"p99={snapshot.p99_queue_wait_ms:.3f}"
    )
    print(f"avg batch size:  {snapshot.avg_batch_size:.2f}")
    print(f"max batch size:  {snapshot.max_batch_size_seen}")
    print(f"memory util:     peak={snapshot.peak_memory_utilization:.2%}")
    print(f"internal frag:   peak={snapshot.peak_internal_fragmentation_ratio:.2%}")
    print(f"external frag:   peak={snapshot.peak_external_fragmentation_ratio:.2%}")


if __name__ == "__main__":
    main()
