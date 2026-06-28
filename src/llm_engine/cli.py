from __future__ import annotations

import argparse

from llm_engine import EngineConfig, InferenceEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the continuous batching LLM engine demo.")
    parser.add_argument("--max-batch-size", type=int, default=3)
    parser.add_argument("--prefill-budget", type=int, default=6)
    parser.add_argument("--cache-blocks", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = InferenceEngine(
        EngineConfig(
            max_batch_size=args.max_batch_size,
            max_prefill_tokens_per_step=args.prefill_budget,
            cache_num_blocks=args.cache_blocks,
            cache_block_size=args.block_size,
        )
    )

    engine.submit([10, 11, 12, 13, 14, 15], max_new_tokens=4, request_id="alpha")
    engine.submit([21, 22, 23], max_new_tokens=5, request_id="bravo")
    engine.submit([31, 32, 33, 34], max_new_tokens=3, request_id="charlie")
    engine.submit([41, 42], max_new_tokens=4, request_id="delta")

    print("step | prefill              | decode          | emitted           | cache")
    print("-----+----------------------+-----------------+-------------------+-------------")
    while engine.has_pending_work:
        result = engine.step()
        prefill = ", ".join(f"{w.request_id}:{len(w.tokens)}" for w in result.prefill) or "-"
        decode = ", ".join(w.request_id for w in result.decode) or "-"
        emitted = ", ".join(f"{e.request_id}->{e.token_id}" for e in result.token_events) or "-"
        cache = f"{result.cache_used_blocks} used/{result.cache_free_blocks} free"
        print(f"{result.step:>4} | {prefill:<20} | {decode:<15} | {emitted:<17} | {cache}")

    print()
    for request in engine.completed_requests:
        print(f"{request.request_id}: {request.generated_tokens}")

    metrics = engine.metrics_snapshot()
    print()
    print(
        "metrics: "
        f"p99_latency_ms={metrics.p99_latency_ms:.3f}, "
        f"token_tps={metrics.token_throughput_tps:.2f}, "
        f"peak_memory={metrics.peak_memory_utilization:.2%}, "
        f"peak_internal_frag={metrics.peak_internal_fragmentation_ratio:.2%}"
    )


if __name__ == "__main__":
    main()
