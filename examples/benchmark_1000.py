from llm_engine.benchmark import BenchmarkConfig, run_benchmark


def main() -> None:
    snapshot = run_benchmark(BenchmarkConfig(request_count=1000))
    print(f"completed={snapshot.completed_requests}")
    print(f"p99_latency_ms={snapshot.p99_latency_ms:.3f}")
    print(f"token_throughput_tps={snapshot.token_throughput_tps:.2f}")
    print(f"peak_memory_utilization={snapshot.peak_memory_utilization:.2%}")
    print(f"peak_internal_fragmentation={snapshot.peak_internal_fragmentation_ratio:.2%}")


if __name__ == "__main__":
    main()
