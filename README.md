# LLM Inference Engine -- Continuous Batching & KV-Cache Manager

A compact LLM serving prototype focused on production inference mechanics:

- **Continuous batching**: new requests can enter the running batch while older requests are decoding.
- **Paged KV-cache management**: sequence cache is allocated in fixed-size blocks, grown token by token, and freed when requests finish.
- **Serving telemetry**: benchmark p50/p95/p99 latency, queue wait, throughput, batch size, memory utilization, and fragmentation.
- **Backend boundary**: default dependency-free fake backend plus optional C++/ONNX Runtime/gRPC integration scaffolding.

The runnable Python path intentionally uses a deterministic fake backend. That keeps the scheduler, cache manager, and benchmark easy to test without downloading model weights. The repository also includes a proto contract and C++ gRPC server skeleton for replacing the fake backend with an ONNX Runtime transformer backend.

## What Is Included

- Request lifecycle: waiting, prefill, decode, finished, failed.
- Chunked prefill under a per-step token budget.
- Decode scheduling with a maximum concurrent sequence count.
- Paged KV-cache allocator with per-request block tables.
- Cache capacity checks before admission and decode growth.
- Internal and external cache fragmentation tracking.
- Metrics snapshot for request latency, queue wait, token throughput, request throughput, batch size, and memory utilization.
- 1,000+ synthetic request benchmark for variable prompt/output lengths.
- Deterministic token generation backend for demos and tests.
- Optional `GrpcModelBackend` client and C++ gRPC backend scaffold.
- Unit tests for cache allocation, out-of-memory behavior, batching, and cleanup.

## Quick Start

```bash
cd outputs/llm-inference-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m llm_engine
python -m llm_engine.benchmark --requests 1000
python -m unittest discover -s tests
```

You can also run the CLI through the installed script:

```bash
llm-engine-demo --cache-blocks 12 --block-size 4 --max-batch-size 3
llm-engine-benchmark --requests 1000 --json
```

## Architecture

```text
Client requests
      |
      v
InferenceEngine
      |
      +--> ContinuousBatchScheduler
      |       - chooses decode requests
      |       - chunks/adopts waiting prefills
      |       - enforces batch and token budgets
      |
      +--> PagedKVCacheManager
      |       - allocates fixed-size blocks
      |       - tracks per-request page tables
      |       - reports utilization and fragmentation
      |       - frees blocks on completion
      |
      +--> MetricsCollector
      |       - p50/p95/p99 latency
      |       - queue wait and throughput
      |       - batch size and cache telemetry
      |
      +--> ModelBackend
              - FakeModelBackend by default
              - optional GrpcModelBackend client
              - optional C++ ONNX Runtime gRPC server scaffold
```

## Example

```python
from llm_engine import EngineConfig, InferenceEngine

engine = InferenceEngine(
    EngineConfig(
        max_batch_size=4,
        max_prefill_tokens_per_step=8,
        cache_block_size=4,
        cache_num_blocks=16,
    )
)

engine.submit([10, 11, 12], max_new_tokens=5)
engine.submit([4, 5], max_new_tokens=4)

while engine.has_pending_work:
    result = engine.step()
    print(result)

print(engine.completed_requests)
print(engine.metrics_snapshot())
```

To run tests without installing the package first:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Synthetic Benchmark

```bash
PYTHONPATH=src python3 -m llm_engine.benchmark --requests 1000
```

The benchmark submits variable-length synthetic requests, runs the continuous batching loop to completion, and reports:

- p50/p95/p99 end-to-end latency.
- p50/p95/p99 queue wait.
- request throughput and token throughput.
- average and maximum scheduled batch size.
- peak memory utilization.
- peak internal and external fragmentation.

## Optional C++ ONNX Runtime gRPC Backend

The serving loop is isolated from token generation behind `ModelBackend`. A real model adapter would:

- Convert scheduled work into tensor batches.
- Bind each request to its KV block table.
- Run prefill chunks and decode tokens on the accelerator.
- Return sampled next tokens to the engine.

Relevant files:

- `proto/inference.proto`: gRPC service contract.
- `src/llm_engine/grpc_backend.py`: optional Python gRPC client backend.
- `cpp_backend/onnx_grpc_server.cc`: C++ gRPC service skeleton.
- `cpp_backend/CMakeLists.txt`: native build shape for gRPC/protobuf and ONNX Runtime.

The native backend is not required for tests. It becomes active after installing/generating the gRPC bindings and linking an ONNX Runtime build.

The KV manager already exposes logical block tables, so this is intentionally close to how paged attention systems wire sequence IDs to physical cache pages.
