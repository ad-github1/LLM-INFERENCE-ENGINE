import unittest

from llm_engine import EngineConfig, InferenceEngine, RequestStatus


class InferenceEngineTests(unittest.TestCase):
    def test_continuous_batching_prefills_new_requests_while_decoding_old_ones(self) -> None:
        engine = InferenceEngine(
            EngineConfig(
                max_batch_size=2,
                max_prefill_tokens_per_step=4,
                cache_block_size=4,
                cache_num_blocks=8,
            )
        )
        engine.submit([1, 2], max_new_tokens=3, request_id="a")

        first = engine.step()
        self.assertEqual([event.request_id for event in first.token_events], ["a"])
        self.assertEqual(engine.request("a").status, RequestStatus.DECODE)

        engine.submit([5, 6, 7], max_new_tokens=2, request_id="b")
        second = engine.step()

        self.assertEqual([work.request_id for work in second.decode], ["a"])
        self.assertEqual([work.request_id for work in second.prefill], ["b"])
        self.assertEqual({event.request_id for event in second.token_events}, {"a", "b"})

    def test_run_until_complete_frees_cache(self) -> None:
        engine = InferenceEngine(
            EngineConfig(
                max_batch_size=4,
                max_prefill_tokens_per_step=8,
                cache_block_size=2,
                cache_num_blocks=16,
            )
        )
        engine.submit([1, 2, 3], max_new_tokens=4, request_id="a")
        engine.submit([4, 5], max_new_tokens=3, request_id="b")

        results = engine.run_until_complete()

        self.assertGreater(len(results), 0)
        self.assertFalse(engine.has_pending_work)
        self.assertEqual(len(engine.completed_requests), 2)
        self.assertEqual(engine.cache.stats().used_blocks, 0)

    def test_cache_pressure_delays_prefill_until_blocks_are_freed(self) -> None:
        engine = InferenceEngine(
            EngineConfig(
                max_batch_size=2,
                max_prefill_tokens_per_step=8,
                cache_block_size=2,
                cache_num_blocks=3,
            )
        )
        engine.submit([1, 2, 3, 4, 5], max_new_tokens=1, request_id="large")
        engine.submit([9, 10], max_new_tokens=1, request_id="small")

        first = engine.step()
        self.assertEqual([work.request_id for work in first.prefill], ["large"])
        self.assertEqual(engine.request("small").status, RequestStatus.WAITING)

        engine.run_until_complete()
        completed_ids = [request.request_id for request in engine.completed_requests]
        self.assertEqual(completed_ids, ["large", "small"])
        self.assertEqual(engine.cache.stats().used_blocks, 0)

    def test_decode_schedule_reserves_cache_for_the_whole_batch(self) -> None:
        engine = InferenceEngine(
            EngineConfig(
                max_batch_size=2,
                max_prefill_tokens_per_step=8,
                cache_block_size=2,
                cache_num_blocks=3,
            )
        )
        engine.submit([1, 2], max_new_tokens=3, request_id="a")
        engine.submit([3, 4], max_new_tokens=3, request_id="b")
        engine.step()

        second = engine.step()

        self.assertEqual([work.request_id for work in second.decode], ["a"])
        self.assertEqual(engine.request("a").status, RequestStatus.DECODE)
        self.assertEqual(engine.request("b").status, RequestStatus.DECODE)

    def test_metrics_snapshot_reports_latency_and_cache_telemetry(self) -> None:
        engine = InferenceEngine(
            EngineConfig(
                max_batch_size=2,
                max_prefill_tokens_per_step=8,
                cache_block_size=4,
                cache_num_blocks=8,
            )
        )
        engine.submit([1, 2, 3], max_new_tokens=2, request_id="a")
        engine.submit([4, 5], max_new_tokens=2, request_id="b")

        engine.run_until_complete()
        snapshot = engine.metrics_snapshot()

        self.assertEqual(snapshot.submitted_requests, 2)
        self.assertEqual(snapshot.completed_requests, 2)
        self.assertEqual(snapshot.failed_requests, 0)
        self.assertEqual(snapshot.total_generated_tokens, 4)
        self.assertGreater(snapshot.step_count, 0)
        self.assertGreaterEqual(snapshot.p99_latency_ms, snapshot.p50_latency_ms)
        self.assertGreater(snapshot.token_throughput_tps, 0)


if __name__ == "__main__":
    unittest.main()
