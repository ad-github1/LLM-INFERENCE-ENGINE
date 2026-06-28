import unittest

from llm_engine.benchmark import BenchmarkConfig, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_completes_variable_length_requests(self) -> None:
        snapshot = run_benchmark(
            BenchmarkConfig(
                request_count=25,
                seed=99,
                min_prompt_tokens=1,
                max_prompt_tokens=12,
                min_new_tokens=1,
                max_new_tokens=4,
                max_batch_size=4,
                prefill_budget=16,
                cache_block_size=4,
                cache_num_blocks=32,
            )
        )

        self.assertEqual(snapshot.submitted_requests, 25)
        self.assertEqual(snapshot.completed_requests, 25)
        self.assertEqual(snapshot.failed_requests, 0)
        self.assertGreater(snapshot.total_generated_tokens, 0)
        self.assertLessEqual(snapshot.max_batch_size_seen, 4)


if __name__ == "__main__":
    unittest.main()
