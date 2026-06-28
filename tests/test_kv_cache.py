import unittest

from llm_engine.kv_cache import KVCacheError, PagedKVCacheManager


class PagedKVCacheManagerTests(unittest.TestCase):
    def test_append_allocates_pages_lazily_and_frees_them(self) -> None:
        cache = PagedKVCacheManager(block_size=4, num_blocks=3)

        self.assertEqual(cache.append_tokens("r1", 3), (0,))
        self.assertEqual(cache.block_table("r1"), (0,))
        self.assertEqual(cache.append_tokens("r1", 2), (1,))
        self.assertEqual(cache.block_table("r1"), (0, 1))
        self.assertEqual(cache.stats().used_blocks, 2)

        freed = cache.free_request("r1")
        self.assertEqual(freed, (0, 1))
        self.assertEqual(cache.stats().used_blocks, 0)
        self.assertEqual(cache.stats().free_blocks, 3)

    def test_oom_does_not_mutate_existing_state(self) -> None:
        cache = PagedKVCacheManager(block_size=2, num_blocks=2)
        cache.append_tokens("r1", 4)

        with self.assertRaises(KVCacheError):
            cache.append_tokens("r2", 1)

        self.assertEqual(cache.block_table("r1"), (0, 1))
        self.assertEqual(cache.stats().used_blocks, 2)
        self.assertFalse(cache.has_request("r2"))

    def test_can_allocate_accounts_for_partial_last_block(self) -> None:
        cache = PagedKVCacheManager(block_size=4, num_blocks=1)
        cache.append_tokens("r1", 3)

        self.assertTrue(cache.can_allocate_tokens("r1", 1))
        self.assertFalse(cache.can_allocate_tokens("r1", 2))

    def test_stats_report_internal_and_external_fragmentation(self) -> None:
        cache = PagedKVCacheManager(block_size=4, num_blocks=4)
        cache.append_tokens("a", 3)
        cache.append_tokens("b", 4)
        cache.append_tokens("c", 1)
        cache.free_request("b")

        stats = cache.stats()

        self.assertEqual(stats.allocated_tokens, 4)
        self.assertEqual(stats.allocated_token_slots, 8)
        self.assertEqual(stats.internal_fragmentation_tokens, 4)
        self.assertEqual(stats.free_blocks, 2)
        self.assertEqual(stats.free_block_runs, 2)
        self.assertAlmostEqual(stats.external_fragmentation_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
