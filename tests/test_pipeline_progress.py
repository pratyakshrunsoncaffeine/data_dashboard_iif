import threading
import time
import unittest

from iif_research.pipeline import _collect_concurrently


class ConcurrentProgressTests(unittest.TestCase):
    def test_progress_callback_runs_on_calling_thread_and_keeps_provider_order(self):
        main_thread = threading.get_ident()
        callback_threads = []
        messages = []

        def slow_provider(config, progress):
            progress("Economy", 1)
            time.sleep(0.03)
            return "economy", []

        def fast_provider(config, progress):
            progress("Index board", 1)
            return "market", []

        results = _collect_concurrently(
            (slow_provider, fast_provider),
            None,
            lambda message: (callback_threads.append(threading.get_ident()), messages.append(message)),
        )

        self.assertEqual(results, [("economy", []), ("market", [])])
        self.assertCountEqual(messages, ["Economy | 1", "Index board | 1"])
        self.assertTrue(callback_threads)
        self.assertTrue(all(thread_id == main_thread for thread_id in callback_threads))


if __name__ == "__main__":
    unittest.main()
