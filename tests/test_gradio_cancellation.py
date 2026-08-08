from __future__ import annotations

import unittest

from app_gradio import begin_submission, demo, stop_current_task


class GradioCancellationTests(unittest.TestCase):
    def test_busy_and_stopped_states_toggle_controls(self):
        busy = begin_submission()
        self.assertEqual(len(busy), 10)
        self.assertIn("处理中", busy[0])
        self.assertFalse(busy[-2]["interactive"])
        self.assertTrue(busy[-1]["interactive"])

        stopped = stop_current_task()
        self.assertEqual(len(stopped), 11)
        self.assertEqual(stopped[0], "")
        self.assertIn("已停止当前任务", stopped[1])
        self.assertTrue(stopped[-2]["interactive"])
        self.assertFalse(stopped[-1]["interactive"])

    def test_stop_event_cancels_submit_and_both_review_events(self):
        cancellation_targets = [
            dependency.get("cancels")
            for dependency in demo.config.get("dependencies", [])
            if dependency.get("cancels")
        ]
        self.assertTrue(any(len(targets) == 3 for targets in cancellation_targets))


if __name__ == "__main__":
    unittest.main()
