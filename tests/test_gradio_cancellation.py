from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_gradio
from equipdoc_agent.conversation_store import ConversationStore


class GradioCancellationTests(unittest.TestCase):
    def test_busy_state_disables_navigation_and_enables_stop(self):
        busy = app_gradio.begin_submission()
        self.assertEqual(len(busy), 16)
        self.assertIn("处理中", busy[0])
        self.assertFalse(busy[7]["interactive"])
        self.assertTrue(busy[8]["interactive"])
        self.assertFalse(busy[9]["interactive"])

    def test_stop_marks_run_canceled_and_preserves_conversation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "conversations.sqlite3")
            conversation_id = store.create_conversation()
            store.start_run(conversation_id, user_content="检查信号")
            with patch.object(app_gradio, "STORE", store):
                stopped = app_gradio.stop_current_task(conversation_id)
            self.assertIsNone(store.get_active_run(conversation_id))

        self.assertEqual(len(stopped), 24)
        self.assertEqual(stopped[0], conversation_id)
        self.assertIn("已停止当前任务", stopped[4])

    def test_stop_event_cancels_submit_and_both_review_events(self):
        cancellation_targets = [
            dependency.get("cancels")
            for dependency in app_gradio.demo.config.get("dependencies", [])
            if dependency.get("cancels")
        ]
        self.assertTrue(any(len(targets) == 3 for targets in cancellation_targets))


if __name__ == "__main__":
    unittest.main()
