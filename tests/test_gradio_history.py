from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

import app_gradio
from equipdoc_agent.conversation_store import ConversationStore


class _FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, payload, config):
        self.calls += 1
        return {
            "messages": [AIMessage(content=f"回答{self.calls}")],
            "retrieval_hits": [
                {
                    "doc_id": "bearing_doc",
                    "chunk_id": f"chunk_{self.calls}",
                    "title": "轴承资料",
                    "heading_path": "故障机理",
                    "text": "可回查证据。",
                    "rrf_score": 0.5,
                }
            ],
        }


class GradioHistoryTests(unittest.TestCase):
    def test_submit_persists_multi_turn_history_and_retrieval_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conversations.sqlite3"
            store = ConversationStore(path)
            conversation_id = store.create_conversation()
            agent = _FakeAgent()
            with patch.object(app_gradio, "STORE", store), patch.object(
                app_gradio, "AGENT", agent
            ):
                first = app_gradio.submit("第一问", None, False, conversation_id)
                second = app_gradio.submit("第二问", None, False, conversation_id)

            self.assertEqual(first[0], conversation_id)
            self.assertEqual(len(first[3]), 2)
            self.assertEqual(len(second[3]), 4)
            self.assertEqual(second[3][-2]["content"], "第二问")
            self.assertEqual(second[3][-1]["content"], "回答2")
            self.assertEqual(second[8][0]["chunk_id"], "chunk_2")

            reopened = ConversationStore(path)
            persisted = reopened.list_messages(conversation_id)
            self.assertEqual(len(persisted), 4)
            self.assertEqual(persisted[-1]["metadata"]["retrieval_hits"][0]["chunk_id"], "chunk_2")

    def test_loading_pending_conversation_restores_review_controls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "conversations.sqlite3")
            conversation_id = store.create_conversation()
            run = store.start_run(conversation_id, user_content="请诊断信号")
            store.mark_awaiting_review(
                run["run_id"],
                {"name": "diagnose_bearing", "args": {"signal_file": "safe.npy"}},
            )
            with patch.object(app_gradio, "STORE", store):
                view = app_gradio.load_conversation(conversation_id)

        self.assertEqual(view[0], conversation_id)
        self.assertEqual(view[4], "等待人工审核")
        self.assertIn("safe.npy", view[5])
        self.assertTrue(view[6]["interactive"])
        self.assertTrue(view[7]["interactive"])

    def test_search_and_delete_flow_updates_visible_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "conversations.sqlite3")
            keep_id = store.create_conversation(title="保留：轴承")
            delete_id = store.create_conversation(title="删除：泵站")
            with patch.object(app_gradio, "STORE", store), patch.object(
                app_gradio, "delete_checkpoint_threads", return_value=1
            ):
                filtered = app_gradio.filter_history("泵站", False, 0, keep_id)
                deleted = app_gradio.permanently_delete_conversation(
                    delete_id, True, "泵站", False, 0
                )
            self.assertIsNone(store.get_conversation(delete_id))

        self.assertEqual(filtered[0], delete_id)
        self.assertEqual(len(filtered[1]["choices"]), 1)
        self.assertIn("永久删除", deleted[4])


if __name__ == "__main__":
    unittest.main()
