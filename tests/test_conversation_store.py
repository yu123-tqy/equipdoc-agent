from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from equipdoc_agent.conversation_store import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def _store(self, temp_dir: str) -> ConversationStore:
        return ConversationStore(Path(temp_dir) / "conversations.sqlite3")

    def test_conversation_messages_rename_and_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            conversation_id = store.create_conversation()
            run = store.start_run(
                conversation_id,
                user_content="吊舱推进器试验台转速是多少？",
                message_metadata={"signal_file": ""},
            )
            self.assertTrue(
                store.complete_run(
                    run["run_id"],
                    assistant_content="设计范围为0～2000 rpm。",
                    metadata={"retrieval_hits": [{"chunk_id": "c1"}]},
                )
            )

            messages = store.list_messages(conversation_id)
            self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
            self.assertEqual(messages[0]["turn_id"], messages[1]["turn_id"])
            self.assertEqual(messages[1]["metadata"]["retrieval_hits"][0]["chunk_id"], "c1")
            self.assertIn("吊舱推进器试验台转速", store.get_conversation(conversation_id)["title"])

            self.assertTrue(store.rename_conversation(conversation_id, "试验台参数"))
            self.assertEqual(store.get_conversation(conversation_id)["title"], "试验台参数")
            self.assertTrue(store.archive_conversation(conversation_id))
            self.assertEqual(store.list_conversations(), [])

    def test_pending_review_can_resume_and_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            conversation_id = store.create_conversation()
            run = store.start_run(conversation_id, user_content="诊断轴承信号")
            payload = {"name": "diagnose_bearing", "args": {"signal_file": "safe.npy"}}

            self.assertTrue(store.mark_awaiting_review(run["run_id"], payload))
            active = store.get_active_run(conversation_id)
            self.assertEqual(active["status"], "awaiting_review")
            self.assertEqual(active["pending_review"], payload)
            self.assertTrue(store.begin_review_resume(run["run_id"]))
            self.assertTrue(
                store.complete_run(run["run_id"], assistant_content="诊断已完成。")
            )
            self.assertIsNone(store.get_active_run(conversation_id))

    def test_canceled_run_rejects_late_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            conversation_id = store.create_conversation()
            previous_agent_thread = store.get_agent_thread_id(conversation_id)
            run = store.start_run(conversation_id, user_content="一个耗时问题")
            self.assertTrue(store.cancel_active_run(conversation_id))
            self.assertNotEqual(
                store.get_agent_thread_id(conversation_id),
                previous_agent_thread,
            )
            self.assertFalse(
                store.complete_run(run["run_id"], assistant_content="这个结果不应落库")
            )
            messages = store.list_messages(conversation_id)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")

    def test_process_recovery_releases_running_but_keeps_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            running_id = store.create_conversation()
            store.start_run(running_id, user_content="运行中")
            review_id = store.create_conversation()
            review_run = store.start_run(review_id, user_content="待审核")
            store.mark_awaiting_review(review_run["run_id"], {"name": "diagnose_bearing"})

            self.assertEqual(store.recover_interrupted_runs(), 1)
            self.assertIsNone(store.get_active_run(running_id))
            recovered_messages = store.list_messages(running_id)
            self.assertTrue(recovered_messages[-1]["metadata"]["recovered_after_restart"])
            self.assertEqual(store.get_active_run(review_id)["status"], "awaiting_review")

    def test_owner_isolation_covers_conversations_messages_and_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conversations.sqlite3"
            alice = ConversationStore(path, owner_id="alice")
            bob = alice.for_owner("bob")
            alice_id = alice.create_conversation(title="Alice 私有会话")
            run = alice.start_run(alice_id, user_content="仅 Alice 可见")
            alice.complete_run(run["run_id"], assistant_content="私有回答")

            self.assertEqual(bob.list_conversations(), [])
            self.assertIsNone(bob.get_conversation(alice_id))
            self.assertEqual(bob.list_messages(alice_id), [])
            self.assertIsNone(bob.get_active_run(alice_id))
            with self.assertRaises(PermissionError):
                bob.ensure_conversation(alice_id)

    def test_search_pagination_archive_restore_and_permanent_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            ids = []
            for index in range(5):
                conversation_id = store.create_conversation(title=f"巡检 {index}")
                run = store.start_run(
                    conversation_id,
                    user_content=f"泵站关键词 {index}" if index < 3 else f"轴承问题 {index}",
                )
                store.complete_run(run["run_id"], assistant_content="已记录")
                ids.append(conversation_id)

            self.assertEqual(store.count_conversations(search="泵站关键词"), 3)
            first_page = store.list_conversations(search="泵站关键词", limit=2, offset=0)
            second_page = store.list_conversations(search="泵站关键词", limit=2, offset=2)
            self.assertEqual(len(first_page), 2)
            self.assertEqual(len(second_page), 1)

            self.assertTrue(store.archive_conversation(ids[0]))
            self.assertNotIn(ids[0], {item["id"] for item in store.list_conversations()})
            self.assertIn(
                ids[0],
                {item["id"] for item in store.list_conversations(include_archived=True)},
            )
            self.assertTrue(store.restore_conversation(ids[0]))
            thread_ids = store.delete_conversation(ids[0])
            self.assertTrue(thread_ids)
            self.assertIsNone(store.get_conversation(ids[0]))
            self.assertEqual(store.list_messages(ids[0]), [])

    def test_long_conversation_builds_summary_and_rotates_agent_thread(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(
                Path(temp_dir) / "conversations.sqlite3",
                recent_context_messages=2,
                summary_max_chars=800,
                compaction_turns=2,
            )
            conversation_id = store.create_conversation()
            original_thread = store.get_agent_thread_id(conversation_id)
            for index in range(2):
                run = store.start_run(conversation_id, user_content=f"历史问题 {index}")
                store.complete_run(
                    run["run_id"],
                    assistant_content=f"历史回答 {index}",
                    metadata={"agent_memory": {"last_search_query": f"query-{index}"}},
                )

            context = store.conversation_context(conversation_id)
            self.assertIn("历史问题 0", context["summary"])
            self.assertEqual(len(context["recent_messages"]), 2)
            self.assertNotEqual(store.get_agent_thread_id(conversation_id), original_thread)
            self.assertEqual(
                store.latest_agent_memory(conversation_id)["last_search_query"],
                "query-1",
            )
            self.assertEqual(len(store.list_messages(conversation_id)), 4)

    def test_concurrent_start_allows_only_one_active_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            conversation_id = store.create_conversation()
            barrier = threading.Barrier(2)
            outcomes = []

            def start(index):
                barrier.wait()
                try:
                    store.start_run(conversation_id, user_content=f"问题 {index}")
                    outcomes.append("started")
                except RuntimeError:
                    outcomes.append("blocked")

            threads = [threading.Thread(target=start, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertCountEqual(outcomes, ["started", "blocked"])

    def test_existing_phase_one_database_is_migrated_in_place(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conversations.sqlite3"
            import sqlite3

            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, title TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    archived_at TEXT
                );
                CREATE TABLE messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,
                    content TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    status TEXT NOT NULL, pending_review_json TEXT NOT NULL,
                    started_at TEXT NOT NULL, finished_at TEXT
                );
                INSERT INTO conversations
                    (id, owner_id, title, status, created_at, updated_at)
                VALUES ('legacy', 'local', '一期会话', 'active', '2026-01-01', '2026-01-01');
                """
            )
            connection.close()

            store = ConversationStore(path)
            self.assertEqual(store.schema_version(), 2)
            self.assertEqual(store.get_conversation("legacy")["title"], "一期会话")
            self.assertEqual(store.get_agent_thread_id("legacy"), "legacy")


if __name__ == "__main__":
    unittest.main()
