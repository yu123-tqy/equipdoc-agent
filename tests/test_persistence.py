from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

from langgraph.graph import END, START, StateGraph

from equipdoc_agent.persistence import (
    create_checkpointer,
    create_sqlite_checkpointer,
    delete_checkpoint_threads,
)


class _State(TypedDict, total=False):
    value: str


def _build_test_graph(checkpointer):
    builder = StateGraph(_State)
    builder.add_node("copy", lambda state: {"value": state["value"]})
    builder.add_edge(START, "copy")
    builder.add_edge("copy", END)
    return builder.compile(checkpointer=checkpointer)


class PersistenceTests(unittest.TestCase):
    def test_sqlite_checkpoint_survives_new_graph_instance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoints.sqlite3"
            first_saver = create_sqlite_checkpointer(path)
            first_graph = _build_test_graph(first_saver)
            config = {"configurable": {"thread_id": "durable-thread"}}
            first_graph.invoke({"value": "saved"}, config=config)
            first_saver.conn.close()

            second_saver = create_sqlite_checkpointer(path)
            second_graph = _build_test_graph(second_saver)
            snapshot = second_graph.get_state(config)
            self.assertEqual(snapshot.values["value"], "saved")
            second_saver.conn.close()

    def test_postgres_url_selects_production_checkpointer(self):
        marker = object()
        with patch(
            "equipdoc_agent.persistence.create_postgres_checkpointer",
            return_value=marker,
        ) as factory:
            result = create_checkpointer("postgresql+psycopg://user:pass@db/equipdoc")

        self.assertIs(result, marker)
        factory.assert_called_once_with("postgresql://user:pass@db/equipdoc")

    def test_checkpoint_thread_deletion_is_deduplicated(self):
        class Saver:
            def __init__(self):
                self.deleted = []

            def delete_thread(self, thread_id):
                self.deleted.append(thread_id)

        saver = Saver()
        count = delete_checkpoint_threads(saver, ["thread-a", "thread-a", "thread-b"])
        self.assertEqual(count, 2)
        self.assertEqual(saver.deleted, ["thread-a", "thread-b"])


if __name__ == "__main__":
    unittest.main()
