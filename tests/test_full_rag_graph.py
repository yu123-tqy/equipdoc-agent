import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from equipdoc_agent.agent import build_graph
from equipdoc_agent.agent.graph import _knowledge_filter
from equipdoc_agent.config import Settings


ROOT = Path(__file__).resolve().parents[1]


class FullRagGraphTests(unittest.TestCase):
    def _settings(self):
        return replace(
            Settings.from_env(ROOT),
            demo_mode=False,
            agentic_mode=False,
            rag_db_dir=(ROOT / "runtime/test_full_no_vector_db").resolve(),
        )

    def test_focus_filter_uses_primary_fault_mentioned_first(self):
        filters = _knowledge_filter("为什么滚动体故障的频谱比外圈故障复杂？")
        self.assertEqual(filters, {"equipment": "bearing", "fault_type": "ball"})

    def test_full_knowledge_question_sends_retrieved_evidence_to_llm(self):
        settings = self._settings()
        with patch("equipdoc_agent.agent.graph.ChatOpenAI") as chat_class:
            llm = chat_class.return_value
            llm.invoke.return_value = AIMessage(content="EVIDENCE_IDS: E01,E02,E03,E04")
            graph = build_graph(settings)
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="轴承外圈故障为什么会产生周期性冲击？")],
                    "signal_path": "",
                },
                config={"configurable": {"thread_id": "test_full_rag"}},
            )

        prompt_messages = llm.invoke.call_args.args[0]
        combined_prompt = "\n".join(str(message.content) for message in prompt_messages)
        self.assertIn("EVIDENCE_IDS", combined_prompt)
        self.assertIn("候选证据句", combined_prompt)
        self.assertIn("BPFO", combined_prompt)
        self.assertIn("周期性冲击", result["messages"][-1].content)
        self.assertNotIn("BPFI", result["messages"][-1].content)
        self.assertEqual(len(result["retrieval_hits"]), 5)
        self.assertTrue(all(item.get("chunk_id") for item in result["retrieval_hits"]))

    def test_ball_fault_first_four_cover_spectrum_and_review(self):
        with patch("equipdoc_agent.agent.graph.ChatOpenAI") as chat_class:
            llm = chat_class.return_value
            llm.invoke.return_value = AIMessage(
                content="EVIDENCE_IDS: E01,E02,E03,E04"
            )
            graph = build_graph(self._settings())
            result = graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="为什么滚动体故障的频谱通常比外圈故障复杂，现场还应复核什么？"
                        )
                    ],
                    "signal_path": "",
                },
                config={"configurable": {"thread_id": "test_ball_evidence_order"}},
            )

        answer = result["messages"][-1].content
        for keyword in ("滚动体", "BSF", "调制", "润滑"):
            self.assertIn(keyword, answer)

    def test_missing_evidence_ids_triggers_one_retry(self):
        with patch("equipdoc_agent.agent.graph.ChatOpenAI") as chat_class:
            llm = chat_class.return_value
            llm.invoke.side_effect = [
                AIMessage(content="没有证据ID的第一版"),
                AIMessage(content="EVIDENCE_IDS: E01,E02,E03,E04"),
            ]
            graph = build_graph(self._settings())
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="轴承外圈故障有什么特征？")],
                    "signal_path": "",
                },
                config={"configurable": {"thread_id": "test_full_retry"}},
            )

        final = result["messages"][-1]
        guard = final.response_metadata["equipdoc_answer_guard"]
        self.assertEqual(llm.invoke.call_count, 2)
        self.assertEqual(guard["generation_path"], "retry")

    def test_unknown_evidence_id_triggers_retry(self):
        with patch("equipdoc_agent.agent.graph.ChatOpenAI") as chat_class:
            llm = chat_class.return_value
            llm.invoke.side_effect = [
                AIMessage(content="EVIDENCE_IDS: E98,E99"),
                AIMessage(content="EVIDENCE_IDS: E01,E02,E03,E04"),
            ]
            graph = build_graph(self._settings())
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="轴承外圈故障有什么特征？")],
                    "signal_path": "",
                },
                config={"configurable": {"thread_id": "test_full_selection"}},
            )

        final = result["messages"][-1]
        guard = final.response_metadata["equipdoc_answer_guard"]
        self.assertEqual(llm.invoke.call_count, 2)
        self.assertEqual(guard["generation_path"], "retry")
        self.assertEqual(
            guard["final_citation_validation"]["claim_citation_coverage"], 1.0
        )

    def test_second_invalid_answer_uses_extractive_fallback(self):
        with patch("equipdoc_agent.agent.graph.ChatOpenAI") as chat_class:
            llm = chat_class.return_value
            llm.invoke.side_effect = [
                AIMessage(content="第一版没有证据ID"),
                AIMessage(content="第二版仍然没有证据ID"),
            ]
            graph = build_graph(self._settings())
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="轴承外圈故障有什么特征？")],
                    "signal_path": "",
                },
                config={"configurable": {"thread_id": "test_full_fallback"}},
            )

        final = result["messages"][-1]
        guard = final.response_metadata["equipdoc_answer_guard"]
        self.assertEqual(guard["generation_path"], "extractive_fallback")
        self.assertIn("回答降级", final.content)
        self.assertNotIn("第二版仍然没有证据ID", final.content)


if __name__ == "__main__":
    unittest.main()
