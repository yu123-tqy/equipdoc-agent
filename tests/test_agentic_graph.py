from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from equipdoc_agent.agent.agentic_graph import (
    _collect_search_hits,
    _merge_memory_after_tool,
    build_agentic_graph,
)
from equipdoc_agent.config import Settings


class _FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.outputs:
            raise AssertionError("Unexpected LLM call")
        output = self.outputs.pop(0)
        return output if isinstance(output, AIMessage) else AIMessage(content=output)


class _FakeRetriever:
    warnings = []

    def search(self, query, filters=None, top_k=None):
        return [
            {
                "doc_id": "bearing_outer_race_fault",
                "chunk_id": "bearing_outer_race_fault_c001",
                "title": "轴承外圈故障",
                "text": "外圈缺陷会产生周期性冲击。",
                "rrf_score": 0.02,
                "lexical_score": 4.0,
            },
            {
                "doc_id": "bearing_review",
                "chunk_id": "bearing_review_c001",
                "title": "轴承现场复核",
                "text": "现场应复核安装松动、润滑状态、温度和异常噪声。",
                "rrf_score": 0.01,
                "lexical_score": 2.0,
            },
            {
                "doc_id": "maintenance_boundary",
                "chunk_id": "maintenance_boundary_c001",
                "title": "诊断边界",
                "text": (
                    "样本和工况差异会导致模型置信度变化；"
                    "模型置信度不能替代现场复核，也不足以直接推断剩余寿命。"
                ),
                "rrf_score": 0.005,
                "lexical_score": 1.0,
            },
        ][:top_k]


def _knowledge_plan():
    return json.dumps(
        {
            "intent": "knowledge_qa",
            "confidence": 0.9,
            "equipment": "bearing",
            "missing_fields": [],
            "clarification_question": "",
            "plan": [
                {
                    "step_id": "S1",
                    "tool": "search_maintenance_knowledge",
                    "arguments": {
                        "query": "轴承外圈故障",
                        "equipment": "bearing",
                        "fault_type": "outer_race",
                        "top_k": 3,
                    },
                    "depends_on": [],
                }
            ],
        },
        ensure_ascii=False,
    )


def _diagnosis_plan():
    return json.dumps(
        {
            "intent": "diagnosis",
            "confidence": 0.95,
            "equipment": "bearing",
            "missing_fields": [],
            "clarification_question": "",
            "plan": [
                {
                    "step_id": "S1",
                    "tool": "diagnose_bearing",
                    "arguments": {},
                    "depends_on": [],
                }
            ],
        },
        ensure_ascii=False,
    )


def _inspection_plan():
    return json.dumps(
        {
            "intent": "signal_inspection",
            "confidence": 0.95,
            "equipment": "bearing",
            "missing_fields": [],
            "clarification_question": "",
            "plan": [
                {
                    "step_id": "S1",
                    "tool": "inspect_signal",
                    "arguments": {},
                    "depends_on": [],
                }
            ],
        },
        ensure_ascii=False,
    )


def _observer_answer():
    return json.dumps(
        {
            "action": "answer",
            "tool": None,
            "arguments": {},
            "reason": "已有足够结果",
            "clarification_question": "",
        },
        ensure_ascii=False,
    )


def _observer_clarify():
    return json.dumps(
        {
            "action": "clarify",
            "tool": None,
            "arguments": {},
            "reason": "还需要更多工况",
            "clarification_question": "请补充转速、负荷和润滑状况。",
        },
        ensure_ascii=False,
    )


def _grounded_draft():
    return (
        "## 综合解释\n\n"
        "外圈缺陷会产生周期性冲击 "
        "[bearing_outer_race_fault#bearing_outer_race_fault_c001]"
    )


def _grounded_draft_with_review():
    return (
        "## 综合解释\n\n"
        "外圈缺陷会产生周期性冲击 "
        "[bearing_outer_race_fault#bearing_outer_race_fault_c001]\n\n"
        "## 现场复核\n\n"
        "现场应复核安装松动、润滑状态、温度和异常噪声 "
        "[bearing_review#bearing_review_c001]"
    )


def _grounded_confidence_draft():
    return (
        "样本和工况差异会导致模型置信度变化 "
        "[maintenance_boundary#maintenance_boundary_c001]\n\n"
        "模型置信度不能替代现场复核，也不足以直接推断剩余寿命 "
        "[maintenance_boundary#maintenance_boundary_c001]"
    )


class AgenticGraphTests(unittest.TestCase):
    def _settings(self, root: Path, **overrides):
        settings = replace(
            Settings.from_env(root),
            demo_mode=False,
            agentic_mode=True,
            rag_db_dir=(root / "runtime/no_vector_db").resolve(),
        )
        return replace(settings, **overrides)

    def test_later_targeted_search_hits_are_not_truncated(self):
        observations = [
            {
                "_tool_name": "search_maintenance_knowledge",
                "hits": [{"chunk_id": f"generic_{index}"} for index in range(5)],
            },
            {
                "_tool_name": "search_maintenance_knowledge",
                "hits": [{"chunk_id": "targeted", "focused_match": True}],
            },
        ]
        merged = _collect_search_hits(observations)
        self.assertEqual(len(merged), 6)
        self.assertIn("targeted", {item["chunk_id"] for item in merged})

    def test_failed_tool_is_not_recorded_as_completed(self):
        memory = _merge_memory_after_tool(
            {"last_evidence": [{"citation": "old#c001"}]},
            {
                "_tool_name": "search_maintenance_knowledge",
                "status": "error",
                "error": "temporary failure",
            },
        )
        self.assertNotIn("search_maintenance_knowledge", memory.get("completed_tools", []))
        self.assertIn("search_maintenance_knowledge", memory["attempted_tools"])
        self.assertIn("search_maintenance_knowledge", memory["failed_tools"])
        self.assertEqual(memory["last_evidence"], [{"citation": "old#c001"}])

    def test_knowledge_plan_runs_search_without_review(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            settings = self._settings(Path(temp_dir))
            llm = _FakeLLM(
                [
                    _knowledge_plan(),
                    _grounded_draft(),
                ]
            )
            graph = build_agentic_graph(
                settings,
                llm=llm,
                retriever=_FakeRetriever(),
            )
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="轴承外圈故障有什么特征？")],
                },
                config={"configurable": {"thread_id": "agentic_knowledge"}},
            )

        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["tool_step_count"], 1)
        self.assertEqual(
            result["tool_observations"][0]["_tool_name"],
            "search_maintenance_knowledge",
        )
        self.assertIn("周期性冲击", result["messages"][-1].content)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result["answer_metadata"]["llm_call_count"], 2)
        self.assertEqual(
            result["answer_metadata"]["answer_guard"]["generation_path"],
            "grounded_synthesis",
        )
        self.assertEqual(len(result["retrieval_hits"]), 3)
        self.assertEqual(
            result["retrieval_hits"][0]["chunk_id"],
            "bearing_outer_race_fault_c001",
        )

    def test_overclarified_knowledge_question_is_retried_and_searched(self):
        overclarification = json.dumps(
            {
                "intent": "clarification",
                "confidence": 0.8,
                "equipment": "bearing",
                "missing_fields": ["operating_condition"],
                "clarification_question": "请补充具体工况。",
                "plan": [],
            },
            ensure_ascii=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            settings = self._settings(Path(temp_dir))
            llm = _FakeLLM(
                [
                    overclarification,
                    _knowledge_plan(),
                    _grounded_draft_with_review(),
                ]
            )
            graph = build_agentic_graph(
                settings,
                llm=llm,
                retriever=_FakeRetriever(),
            )
            result = graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="轴承外圈局部故障为什么会产生周期性冲击，现场应复核什么？"
                        )
                    ],
                },
                config={"configurable": {"thread_id": "agentic_overclarification"}},
            )

        self.assertEqual(result["current_plan"]["intent"], "knowledge_qa")
        self.assertEqual(result["tool_step_count"], 1)
        self.assertEqual(
            result["tool_observations"][0]["_tool_name"],
            "search_maintenance_knowledge",
        )
        self.assertEqual(result["planning_metadata"]["generation_path"], "retry")
        self.assertEqual(result["planning_metadata"]["attempts"], 2)
        self.assertIn(
            "self-contained maintenance knowledge question",
            result["planning_metadata"]["validation_errors"][0],
        )
        self.assertEqual(len(llm.calls), 3)

    def test_inspect_signal_is_read_only_and_does_not_interrupt(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "inspect.npy"
            np.save(sample, np.arange(32, dtype=np.float32))
            settings = self._settings(root)
            llm = _FakeLLM([_inspection_plan()])
            graph = build_agentic_graph(settings, llm=llm)
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="检查这段信号的 RMS 和采样点")],
                    "signal_path": str(sample),
                },
                config={"configurable": {"thread_id": "agentic_inspect"}},
            )

        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["tool_observations"][0]["status"], "ok")
        self.assertIn("RMS", result["messages"][-1].content)
        self.assertIn("inspect.npy", result["messages"][-1].content)

    def test_diagnosis_requires_approve_and_reject_skips_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "diagnose.npy"
            np.save(sample, np.arange(1024, dtype=np.float32))
            settings = self._settings(root)

            approve_llm = _FakeLLM([_diagnosis_plan(), _grounded_draft()])
            approve_graph = build_agentic_graph(settings, llm=approve_llm)
            config = {"configurable": {"thread_id": "agentic_approve"}}

            def execute_approved(tool_name, arguments, **_):
                if tool_name == "diagnose_bearing":
                    return {
                        "_tool_name": "diagnose_bearing",
                        "status": "ok",
                        "signal_file": "diagnose.npy",
                        "fault_type": "外圈故障",
                        "confidence": 0.62,
                        "probabilities": {},
                        "signal": {"samples": 1024, "rms": 1.0},
                        "warning": "需要现场复核",
                    }
                return {
                    "_tool_name": "search_maintenance_knowledge",
                    "status": "ok",
                    "query": arguments["query"],
                    "hits": _FakeRetriever().search(arguments["query"], top_k=5),
                }

            with patch(
                "equipdoc_agent.agent.agentic_graph.execute_agentic_tool",
                side_effect=execute_approved,
            ) as execute:
                pending = approve_graph.invoke(
                    {
                        "messages": [HumanMessage(content="请诊断轴承信号")],
                        "signal_path": str(sample),
                    },
                    config=config,
                )
                self.assertIn("__interrupt__", pending)
                result = approve_graph.invoke(Command(resume="approve"), config=config)
                self.assertEqual(execute.call_count, 2)
            self.assertEqual(result["session_memory"]["last_diagnosis"]["confidence"], 0.62)
            self.assertIn("62.00%", result["messages"][-1].content)

            reject_llm = _FakeLLM([_diagnosis_plan()])
            reject_graph = build_agentic_graph(settings, llm=reject_llm)
            reject_config = {"configurable": {"thread_id": "agentic_reject"}}
            with patch("equipdoc_agent.agent.agentic_graph.execute_agentic_tool") as execute:
                reject_graph.invoke(
                    {
                        "messages": [HumanMessage(content="请诊断轴承信号")],
                        "signal_path": str(sample),
                    },
                    config=reject_config,
                )
                rejected = reject_graph.invoke(
                    Command(resume="reject"),
                    config=reject_config,
                )
                execute.assert_not_called()
            self.assertIn("取消", rejected["messages"][-1].content)

    def test_explicit_signal_diagnosis_retries_a_knowledge_only_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "diagnose.npy"
            np.save(sample, np.arange(1024, dtype=np.float32))
            settings = self._settings(root)
            llm = _FakeLLM([_knowledge_plan(), _diagnosis_plan()])
            graph = build_agentic_graph(settings, llm=llm)
            pending = graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=("先用分类模型分析这段轴承振动，再用知识证据解释结果。")
                        )
                    ],
                    "signal_path": str(sample),
                },
                config={"configurable": {"thread_id": "agentic_diagnosis_retry"}},
            )

        self.assertIn("__interrupt__", pending)
        self.assertEqual(pending["current_plan"]["intent"], "diagnosis")
        self.assertEqual(pending["planning_metadata"]["generation_path"], "retry")
        self.assertIn(
            "requires diagnosis",
            pending["planning_metadata"]["validation_errors"][0],
        )

    def test_signal_file_dependency_is_safely_normalized_before_review(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "diagnose.npy"
            np.save(sample, np.arange(1024, dtype=np.float32))
            settings = self._settings(root)
            raw_plan = json.loads(_diagnosis_plan())
            raw_plan["plan"][0]["depends_on"] = ["signal_file"]
            llm = _FakeLLM([json.dumps(raw_plan, ensure_ascii=False)])
            graph = build_agentic_graph(settings, llm=llm)
            pending = graph.invoke(
                {
                    "messages": [HumanMessage(content="请诊断这段轴承振动信号。")],
                    "signal_path": str(sample),
                },
                config={"configurable": {"thread_id": "agentic_signal_dependency"}},
            )

        self.assertIn("__interrupt__", pending)
        self.assertEqual(pending["planning_metadata"]["generation_path"], "first_pass")
        self.assertEqual(pending["current_plan"]["plan"][0]["depends_on"], [])
        self.assertEqual(
            pending["current_plan"]["validation"]["removed_dependencies"],
            [{"step_id": "S1", "dependency": "signal_file"}],
        )

    def test_completed_single_tool_skips_observer_at_max_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "limit.npy"
            np.save(sample, np.arange(32, dtype=np.float32))
            settings = self._settings(root, agent_max_steps=1)
            llm = _FakeLLM([_inspection_plan()])
            graph = build_agentic_graph(settings, llm=llm)
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="检查这段信号")],
                    "signal_path": str(sample),
                },
                config={"configurable": {"thread_id": "agentic_limit"}},
            )

        self.assertEqual(result["tool_step_count"], 1)
        self.assertEqual(result["observation_metadata"], {})
        self.assertEqual(len(llm.calls), 1)

    def test_same_thread_memory_reaches_the_next_planner(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "memory.npy"
            np.save(sample, np.arange(1024, dtype=np.float32))
            settings = self._settings(root)
            llm = _FakeLLM(
                [
                    _diagnosis_plan(),
                    _grounded_draft(),
                    _knowledge_plan(),
                    _grounded_confidence_draft(),
                ]
            )
            graph = build_agentic_graph(
                settings,
                llm=llm,
                retriever=_FakeRetriever(),
            )
            config = {"configurable": {"thread_id": "agentic_memory"}}

            def execute_memory(tool_name, arguments, **_):
                if tool_name == "diagnose_bearing":
                    return {
                        "_tool_name": "diagnose_bearing",
                        "status": "ok",
                        "signal_file": "memory.npy",
                        "fault_type": "外圈故障",
                        "confidence": 0.62,
                        "probabilities": {},
                        "signal": {"samples": 1024},
                        "warning": "采样长度较短",
                    }
                return {
                    "_tool_name": "search_maintenance_knowledge",
                    "status": "ok",
                    "query": arguments["query"],
                    "hits": _FakeRetriever().search(arguments["query"], top_k=5),
                }

            with patch(
                "equipdoc_agent.agent.agentic_graph.execute_agentic_tool",
                side_effect=execute_memory,
            ):
                graph.invoke(
                    {
                        "messages": [HumanMessage(content="请诊断轴承信号")],
                        "signal_path": str(sample),
                    },
                    config=config,
                )
                graph.invoke(Command(resume="approve"), config=config)
            graph.invoke(
                {"messages": [HumanMessage(content="置信度为什么不高？")]},
                config=config,
            )

        second_planner_prompt = "\n".join(str(message.content) for message in llm.calls[2])
        self.assertIn("last_diagnosis", second_planner_prompt)
        self.assertIn("0.62", second_planner_prompt)
        self.assertIn("外圈故障", second_planner_prompt)

    def test_diagnosis_result_cannot_be_hidden_by_clarification(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "chain.npy"
            np.save(sample, np.arange(1024, dtype=np.float32))
            settings = self._settings(root)
            llm = _FakeLLM(
                [
                    _diagnosis_plan(),
                    _grounded_draft(),
                ]
            )
            graph = build_agentic_graph(settings, llm=llm)
            config = {"configurable": {"thread_id": "agentic_chain"}}

            def execute(tool_name, arguments, **_):
                if tool_name == "diagnose_bearing":
                    return {
                        "_tool_name": "diagnose_bearing",
                        "status": "ok",
                        "signal_file": "chain.npy",
                        "fault_type": "外圈故障",
                        "confidence": 0.62,
                        "signal": {"samples": 1024},
                        "warning": "需要现场复核",
                    }
                return {
                    "_tool_name": "search_maintenance_knowledge",
                    "status": "ok",
                    "query": arguments["query"],
                    "filters": {
                        "equipment": "bearing",
                        "fault_type": "outer_race",
                    },
                    "hits": [
                        {
                            "doc_id": "bearing_outer_race_fault",
                            "chunk_id": "bearing_outer_race_fault_c001",
                            "citation": ("bearing_outer_race_fault#bearing_outer_race_fault_c001"),
                            "title": "轴承外圈故障",
                            "text": "外圈缺陷会产生周期性冲击。",
                        }
                    ],
                    "warnings": [],
                }

            with patch(
                "equipdoc_agent.agent.agentic_graph.execute_agentic_tool",
                side_effect=execute,
            ):
                graph.invoke(
                    {
                        "messages": [HumanMessage(content="诊断并解释这段轴承信号")],
                        "signal_path": str(sample),
                    },
                    config=config,
                )
                result = graph.invoke(Command(resume="approve"), config=config)

        self.assertEqual(result["tool_step_count"], 2)
        self.assertEqual(
            [item["_tool_name"] for item in result["tool_observations"]],
            ["diagnose_bearing", "search_maintenance_knowledge"],
        )
        self.assertIn("工具观察", result["messages"][-1].content)
        self.assertIn("故障类别", result["messages"][-1].content)
        self.assertIn("综合解释", result["messages"][-1].content)

    def test_two_invalid_synthesis_drafts_use_structured_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            settings = self._settings(Path(temp_dir))
            llm = _FakeLLM(
                [
                    _knowledge_plan(),
                    (
                        "外圈缺陷会产生周期性冲击 "
                        "[bearing_outer_race_fault#bearing_outer_race_fault_c001]\n"
                        "现场还需要继续检查。"
                    ),
                    "第二版仍然没有引用",
                ]
            )
            graph = build_agentic_graph(
                settings,
                llm=llm,
                retriever=_FakeRetriever(),
            )
            result = graph.invoke(
                {"messages": [HumanMessage(content="外圈故障有什么特征？")]},
                config={"configurable": {"thread_id": "agentic_synthesis_fallback"}},
            )

        final = result["messages"][-1]
        self.assertIn("按证据组织的回答", final.content)
        self.assertIn("周期性冲击", final.content)
        self.assertNotIn("第二版仍然没有引用", final.content)
        self.assertEqual(
            result["answer_metadata"]["answer_guard"]["generation_path"],
            "structured_evidence_answer",
        )
        self.assertEqual(
            len(result["answer_metadata"]["answer_guard"]["synthesis_validations"]),
            2,
        )

    def test_unsupported_synthesis_uses_structured_evidence_without_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            settings = self._settings(Path(temp_dir))
            llm = _FakeLLM(
                [
                    _knowledge_plan(),
                    (
                        "外圈缺陷每转只冲击一次 "
                        "[bearing_outer_race_fault#bearing_outer_race_fault_c001]"
                    ),
                ]
            )
            graph = build_agentic_graph(
                settings,
                llm=llm,
                retriever=_FakeRetriever(),
            )
            result = graph.invoke(
                {"messages": [HumanMessage(content="外圈故障有什么特征？")]},
                config={"configurable": {"thread_id": "agentic_synthesis_no_retry"}},
            )

        guard = result["answer_metadata"]["answer_guard"]
        self.assertEqual(guard["generation_path"], "structured_evidence_answer")
        self.assertEqual(guard["synthesis_attempts"], 1)
        self.assertEqual(len(guard["synthesis_validations"]), 1)
        self.assertTrue(guard["synthesis_validation"]["unsupported_claims"])
        self.assertEqual(len(llm.calls), 2)

    def test_invalid_synthesis_is_retried_once_before_success(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            settings = self._settings(Path(temp_dir))
            llm = _FakeLLM(
                [
                    _knowledge_plan(),
                    (
                        "外圈缺陷会产生周期性冲击 "
                        "[bearing_outer_race_fault#bearing_outer_race_fault_c001]\n"
                        "现场还需要继续检查。"
                    ),
                    _grounded_draft(),
                ]
            )
            graph = build_agentic_graph(
                settings,
                llm=llm,
                retriever=_FakeRetriever(),
            )
            result = graph.invoke(
                {"messages": [HumanMessage(content="外圈故障有什么特征？")]},
                config={"configurable": {"thread_id": "agentic_synthesis_retry"}},
            )

        final = result["messages"][-1]
        self.assertIn("综合解释", final.content)
        self.assertNotIn("第一版没有引用", final.content)
        self.assertEqual(
            result["answer_metadata"]["answer_guard"]["generation_path"],
            "grounded_synthesis_retry",
        )
        self.assertEqual(
            result["answer_metadata"]["answer_guard"]["synthesis_attempts"],
            2,
        )
        self.assertEqual(
            len(result["answer_metadata"]["answer_guard"]["synthesis_validations"]),
            2,
        )

    def test_two_invalid_plans_use_visible_deterministic_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            settings = self._settings(Path(temp_dir))
            llm = _FakeLLM(
                [
                    "not json",
                    '{"intent":"unknown"}',
                    _grounded_draft(),
                ]
            )
            graph = build_agentic_graph(
                settings,
                llm=llm,
                retriever=_FakeRetriever(),
            )
            result = graph.invoke(
                {"messages": [HumanMessage(content="外圈故障有什么特征？")]},
                config={"configurable": {"thread_id": "agentic_plan_fallback"}},
            )

        self.assertEqual(
            result["planning_metadata"]["generation_path"],
            "deterministic_fallback",
        )
        self.assertEqual(result["planning_metadata"]["attempts"], 2)
        self.assertIn("规划降级", result["messages"][-1].content)

    def test_tool_error_can_lead_to_an_explicit_clarification(self):
        clarify = json.dumps(
            {
                "action": "clarify",
                "tool": None,
                "arguments": {},
                "reason": "工具读取失败",
                "clarification_question": "请检查信号文件是否有效后重新上传。",
            },
            ensure_ascii=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            sample = sample_root / "broken.npy"
            np.save(sample, np.arange(32, dtype=np.float32))
            settings = self._settings(root)
            llm = _FakeLLM([_inspection_plan(), clarify])
            graph = build_agentic_graph(settings, llm=llm)
            with patch(
                "equipdoc_agent.agent.agentic_graph.execute_agentic_tool",
                return_value={
                    "_tool_name": "inspect_signal",
                    "status": "error",
                    "error": "SignalValidationError: invalid signal",
                },
            ):
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content="检查这段信号")],
                        "signal_path": str(sample),
                    },
                    config={"configurable": {"thread_id": "agentic_tool_error"}},
                )

        self.assertIn("需要补充信息", result["messages"][-1].content)
        self.assertIn("重新上传", result["messages"][-1].content)
        self.assertIn(
            "重新上传",
            result["session_memory"]["pending_clarification"],
        )

    def test_new_signal_clears_old_diagnosis_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            old_sample = sample_root / "old.npy"
            new_sample = sample_root / "new.npy"
            np.save(old_sample, np.arange(32, dtype=np.float32))
            np.save(new_sample, np.arange(64, dtype=np.float32))
            settings = self._settings(root)
            llm = _FakeLLM(
                [
                    _inspection_plan(),
                    _inspection_plan(),
                ]
            )
            graph = build_agentic_graph(settings, llm=llm)
            config = {"configurable": {"thread_id": "agentic_new_file"}}
            graph.invoke(
                {
                    "messages": [HumanMessage(content="检查旧信号")],
                    "signal_path": str(old_sample),
                    "session_memory": {
                        "signal_file": "old.npy",
                        "last_diagnosis": {
                            "fault_type": "外圈故障",
                            "confidence": 0.62,
                        },
                    },
                },
                config=config,
            )
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="换一个文件重新检查")],
                    "signal_path": str(new_sample),
                },
                config=config,
            )

        self.assertEqual(result["session_memory"]["signal_file"], "new.npy")
        self.assertNotIn("last_diagnosis", result["session_memory"])
        second_planner_context = str(llm.calls[1][-1].content)
        self.assertNotIn("last_diagnosis", second_planner_context)
        self.assertNotIn("外圈故障", second_planner_context)

    def test_requesting_a_replacement_without_upload_does_not_reuse_old_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            sample_root = root / "data/samples"
            sample_root.mkdir(parents=True)
            old_sample = sample_root / "old.npy"
            np.save(old_sample, np.arange(32, dtype=np.float32))
            settings = self._settings(root)
            llm = _FakeLLM([_inspection_plan()])
            graph = build_agentic_graph(settings, llm=llm)
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="换一个文件重新检查")],
                    "signal_path": str(old_sample),
                    "session_memory": {
                        "signal_file": "old.npy",
                        "last_diagnosis": {
                            "fault_type": "外圈故障",
                            "confidence": 0.62,
                        },
                    },
                },
                config={"configurable": {"thread_id": "agentic_replace_missing"}},
            )

        self.assertEqual(result["signal_path"], "")
        self.assertEqual(result["current_plan"]["intent"], "clarification")
        self.assertNotIn("last_diagnosis", result["session_memory"])
        self.assertIn("上传", result["messages"][-1].content)

    def test_safety_boundary_runs_before_the_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            settings = self._settings(Path(temp_dir))
            llm = _FakeLLM([])
            graph = build_agentic_graph(settings, llm=llm)
            result = graph.invoke(
                {"messages": [HumanMessage(content="精确告诉我轴承还能运行多少天")]},
                config={"configurable": {"thread_id": "agentic_safety"}},
            )
        self.assertEqual(llm.calls, [])
        self.assertEqual(result["tool_step_count"] if "tool_step_count" in result else 0, 0)
        self.assertIn("安全边界", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
