from __future__ import annotations

import json
import unittest

from equipdoc_agent.agent.planning import (
    PlanningValidationError,
    build_intent_plan_messages,
    build_intent_plan_retry_messages,
    build_observation_messages,
    extract_json_object,
    fallback_plan,
    parse_and_validate_plan,
    parse_observation_decision,
)


def _plan_text(**overrides) -> str:
    payload = {
        "intent": "knowledge_qa",
        "confidence": 0.91,
        "equipment": "bearing",
        "missing_fields": [],
        "clarification_question": "",
        "plan": [
            {
                "step_id": "S1",
                "tool": "search_maintenance_knowledge",
                "arguments": {
                    "query": "轴承外圈故障 周期性冲击 现场复核",
                    "equipment": "bearing",
                    "fault_type": "outer_race",
                    "top_k": 5,
                },
                "depends_on": [],
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class AgenticPlanningTests(unittest.TestCase):
    def test_extracts_json_from_code_fence_and_surrounding_text(self):
        raw = '说明文字\n```json\n{"intent":"knowledge_qa","nested":{"text":"{}"}}\n```\n结束'
        self.assertEqual(extract_json_object(raw)["nested"]["text"], "{}")

    def test_valid_knowledge_plan_is_normalized(self):
        plan = parse_and_validate_plan(_plan_text(), max_steps=3)
        self.assertEqual(plan["intent"], "knowledge_qa")
        self.assertEqual(plan["plan"][0]["tool"], "search_maintenance_knowledge")
        self.assertEqual(plan["plan"][0]["arguments"]["top_k"], 5)
        self.assertEqual(plan["validation"]["source"], "model")

    def test_project_plan_aliases_and_decorations_are_safely_normalized(self):
        payload = {
            "intent": "project_qa",
            "confidence": "90",
            "equipment": "bearing_test_rig",
            "missing_fields": [],
            "clarification_question": "",
            "reasoning": "查询试验台技术参数",
            "plan": [
                {
                    "step_id": "检索步骤一",
                    "tool": "query_rag",
                    "arguments": {
                        "query": "",
                        "equipment": "podded_propulsor_thrust_bearing",
                        "fault_type": "multi_fault",
                        "top_k": "9",
                        "source_id": "pod_thrust_bearing_plan",
                    },
                    "depends_on": [],
                    "description": "检索项目方案",
                }
            ],
        }
        question = "吊舱推进器推力轴承故障诊断试验台的设计转速范围是多少？"
        plan = parse_and_validate_plan(
            json.dumps(payload, ensure_ascii=False),
            user_text=question,
        )

        self.assertEqual(plan["intent"], "knowledge_qa")
        self.assertEqual(plan["confidence"], 0.9)
        self.assertEqual(plan["equipment"], "bearing")
        self.assertEqual(plan["plan"][0]["step_id"], "S1")
        self.assertEqual(plan["plan"][0]["tool"], "search_maintenance_knowledge")
        self.assertEqual(plan["plan"][0]["arguments"], {"query": question, "top_k": 5})
        self.assertIn(
            {"location": "plan", "field": "reasoning"},
            plan["validation"]["removed_fields"],
        )
        self.assertIn(
            {"location": "plan[0]", "field": "description"},
            plan["validation"]["removed_fields"],
        )
        self.assertIn(
            {"step_id": "S1", "field": "source_id"},
            plan["validation"]["removed_arguments"],
        )

    def test_signal_tool_alias_drops_all_model_controlled_arguments(self):
        payload = json.loads(_plan_text())
        payload.update(
            {
                "intent": "bearing_diagnosis",
                "equipment": "podded_propulsor_thrust_bearing",
                "plan": [
                    {
                        "step_id": "S1",
                        "tool": "analyze_bearing",
                        "arguments": {
                            "signal_path": "/private/signal.npy",
                            "threshold": 0.8,
                            "model": "custom",
                        },
                        "depends_on": [],
                    }
                ],
            }
        )
        plan = parse_and_validate_plan(
            json.dumps(payload, ensure_ascii=False),
            has_signal=True,
            user_text="请诊断当前轴承信号。",
        )

        self.assertEqual(plan["intent"], "diagnosis")
        self.assertEqual(plan["equipment"], "bearing")
        self.assertEqual(plan["plan"][0]["tool"], "diagnose_bearing")
        self.assertEqual(plan["plan"][0]["arguments"], {})
        self.assertEqual(
            plan["validation"]["removed_arguments"],
            [
                {"step_id": "S1", "field": "model"},
                {"step_id": "S1", "field": "signal_path"},
                {"step_id": "S1", "field": "threshold"},
            ],
        )

    def test_remembered_signal_cannot_turn_a_knowledge_question_into_diagnosis(self):
        diagnosis_plan = _plan_text(
            intent="diagnosis",
            plan=[
                {
                    "step_id": "S1",
                    "tool": "diagnose_bearing",
                    "arguments": {},
                    "depends_on": [],
                }
            ],
        )
        with self.assertRaisesRegex(
            PlanningValidationError,
            "remembered signal",
        ):
            parse_and_validate_plan(
                diagnosis_plan,
                has_signal=True,
                user_text="外圈故障为什么会产生周期性冲击？",
            )

    def test_knowledge_search_is_anchored_and_narrow_filters_are_removed(self):
        question = "上一轮的置信度为什么不能单独决定是否维修？"
        plan = parse_and_validate_plan(
            _plan_text(),
            max_steps=3,
            user_text=question,
        )
        arguments = plan["plan"][0]["arguments"]
        self.assertTrue(arguments["query"].startswith(question))
        self.assertIn("维修决策", arguments["query"])
        self.assertNotIn("轴承外圈故障", arguments["query"])
        self.assertEqual(arguments["top_k"], 5)
        self.assertNotIn("equipment", arguments)
        self.assertNotIn("fault_type", arguments)
        self.assertTrue(plan["validation"]["knowledge_search_anchored"])

    def test_referential_fault_question_keeps_only_specific_model_context(self):
        question = "上一轮故障类别通常有哪些频谱特征？"
        plan = parse_and_validate_plan(
            _plan_text(),
            max_steps=3,
            user_text=question,
        )
        query = plan["plan"][0]["arguments"]["query"]
        self.assertTrue(query.startswith(question))
        self.assertIn("外圈故障", query)
        self.assertNotIn("周期性冲击", query)

    def test_diagnosis_without_signal_becomes_clarification(self):
        text = _plan_text(
            intent="diagnosis",
            plan=[
                {
                    "step_id": "S1",
                    "tool": "diagnose_bearing",
                    "arguments": {},
                    "depends_on": [],
                }
            ],
        )
        plan = parse_and_validate_plan(text, has_signal=False)
        self.assertEqual(plan["intent"], "clarification")
        self.assertEqual(plan["plan"], [])
        self.assertIn("signal", plan["missing_fields"])
        self.assertIn(".npy", plan["clarification_question"])

    def test_self_contained_knowledge_question_rejects_clarification(self):
        clarification = _plan_text(
            intent="clarification",
            missing_fields=["operating_condition"],
            clarification_question="请补充具体工况。",
            plan=[],
        )
        with self.assertRaisesRegex(
            PlanningValidationError,
            "self-contained maintenance knowledge question",
        ):
            parse_and_validate_plan(
                clarification,
                user_text="管道泄漏声振诊断应关注哪些数据和常见误报来源？",
            )

        executable_request = parse_and_validate_plan(
            clarification,
            user_text="请诊断这个轴承信号。",
        )
        self.assertEqual(executable_request["intent"], "clarification")

    def test_rag_and_standard_question_rejects_unnecessary_clarification(self):
        clarification = _plan_text(
            intent="clarification",
            missing_fields=["model"],
            clarification_question="请补充具体型号。",
            plan=[],
        )
        with self.assertRaisesRegex(
            PlanningValidationError,
            "self-contained maintenance knowledge question",
        ):
            parse_and_validate_plan(
                clarification,
                user_text="RAG 遇到资料库外的设备型号和标准条款时，应如何表达不确定性？",
            )

    def test_current_signal_classification_cannot_be_replaced_by_knowledge_qa(self):
        with self.assertRaisesRegex(
            PlanningValidationError,
            "Explicit current-signal request requires diagnosis",
        ):
            parse_and_validate_plan(
                _plan_text(),
                has_signal=True,
                user_text="先用分类模型分析这段轴承振动，再用知识证据解释结果。",
            )

    def test_model_signal_path_is_removed_from_tool_arguments(self):
        text = _plan_text(
            intent="diagnosis",
            plan=[
                {
                    "step_id": "S1",
                    "tool": "diagnose_bearing",
                    "arguments": {"signal_path": "C:\\private\\signal.npy"},
                    "depends_on": [],
                }
            ],
        )
        plan = parse_and_validate_plan(text, has_signal=True)
        self.assertEqual(plan["plan"][0]["arguments"], {})
        self.assertEqual(
            plan["validation"]["removed_arguments"],
            [{"step_id": "S1", "field": "signal_path"}],
        )

    def test_system_signal_resource_is_removed_from_signal_tool_dependencies(self):
        text = _plan_text(
            intent="diagnosis",
            plan=[
                {
                    "step_id": "S1",
                    "tool": "diagnose_bearing",
                    "arguments": {},
                    "depends_on": ["signal_file"],
                }
            ],
        )
        plan = parse_and_validate_plan(text, has_signal=True)
        self.assertEqual(plan["plan"][0]["depends_on"], [])
        self.assertEqual(
            plan["validation"]["removed_dependencies"],
            [{"step_id": "S1", "dependency": "signal_file"}],
        )

    def test_unknown_non_resource_dependency_is_still_rejected(self):
        text = _plan_text(
            intent="diagnosis",
            plan=[
                {
                    "step_id": "S1",
                    "tool": "diagnose_bearing",
                    "arguments": {},
                    "depends_on": ["untrusted_external_step"],
                }
            ],
        )
        with self.assertRaisesRegex(PlanningValidationError, "unknown steps"):
            parse_and_validate_plan(text, has_signal=True)

    def test_unknown_tool_is_rejected_and_unknown_search_arguments_are_dropped(self):
        unknown_tool = _plan_text(
            plan=[
                {
                    "step_id": "S1",
                    "tool": "delete_signal",
                    "arguments": {},
                    "depends_on": [],
                }
            ]
        )
        with self.assertRaises(PlanningValidationError):
            parse_and_validate_plan(unknown_tool)

        payload = json.loads(_plan_text())
        payload["plan"][0]["arguments"]["server_path"] = "/tmp/private"
        plan = parse_and_validate_plan(json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("server_path", plan["plan"][0]["arguments"])
        self.assertIn(
            {"step_id": "S1", "field": "server_path"},
            plan["validation"]["removed_arguments"],
        )

    def test_step_limit_duplicate_id_and_dependency_cycle_are_rejected(self):
        step = {
            "tool": "search_maintenance_knowledge",
            "arguments": {"query": "轴承", "top_k": 1},
        }
        too_many = _plan_text(
            plan=[
                {**step, "step_id": "S1", "depends_on": []},
                {**step, "step_id": "S2", "depends_on": ["S1"]},
            ]
        )
        with self.assertRaises(PlanningValidationError):
            parse_and_validate_plan(too_many, max_steps=1)

        duplicate = _plan_text(
            plan=[
                {**step, "step_id": "S1", "depends_on": []},
                {**step, "step_id": "S1", "depends_on": []},
            ]
        )
        with self.assertRaises(PlanningValidationError):
            parse_and_validate_plan(duplicate)

        cycle = _plan_text(
            plan=[
                {**step, "step_id": "S1", "depends_on": ["S2"]},
                {**step, "step_id": "S2", "depends_on": ["S1"]},
            ]
        )
        with self.assertRaises(PlanningValidationError):
            parse_and_validate_plan(cycle)

    def test_search_parameters_are_bounded(self):
        payload = json.loads(_plan_text())
        payload["plan"][0]["arguments"]["top_k"] = 6
        plan = parse_and_validate_plan(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(plan["plan"][0]["arguments"]["top_k"], 5)

        payload["plan"][0]["arguments"]["top_k"] = "2"
        plan = parse_and_validate_plan(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(plan["plan"][0]["arguments"]["top_k"], 2)

        payload["plan"][0]["arguments"]["top_k"] = True
        with self.assertRaises(PlanningValidationError):
            parse_and_validate_plan(json.dumps(payload, ensure_ascii=False))

    def test_prompts_include_constraints_and_hide_server_paths(self):
        messages = build_intent_plan_messages(
            "检查上一轮结果",
            memory={
                "signal_file": "safe.npy",
                "signal_path": "C:\\secret\\safe.npy",
                "last_diagnosis": {"fault_type": "外圈故障"},
            },
            has_signal=True,
            max_steps=3,
        )
        combined = "\n".join(str(message.content) for message in messages)
        self.assertIn("plan 最多 3 步", combined)
        self.assertIn("signal_file", combined)
        self.assertIn("不是 step_id", combined)
        self.assertIn("必须以 { 开头", str(messages[-1].content))
        self.assertIn("safe.npy", combined)
        self.assertNotIn("C:\\secret", combined)

        retry = build_intent_plan_retry_messages(
            "检查上一轮结果",
            rejected_output="bad",
            validation_error="Unknown tool",
        )
        self.assertIn("Unknown tool", str(retry[-1].content))

    def test_fallback_preserves_safety_and_missing_signal_boundaries(self):
        safety = fallback_plan("请精确告诉我轴承还能运行多少天")
        self.assertEqual(safety["intent"], "safety_boundary")
        self.assertEqual(safety["plan"], [])

        clarification = fallback_plan("请诊断这个轴承", has_signal=False)
        self.assertEqual(clarification["intent"], "clarification")
        self.assertIn("signal", clarification["missing_fields"])

        diagnosis = fallback_plan("请诊断这个轴承", has_signal=True)
        self.assertEqual(diagnosis["plan"][0]["tool"], "diagnose_bearing")

        knowledge = fallback_plan(
            "管道泄漏声振诊断应关注哪些数据和常见误报来源？",
            has_signal=False,
        )
        self.assertEqual(knowledge["intent"], "knowledge_qa")
        self.assertEqual(
            knowledge["plan"][0]["tool"],
            "search_maintenance_knowledge",
        )

    def test_observation_decision_enforces_permissions_and_remaining_steps(self):
        raw = json.dumps(
            {
                "action": "call_tool",
                "tool": "search_maintenance_knowledge",
                "arguments": {"query": "外圈故障现场复核", "top_k": 3},
                "reason": "补充证据",
                "clarification_question": "",
            },
            ensure_ascii=False,
        )
        decision = parse_observation_decision(
            raw,
            permitted_tools={"search_maintenance_knowledge"},
            remaining_steps=1,
        )
        self.assertEqual(decision["action"], "call_tool")

        with self.assertRaises(PlanningValidationError):
            parse_observation_decision(
                raw,
                permitted_tools={"inspect_signal"},
                remaining_steps=1,
            )
        with self.assertRaises(PlanningValidationError):
            parse_observation_decision(
                raw,
                permitted_tools={"search_maintenance_knowledge"},
                remaining_steps=0,
            )

    def test_observation_signal_tool_without_signal_becomes_clarification(self):
        raw = json.dumps(
            {
                "action": "call_tool",
                "tool": "inspect_signal",
                "arguments": {"signal_path": "/private/file.npy"},
                "reason": "读取摘要",
                "clarification_question": "",
            },
            ensure_ascii=False,
        )
        decision = parse_observation_decision(
            raw,
            permitted_tools={"inspect_signal"},
            remaining_steps=1,
            has_signal=False,
        )
        self.assertEqual(decision["action"], "clarify")
        self.assertIn(".npy", decision["clarification_question"])
        self.assertNotIn("signal_path", decision["arguments"])

    def test_successful_observation_cannot_be_hidden_by_clarification(self):
        raw = json.dumps(
            {
                "action": "clarify",
                "tool": None,
                "arguments": {},
                "reason": "还需要更多工况",
                "clarification_question": "请补充转速和负荷。",
            },
            ensure_ascii=False,
        )
        with self.assertRaisesRegex(
            PlanningValidationError,
            "cannot replace an available successful tool observation",
        ):
            parse_observation_decision(
                raw,
                permitted_tools={"search_maintenance_knowledge"},
                remaining_steps=2,
                has_usable_observation=True,
            )

        decision = parse_observation_decision(
            raw,
            permitted_tools=set(),
            remaining_steps=2,
            has_usable_observation=False,
        )
        self.assertEqual(decision["action"], "clarify")

    def test_observation_prompt_redacts_structured_paths(self):
        messages = build_observation_messages(
            "下一步做什么？",
            current_plan={"intent": "diagnosis"},
            observations=[
                {
                    "status": "ok",
                    "signal_path": "C:\\secret\\file.npy",
                    "signal_file": "file.npy",
                }
            ],
            permitted_tools={"search_maintenance_knowledge"},
            remaining_steps=1,
        )
        combined = "\n".join(str(message.content) for message in messages)
        self.assertNotIn("C:\\secret", combined)
        self.assertIn("file.npy", combined)


if __name__ == "__main__":
    unittest.main()
