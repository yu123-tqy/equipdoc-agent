from __future__ import annotations

import unittest

from equipdoc_agent.agent.knowledge_answer import (
    build_grounded_synthesis_messages,
    build_grounded_synthesis_retry_messages,
    render_tool_observation_section,
    should_retry_grounded_synthesis,
    validate_grounded_draft,
)


class GroundedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            {
                "evidence_id": "E01",
                "citation": "bearing_outer_race_fault#bearing_outer_race_fault_c001",
                "text": "轴承外圈局部缺陷会产生周期性冲击，包络谱可关注 BPFO。",
            },
            {
                "evidence_id": "E02",
                "citation": "bearing_review#bearing_review_c001",
                "text": "现场应复核润滑状态、安装松动、温度和异常噪声。",
            },
        ]

    def test_grounded_paraphrase_with_sentence_citations_is_valid(self):
        draft = (
            "## 直接回答\n\n"
            "轴承外圈存在局部缺陷时会形成周期性冲击 "
            "[bearing_outer_race_fault#bearing_outer_race_fault_c001]\n\n"
            "## 现场复核\n\n"
            "现场应检查润滑状态与异常噪声 "
            "[bearing_review#bearing_review_c001]"
        )
        validation = validate_grounded_draft(draft, self.evidence)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["claim_citation_coverage"], 1.0)

    def test_unknown_or_trailing_paragraph_citation_is_invalid(self):
        unknown = "外圈缺陷会产生冲击 [unknown#unknown_c001]"
        self.assertFalse(validate_grounded_draft(unknown, self.evidence)["valid"])

        paragraph = (
            "外圈缺陷会产生周期性冲击。还需要检查润滑状态 [bearing_review#bearing_review_c001]"
        )
        validation = validate_grounded_draft(paragraph, self.evidence)
        self.assertFalse(validation["valid"])
        self.assertEqual(len(validation["uncited_claims"]), 1)

    def test_unknown_acronym_and_number_are_rejected(self):
        draft = "应关注 BPFI 和 99Hz [bearing_outer_race_fault#bearing_outer_race_fault_c001]"
        validation = validate_grounded_draft(draft, self.evidence)
        terms = {item["term"] for item in validation["unsupported_terms"]}
        self.assertIn("BPFI", terms)
        self.assertIn("99", terms)
        self.assertFalse(validation["valid"])

    def test_forbidden_control_claim_is_rejected(self):
        draft = "系统已执行停机并检查润滑状态 [bearing_review#bearing_review_c001]"
        validation = validate_grounded_draft(draft, self.evidence)
        self.assertIn("已执行停机", validation["forbidden_terms"])
        self.assertFalse(validation["valid"])

    def test_superficial_token_overlap_cannot_support_a_new_causal_claim(self):
        draft = (
            "轴承外圈缺陷会导致润滑油变成蓝色 "
            "[bearing_outer_race_fault#bearing_outer_race_fault_c001]"
        )
        validation = validate_grounded_draft(draft, self.evidence)
        self.assertFalse(validation["valid"])
        self.assertEqual(len(validation["unsupported_claims"]), 1)

    def test_question_slot_coverage_is_checked(self):
        draft = "外圈缺陷会产生周期性冲击 [bearing_outer_race_fault#bearing_outer_race_fault_c001]"
        validation = validate_grounded_draft(
            draft,
            self.evidence,
            question="为什么会产生冲击，现场应复核什么？",
        )
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["missing_slots"], ["field_review"])

    def test_negated_maintenance_boundary_cannot_replace_treatment_advice(self):
        evidence = [
            {
                "evidence_id": "E01",
                "citation": "ball#cause",
                "text": "滚动体故障的常见原因包括润滑不足、污染和冲击过载。",
            },
            {
                "evidence_id": "E02",
                "citation": "boundary#no_signal",
                "text": "资料不足时不得声称已经诊断出滚动体故障，不得编造维修工单。",
            },
            {
                "evidence_id": "E03",
                "citation": "ball#treatment",
                "text": "建议检查润滑污染、载荷冲击和安装状态。",
            },
        ]
        rejected = (
            "滚动体故障的常见原因包括润滑不足、污染和冲击过载 "
            "[ball#cause]\n\n"
            "资料不足时不得编造维修工单 [boundary#no_signal]"
        )
        validation = validate_grounded_draft(
            rejected,
            evidence,
            question="滚动体发生故障的原因是什么，应该怎么处理",
        )

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["missing_slots"], ["maintenance"])

        accepted = (
            "滚动体故障的常见原因包括润滑不足、污染和冲击过载 "
            "[ball#cause]\n\n"
            "建议检查润滑污染、载荷冲击和安装状态 [ball#treatment]"
        )
        self.assertTrue(
            validate_grounded_draft(
                accepted,
                evidence,
                question="滚动体发生故障的原因是什么，应该怎么处理",
            )["valid"]
        )

    def test_synthesis_prompts_include_only_selected_evidence_and_redact_paths(self):
        messages = build_grounded_synthesis_messages(
            "为什么是外圈故障？",
            self.evidence,
            [
                {
                    "_tool_name": "diagnose_bearing",
                    "status": "ok",
                    "signal_path": "C:\\private\\signal.npy",
                    "signal_file": "signal.npy",
                    "fault_type": "外圈故障",
                }
            ],
        )
        combined = "\n".join(str(message.content) for message in messages)
        self.assertIn("bearing_outer_race_fault#bearing_outer_race_fault_c001", combined)
        self.assertIn("signal.npy", combined)
        self.assertNotIn("C:\\private", combined)
        self.assertIn("每一个技术陈述句", combined)

        validation = validate_grounded_draft("没有引用", self.evidence)
        retry = build_grounded_synthesis_retry_messages(
            "为什么？",
            self.evidence,
            [],
            "没有引用",
            validation,
        )
        self.assertIn("uncited_claims", str(retry[-1].content))

    def test_retry_policy_repairs_semantic_and_citation_failures_once(self):
        unsupported = validate_grounded_draft(
            "外圈故障每转只冲击一次 [bearing_outer_race_fault#bearing_outer_race_fault_c001]",
            self.evidence,
        )
        self.assertTrue(should_retry_grounded_synthesis(unsupported))

        zero_citation = validate_grounded_draft("第一版没有引用", self.evidence)
        self.assertTrue(should_retry_grounded_synthesis(zero_citation))

        partially_cited = validate_grounded_draft(
            "轴承外圈局部缺陷会产生周期性冲击 "
            "[bearing_outer_race_fault#bearing_outer_race_fault_c001]\n"
            "现场还需要继续检查。",
            self.evidence,
        )
        self.assertTrue(should_retry_grounded_synthesis(partially_cited))

    def test_tool_observation_is_rendered_deterministically(self):
        rendered = render_tool_observation_section(
            [
                {
                    "_tool_name": "diagnose_bearing",
                    "status": "ok",
                    "signal_file": "signal.npy",
                    "fault_type": "外圈故障",
                    "confidence": 0.62,
                    "signal": {"samples": 1024, "rms": 1.25},
                    "warning": "需要现场复核",
                }
            ]
        )
        self.assertIn("signal.npy", rendered)
        self.assertIn("62.00%", rendered)
        self.assertIn("RMS=1.25", rendered)
        self.assertIn("需要现场复核", rendered)


if __name__ == "__main__":
    unittest.main()
