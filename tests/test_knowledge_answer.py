import unittest

from equipdoc_agent.agent.knowledge_answer import (
    build_evidence_candidates,
    build_citation_retry_messages,
    build_full_rag_messages,
    detect_question_requirements,
    extract_citations,
    extract_evidence_selection,
    render_extractive_fallback,
    render_selected_evidence,
    render_structured_evidence_answer,
    render_retrieval_context,
    select_evidence_for_question,
    validate_answer_citations,
    validate_evidence_selection,
)


class KnowledgeAnswerTests(unittest.TestCase):
    def setUp(self):
        self.hits = [
            {
                "doc_id": "bearing_outer_race_fault",
                "chunk_id": "bearing_outer_race_fault_c001",
                "text": "外圈缺陷会产生周期性冲击。",
            }
        ]

    def test_context_contains_machine_checkable_citation(self):
        context = render_retrieval_context(self.hits)
        self.assertIn(
            "[bearing_outer_race_fault#bearing_outer_race_fault_c001]",
            context,
        )

    def test_full_prompt_requires_evidence_id_selection(self):
        messages = build_full_rag_messages("外圈故障有什么特征？", self.hits)
        combined = "\n".join(str(message.content) for message in messages)
        self.assertIn("EVIDENCE_IDS", combined)
        self.assertIn("[E01]", combined)

    def test_extracts_full_citations_and_ignores_numeric_markers(self):
        citations = extract_citations(
            "外圈故障会产生冲击 [bearing_outer_race_fault#bearing_outer_race_fault_c001] [1]"
        )
        self.assertEqual(
            citations,
            [("bearing_outer_race_fault", "bearing_outer_race_fault_c001")],
        )

    def test_retry_prompt_lists_only_allowed_evidence_ids(self):
        messages = build_citation_retry_messages("问题", self.hits, "没有ID的输出")
        combined = "\n".join(str(message.content) for message in messages)
        self.assertIn("唯一允许选择的证据句ID", combined)
        self.assertIn("E01", combined)

    def test_extracts_and_validates_evidence_selection(self):
        candidates = build_evidence_candidates(self.hits)
        selected = extract_evidence_selection("EVIDENCE_IDS: E01,E02,E01")
        self.assertEqual(selected, ["E01", "E02"])
        validation = validate_evidence_selection(selected, candidates)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["unknown_ids"], ["E02"])

    def test_selection_requires_slots_but_not_an_exact_top_four_set(self):
        candidates = [
            {
                "evidence_id": f"E{index:02d}",
                "citation": f"doc#chunk_{index}",
                "text": text,
                "focused_match": False,
            }
            for index, text in enumerate(
                (
                    "RMS 反映整体能量。",
                    "峭度对冲击较敏感。",
                    "一般轴承知识。",
                    "通用现场建议。",
                    "外圈故障说明。",
                    "内圈故障说明。",
                    "润滑状态说明。",
                    "温度检查说明。",
                ),
                start=1,
            )
        ]
        validation = validate_evidence_selection(
            ["E05", "E06", "E07", "E08"],
            candidates,
            question="RMS 和峭度分别反映什么？",
        )
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["minimum_relevance_matches"], 0)
        self.assertEqual(validation["relevance_matches"], [])
        self.assertEqual(validation["missing_slots"], ["signal_feature"])

        partial = validate_evidence_selection(
            ["E01", "E02", "E05", "E06"],
            candidates,
            question="RMS 和峭度分别反映什么？",
        )
        self.assertTrue(partial["valid"])
        self.assertEqual(partial["relevance_matches"], ["E01", "E02"])

    def test_deterministic_selection_covers_question_slots(self):
        candidates = [
            {
                "evidence_id": "E01",
                "citation": "doc#mechanism",
                "text": "外圈缺陷会产生周期性冲击。",
                "focused_match": True,
            },
            {
                "evidence_id": "E02",
                "citation": "doc#review",
                "text": "现场应复核安装松动、润滑状态和异常噪声。",
                "focused_match": False,
            },
            {
                "evidence_id": "E03",
                "citation": "doc#other",
                "text": "这是无关的通用描述。",
                "focused_match": True,
            },
        ]
        selection = select_evidence_for_question(
            "外圈故障为什么产生冲击，现场应检查什么？",
            candidates,
        )
        self.assertTrue(selection["valid"])
        self.assertEqual(selection["missing_slots"], [])
        self.assertIn("E01", selection["selected_ids"])
        self.assertIn("E02", selection["selected_ids"])

        answer = render_structured_evidence_answer(
            "外圈故障为什么产生冲击，现场应检查什么？",
            candidates,
            selection["selected_ids"],
        )
        self.assertIn("## 直接回答", answer)
        self.assertIn("外圈缺陷会产生周期性冲击", answer)
        self.assertIn("现场应复核", answer)
        self.assertIn("## 补充依据", answer)

    def test_ball_fault_question_answers_cause_and_treatment_not_data_boundary(self):
        question = "滚动体发生故障的原因是什么，应该怎么处理"
        candidates = [
            {
                "evidence_id": "E01",
                "citation": "no_signal#boundary",
                "text": "资料不足时不得声称已经诊断出滚动体故障，不得编造维修工单。",
                "focused_match": True,
            },
            {
                "evidence_id": "E02",
                "citation": "ball#cause",
                "text": "滚动体故障的常见原因包括润滑不足、异物污染、冲击过载和安装偏斜。",
                "focused_match": True,
            },
            {
                "evidence_id": "E03",
                "citation": "ball#treatment",
                "text": "建议检查润滑污染、轴承温度、载荷冲击和安装状态。",
                "focused_match": True,
            },
            {
                "evidence_id": "E04",
                "citation": "ball#signal",
                "text": "滚动体故障可能伴随 BSF 附近能量升高。",
                "focused_match": True,
            },
        ]

        self.assertEqual(
            detect_question_requirements(question),
            ["mechanism", "maintenance"],
        )
        selection = select_evidence_for_question(question, candidates)
        self.assertTrue(selection["valid"])
        self.assertEqual(selection["slot_assignments"]["mechanism"], "E02")
        self.assertEqual(selection["slot_assignments"]["maintenance"], "E03")

        answer = render_structured_evidence_answer(
            question,
            candidates,
            selection["selected_ids"],
            selection["slot_assignments"],
        )
        direct = answer.split("## 补充依据", maxsplit=1)[0]
        self.assertIn("常见原因包括", direct)
        self.assertIn("建议检查", direct)
        self.assertNotIn("资料不足", direct)

    def test_parameter_fallback_answers_speed_before_listing_contract_evidence(self):
        candidates = [
            {
                "evidence_id": "E01",
                "citation": "plan#speed",
                "text": "驱动系统可实现0–2000 rpm范围内的稳定运行。",
                "focused_match": True,
                "source_priority": 100,
            },
            {
                "evidence_id": "E02",
                "citation": "contract#speed",
                "text": "转速可调范围不低于0～1400 r/min。",
                "focused_match": True,
                "source_priority": 70,
            },
        ]
        question = "试验台的设计转速范围是多少？"
        selection = select_evidence_for_question(question, candidates, limit=2)
        answer = render_structured_evidence_answer(
            question,
            candidates,
            selection["selected_ids"],
            selection["slot_assignments"],
        )

        direct, supplemental = answer.split("## 补充依据", maxsplit=1)
        self.assertIn("0–2000 rpm", direct)
        self.assertNotIn("0～1400 r/min", direct)
        self.assertIn("0～1400 r/min", supplemental)

    def test_parameter_fallback_formats_both_bearing_models_as_sentences(self):
        candidates = [
            {
                "evidence_id": "E01",
                "citation": "contract#models",
                "text": "| 1 | 被测推力轴承 | 29412（球面滚子推力轴承） |",
                "focused_match": True,
                "source_priority": 70,
            },
            {
                "evidence_id": "E02",
                "citation": "contract#models",
                "text": "| 2 | 被测支撑轴承 | NU 212EM（单列圆柱滚子轴承） |",
                "focused_match": True,
                "source_priority": 70,
            },
        ]
        question = "试验台使用的轴承型号是什么？"
        selection = select_evidence_for_question(question, candidates, limit=2)
        answer = render_structured_evidence_answer(
            question,
            candidates,
            selection["selected_ids"],
            selection["slot_assignments"],
        )

        self.assertIn("被测推力轴承型号为 29412", answer)
        self.assertIn("被测支撑轴承型号为 NU 212EM", answer)

    def test_review_ranking_treats_inspection_as_field_review_evidence(self):
        candidates = [
            {
                "evidence_id": f"E{index:02d}",
                "citation": f"doc#chunk_{index}",
                "text": text,
                "focused_match": index <= 5,
            }
            for index, text in enumerate(
                (
                    "对于高风险介质，应优先按安全流程隔离和复核。",
                    "管道泄漏可能引起局部振动。",
                    "环境噪声会影响信号特征。",
                    "不能只依赖单一算法判断。",
                    "应结合压力、流量、声发射和人工巡检。",
                    "滚动体故障不能只看单一峰值。",
                ),
                start=1,
            )
        ]
        validation = validate_evidence_selection(
            [],
            candidates,
            question="怀疑高风险介质管道泄漏时应如何复核？",
        )
        self.assertIn("E05", validation["recommended_ids"])
        self.assertNotIn("E06", validation["recommended_ids"])

    def test_focus_specific_evidence_beats_generic_multi_slot_sentences(self):
        candidates = [
            {
                "evidence_id": "E01",
                "citation": "generic#c001",
                "text": "工程上需要结合包络谱、工况和传感器进行现场复核。",
                "focused_match": True,
            },
            {
                "evidence_id": "E02",
                "citation": "outer#c002",
                "text": "外圈故障在包络谱中 BPFO 附近峰值更明显。",
                "focused_match": False,
            },
            {
                "evidence_id": "E03",
                "citation": "outer#c003",
                "text": "外圈故障应复核转速、负载和传感器安装状态，再现场检查。",
                "focused_match": False,
            },
            {
                "evidence_id": "E04",
                "citation": "inner#c002",
                "text": "内圈故障应结合转速和历史趋势判断。",
                "focused_match": True,
            },
        ]
        question = "轴承外圈点蚀时，包络谱有什么表现，现场应怎样复核？"
        selection = select_evidence_for_question(question, candidates)

        self.assertEqual(selection["slot_assignments"]["signal_feature"], "E02")
        self.assertEqual(selection["slot_assignments"]["field_review"], "E03")

        answer = render_structured_evidence_answer(
            question,
            candidates,
            selection["selected_ids"],
            selection["slot_assignments"],
        )
        direct_section, supplemental_section = answer.split("## 补充依据", maxsplit=1)
        self.assertIn("外圈故障在包络谱中 BPFO", direct_section)
        self.assertIn("外圈故障应复核转速", direct_section)
        self.assertNotIn("这是无关", direct_section)
        self.assertIn("工程上需要结合包络谱", supplemental_section)

    def test_causal_connector_is_classified_as_mechanism(self):
        candidates = [
            {
                "evidence_id": "E01",
                "citation": "inner#c001",
                "text": "内圈随轴旋转，因此信号常带有转频调制特征。",
                "focused_match": True,
            },
            {
                "evidence_id": "E02",
                "citation": "inner#c002",
                "text": "内圈故障在包络谱中 BPFI 两侧出现转频边带。",
                "focused_match": True,
            },
            {
                "evidence_id": "E03",
                "citation": "inner#c003",
                "text": "内圈故障复核时应检查负载、转速和传感器状态。",
                "focused_match": True,
            },
        ]
        selection = select_evidence_for_question(
            "内圈故障为什么有转频边带，复核时看什么？",
            candidates,
        )
        self.assertEqual(selection["slot_assignments"]["mechanism"], "E01")
        self.assertTrue(selection["valid"])

    def test_one_chunk_can_supply_three_complementary_answer_facts(self):
        candidates = [
            {
                "evidence_id": "E01",
                "citation": "maintenance#c001",
                "text": ("模型可以给出置信度，但是否维修需要结合负载、温度、历史趋势和生产风险。"),
                "focused_match": True,
            },
            {
                "evidence_id": "E02",
                "citation": "maintenance#c001",
                "text": "若置信度中等，建议复测多段信号后再决定是否拆检。",
                "focused_match": True,
            },
            {
                "evidence_id": "E03",
                "citation": "maintenance#c001",
                "text": "诊断报告应区分模型结论和维修决策。",
                "focused_match": True,
            },
            {
                "evidence_id": "E04",
                "citation": "boundary#c001",
                "text": "不能仅凭一次结果推断精确剩余寿命。",
                "focused_match": False,
            },
        ]
        selection = select_evidence_for_question(
            "置信度为什么不能单独决定是否维修？",
            candidates,
        )
        self.assertIn("E03", selection["selected_ids"])

        answer = render_structured_evidence_answer(
            "置信度为什么不能单独决定是否维修？",
            candidates,
            selection["selected_ids"],
            selection["slot_assignments"],
        )
        self.assertIn("维修决策", answer)

    def test_selected_evidence_is_rendered_with_source_citation(self):
        candidates = build_evidence_candidates(self.hits)
        answer = render_selected_evidence(candidates, ["E01"])
        self.assertTrue(validate_answer_citations(answer, self.hits)["valid"])
        self.assertIn("[bearing_outer_race_fault#bearing_outer_race_fault_c001]", answer)

    def test_invalid_answer_falls_back_to_exact_cited_evidence(self):
        validation = validate_answer_citations("没有引用", self.hits)
        self.assertFalse(validation["valid"])
        fallback = render_extractive_fallback(self.hits)
        self.assertTrue(validate_answer_citations(fallback, self.hits)["valid"])
        self.assertIn("系统已隐藏未验证输出", fallback)

    def test_one_trailing_citation_cannot_cover_multiple_sentences(self):
        answer = (
            "外圈缺陷会产生周期性冲击。"
            "每转只重复一次 [bearing_outer_race_fault#bearing_outer_race_fault_c001]"
        )
        validation = validate_answer_citations(answer, self.hits)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["claim_count"], 2)
        self.assertEqual(validation["cited_claim_count"], 1)
        self.assertEqual(validation["claim_citation_coverage"], 0.5)

    def test_each_claim_with_allowed_citation_is_valid(self):
        answer = (
            "- 外圈缺陷会产生周期性冲击 "
            "[bearing_outer_race_fault#bearing_outer_race_fault_c001]\n"
            "- 外圈缺陷会产生周期性冲击 "
            "[bearing_outer_race_fault#bearing_outer_race_fault_c001]"
        )
        validation = validate_answer_citations(answer, self.hits)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["claim_citation_coverage"], 1.0)
        self.assertEqual(validation["claim_evidence_match_rate"], 1.0)

    def test_cited_but_unsupported_paraphrase_is_invalid(self):
        answer = "外圈缺陷每转只冲击一次 [bearing_outer_race_fault#bearing_outer_race_fault_c001]"
        validation = validate_answer_citations(answer, self.hits)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["claim_citation_coverage"], 1.0)
        self.assertEqual(validation["claim_evidence_match_rate"], 0.0)
        self.assertEqual(len(validation["unsupported_claims"]), 1)


if __name__ == "__main__":
    unittest.main()
