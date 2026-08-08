import unittest
from dataclasses import replace
from pathlib import Path

from equipdoc_agent.config import Settings
from equipdoc_agent.rag import KnowledgeRetriever


ROOT = Path(__file__).resolve().parents[1]


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = replace(
            Settings.from_env(ROOT),
            rag_db_dir=(ROOT / "runtime/test_no_vector_db").resolve(),
        )
        cls.retriever = KnowledgeRetriever(settings)

    def assert_doc_recalled(self, query, expected_doc_id, top_k=5):
        doc_ids = [item["doc_id"] for item in self.retriever.search(query, top_k=top_k)]
        self.assertIn(expected_doc_id, doc_ids)

    def test_retrieves_gearbox_wear_notes(self):
        self.assert_doc_recalled("齿轮箱齿面磨损有哪些信号特征", "pump_gearbox_faults")

    def test_retrieves_pump_cavitation_notes(self):
        self.assert_doc_recalled("泵发生汽蚀有什么表现", "pump_gearbox_faults")

    def test_retrieves_outer_race_notes(self):
        self.assert_doc_recalled("轴承外圈故障的振动和包络谱特征", "bearing_outer_race_fault")

    def test_speed_parameter_question_promotes_the_numeric_requirement(self):
        hits = self.retriever.search(
            "吊舱推进器推力轴承故障诊断试验台的设计转速范围是多少？请给出文档依据。",
            top_k=5,
        )
        self.assertEqual(hits[0]["chunk_id"], "pod_thrust_bearing_plan_ch2_c008")
        self.assertIn("0–2000 rpm", hits[0]["text"])
        self.assertEqual(hits[0]["source_priority"], 100.0)

    def test_bearing_model_question_promotes_the_configuration_table(self):
        hits = self.retriever.search(
            "试验台使用的轴承型号是什么？请给出文档依据。",
            top_k=5,
        )
        self.assertEqual(hits[0]["chunk_id"], "test_rig_technical_spec_c003")
        self.assertIn("29412", hits[0]["text"])
        self.assertIn("NU 212EM", hits[0]["text"])


if __name__ == "__main__":
    unittest.main()
