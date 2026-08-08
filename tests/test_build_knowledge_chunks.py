from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_knowledge_chunks.py"
SPEC = importlib.util.spec_from_file_location("build_knowledge_chunks", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KnowledgeChunkBuilderTests(unittest.TestCase):
    def test_long_section_is_split_with_length_bound_and_priority(self):
        runtime = ROOT / "runtime"
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as temp_dir:
            knowledge_dir = Path(temp_dir) / "knowledge"
            knowledge_dir.mkdir()
            document = knowledge_dir / "summary.md"
            document.write_text(
                "---\n"
                "doc_id: summary_doc\n"
                "title: 轴承诊断摘要\n"
                "source_type: authoritative_summary\n"
                "---\n"
                "# 轴承诊断摘要\n\n"
                "## 失效分析\n\n"
                + "。".join([f"第{i}条诊断事实需要结合载荷转速润滑和振动趋势复核" for i in range(80)])
                + "。\n",
                encoding="utf-8",
            )

            chunks = MODULE.build_chunks(
                knowledge_dir,
                target_chars=420,
                max_chars=500,
                overlap_chars=80,
            )

            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(len(item["text"]) <= 500 for item in chunks))
            self.assertTrue(all(item["metadata"]["source_priority"] == 60 for item in chunks))
            self.assertTrue(all("轴承诊断摘要" in item["text"] for item in chunks))

    def test_exact_override_keeps_page_anchor_and_plan_priority(self):
        runtime = ROOT / "runtime"
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as temp_dir:
            temp_path = Path(temp_dir)
            knowledge_dir = temp_path / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "plan.md").write_text(
                "---\n"
                "doc_id: plan_doc\n"
                "title: 实验方案\n"
                "source_type: project_document\n"
                "source_authority: experiment_plan_primary\n"
                "---\n"
                "# 实验方案\n\n## 第一章\n\n原始内容。\n",
                encoding="utf-8",
            )
            override_path = temp_path / "overrides.jsonl"
            override = {
                "doc_id": "plan_doc",
                "chunk_id": "plan_doc#0001",
                "text": "实验方案 > 第一章\n原始技术段落。",
                "metadata": {"page_start": 7, "page_end": 7, "block_start": 21, "block_end": 22},
            }
            override_path.write_text(json.dumps(override, ensure_ascii=False) + "\n", encoding="utf-8")

            chunks = MODULE.build_chunks(knowledge_dir, chunk_overrides_path=override_path)

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["text"], override["text"])
            self.assertEqual(chunks[0]["metadata"]["page_start"], 7)
            self.assertEqual(chunks[0]["metadata"]["source_priority"], 100)
            self.assertTrue(chunks[0]["metadata"]["source_path"].endswith("plan.md"))


if __name__ == "__main__":
    unittest.main()
