import unittest

from equipdoc_agent.retrieval_display import (
    render_retrieval_hits_markdown,
    sanitize_retrieval_hits,
)


class RetrievalDisplayTests(unittest.TestCase):
    def test_snapshot_is_limited_and_drops_private_or_non_json_fields(self):
        hits = [
            {
                "doc_id": f"doc_{index}",
                "chunk_id": f"chunk_{index}",
                "title": "测试文档",
                "heading_path": "第一章 > 试验",
                "text": "召回原文",
                "rank_score": 0.02,
                "dense_score": 0.8,
                "metadata": {
                    "source_priority": 100,
                    "source_path": r"C:\private\source.docx",
                },
                "embedding": object(),
            }
            for index in range(7)
        ]

        snapshot = sanitize_retrieval_hits(hits)

        self.assertEqual(len(snapshot), 5)
        self.assertEqual(snapshot[0]["source_priority"], 100.0)
        self.assertNotIn("metadata", snapshot[0])
        self.assertNotIn("source_path", snapshot[0])
        self.assertNotIn("embedding", snapshot[0])

    def test_markdown_contains_ids_scores_and_escaped_chunk_text(self):
        markdown = render_retrieval_hits_markdown(
            [
                {
                    "doc_id": "plan_ch2",
                    "chunk_id": "plan_ch2_c009",
                    "title": "试验台设计",
                    "heading_path": "第二章 > 转速范围",
                    "text": "转速范围为0～2000 r/min。<script>alert(1)</script>",
                    "rrf_score": 0.03,
                    "dense_score": 0.82,
                    "source_priority": 100,
                }
            ]
        )

        self.assertIn("plan_ch2", markdown)
        self.assertIn("plan_ch2_c009", markdown)
        self.assertIn("向量 0.8200", markdown)
        self.assertIn("来源优先级 100", markdown)
        self.assertIn("&lt;script&gt;", markdown)
        self.assertNotIn("<script>", markdown)

    def test_empty_snapshot_has_explicit_message(self):
        self.assertIn("没有可展示", render_retrieval_hits_markdown([]))


if __name__ == "__main__":
    unittest.main()
