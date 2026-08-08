import unittest
from types import SimpleNamespace

from equipdoc_agent.rag.retriever import KnowledgeRetriever, _source_priority


class RetrieverSourcePriorityTests(unittest.TestCase):
    def test_reads_and_clamps_source_priority(self):
        self.assertEqual(_source_priority({"metadata": {"source_priority": 100}}), 100.0)
        self.assertEqual(_source_priority({"metadata": {"source_priority": 130}}), 100.0)
        self.assertEqual(_source_priority({"metadata": {"source_priority": -5}}), 0.0)
        self.assertEqual(_source_priority({"metadata": {"source_priority": "70"}}), 70.0)
        self.assertEqual(_source_priority({"metadata": {"source_priority": "unknown"}}), 0.0)

    def test_top_k_reserves_eighty_percent_for_distinct_documents(self):
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.settings = SimpleNamespace(rag_top_k=5)
        lexical = [
            {"chunk_id": "a1", "doc_id": "a", "metadata": {}},
            {"chunk_id": "a2", "doc_id": "a", "metadata": {}},
            {"chunk_id": "b1", "doc_id": "b", "metadata": {}},
            {"chunk_id": "c1", "doc_id": "c", "metadata": {}},
            {"chunk_id": "d1", "doc_id": "d", "metadata": {}},
            {"chunk_id": "e1", "doc_id": "e", "metadata": {}},
        ]
        retriever._lexical_search = lambda query, filters, limit: lexical
        retriever._dense_search = lambda query, filters, limit: []

        hits = retriever.search("轴承", top_k=5)

        self.assertEqual([item["doc_id"] for item in hits[:4]], ["a", "b", "c", "d"])
        self.assertEqual(hits[4]["chunk_id"], "a2")


if __name__ == "__main__":
    unittest.main()
