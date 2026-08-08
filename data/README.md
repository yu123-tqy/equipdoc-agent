# Data layout

- `samples/`: small files required for the safe Demo mode.
- `knowledge/`: Markdown knowledge sources. The expanded corpus includes self-written notes, reviewed bearing-domain summaries, and sanitized project-document extracts.
- `knowledge_assets/`: figures extracted from the project documents. Markdown uses relative links; image pixels are not embedded by the current text-only pipeline.
- `knowledge_chunk_overrides.jsonl`: exact, page-anchored chunks for the experiment plan and contract technical specification.
- `knowledge_source_anchors.jsonl`: page/block-to-section traceability records for manual review; not inserted into Chroma as standalone documents.
- `knowledge_chunks.jsonl`: deterministic, pre-built chunks used by lexical retrieval and by the Chroma index builder.
- `eval/`: Agent and RAG evaluation inputs, including `rag_project_eval.jsonl` for the newly added project knowledge.
- `raw/` and `processed/`: generated locally and ignored by Git.

## Source precedence

For project-specific questions, the experiment plan is primary (`source_priority=100`) and the contract is supporting (`source_priority=70`). The contract may supplement dimensions, quantities, electrical requirements, safety provisions, and acceptance criteria. If statements conflict, the experiment plan governs and the answer should expose the source boundary.

## Chunk policy

After editing `data/knowledge/*.md`, rebuild and verify the committed chunks:

```bash
python scripts/build_knowledge_chunks.py
python scripts/build_knowledge_chunks.py --check
```

General Markdown is split by heading and sentence with a 420-character target, 500-character hard maximum, and 80-character overlap. Project-document overrides preserve the reviewed 206 exact chunks and their page/block anchors while enforcing the same 500-character maximum.

Do not add employer data, real customer/equipment identifiers, contact details, signatures, internal manuals, or confidential operating parameters.
