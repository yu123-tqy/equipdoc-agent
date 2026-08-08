# Third-party assets and publication boundary

- The Qwen2.5 base model and any derived LoRA/merged weights are not included.
- CWRU bearing data is not committed in `data/raw/`; the legacy download script records the source URLs.
- `data/samples/test_signal.npy` is a small demonstration sample derived from the original experiment package.
- `data/knowledge/` contains self-written notes, reviewed domain summaries, and sanitized technical extracts from user-provided project documents. The original Word files, signatures, contact details, account information, and other non-technical contract content are not included.
- Images in `data/knowledge_assets/` are retained only when needed to preserve the technical structure or meaning of the project extracts. Their publication and downstream reuse remain subject to the rights of the original document owner.
- Domain summaries cite their source families in `docs/rag_knowledge_source_catalog.md`; they are not substitutes for current manufacturer instructions, standards, or site procedures.
- The repository must not contain employer/customer code, internal documents, real equipment identifiers, or confidential operating data.
- Before changing this repository to an OSI open-source license, confirm the publication rights for every code, document, image, and data asset.
