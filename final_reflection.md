# Engineering Reflection: Building 'Chat With Your PDF' RAG System

**Project:** Chat With Your PDF — Production Document Q&A  
**Architecture:** Grounded Retrieval-Augmented Generation (RAG)  
**Technology Stack:** Python 3.10+, Streamlit, Groq API (LPU Inference), sentence-transformers (`all-MiniLM-L6-v2`), ChromaDB (HNSW), pypdf  

---

## 1. LLM API Integration & Prompt Engineering Principles

Building interactive LLM-driven applications requires treating language models not as speculative black boxes, but as deterministic reasoning engines operating under strict constraints:

- **System Prompts & Personas:** The `system` role sets the foundational guardrails, persona, and behavioral boundaries of the model. By adjusting the system instruction, we transform a conversationalist into a concise, laser-focused technical document assistant that adheres strictly to provided facts.
- **Temperature & Determinism:** Controlling temperature balances creativity and consistency. For tasks requiring strict factual fidelity (like document Q&A), low temperatures (0.0 to 0.2) drastically reduce generative randomness and speculative token paths.
- **Conversation Memory Management:** Maintaining multi-turn conversation state requires explicitly tracking the message history array and interleaving `user` and `assistant` turns. Defensive programming is vital: if an API call fails mid-turn, orphaned user messages must be cleaned up to avoid corrupting subsequent API request structures.

---

## 2. Dense Vector Embeddings and Semantic Search

Moving beyond string matching requires understanding how high-dimensional semantic spaces function:

- **Mathematical Meaning:** Traditional keyword matching relies on exact string overlap and fails when users ask questions with synonyms or varied phrasings (e.g., "halt my plan" vs. "pause subscription"). Text embeddings transform text into dense 384-dimensional vectors using `all-MiniLM-L6-v2`, mapping semantic concepts into a shared geometric coordinate space.
- **Cosine Similarity:** By calculating the dot product of normalized vectors, semantic search measures the cosine of the angle between query and document vectors. A smaller angle corresponds to higher conceptual similarity, regardless of shared vocabulary.
- **Scalability via Dedicated Vector Indices:** While in-memory brute-force linear search ($O(N)$) works for tiny lists, real-world applications with multi-page documents require specialized vector index structures like Hierarchical Navigable Small World (HNSW) to avoid linear latency scaling and memory bottlenecks.

---

## 3. Vector Storage & RAG Architecture

Combining retrieval and generation into a modular Retrieval-Augmented Generation (RAG) pipeline delivers grounded, authoritative document question-answering:

- **Chunking with Overlap:** Feeding an entire multi-thousand-word document into an LLM wastes context tokens and degrades retrieval precision. By slicing text into overlapping word windows (250 words per chunk with 50 words of overlap), each chunk maintains coherent context without splitting critical thoughts across arbitrary boundaries.
- **Vector Storage in ChromaDB:** ChromaDB provides a persistent, disk-backed vector store utilizing HNSW indexing, allowing sub-millisecond approximate nearest neighbor (ANN) retrieval.
- **Grounded Generation:** Grounding fundamentally mitigates hallucinations. Instead of allowing the model to answer from general training data, we inject the top-$k$ retrieved chunks directly into the prompt and constrain the LLM with a strict fallback instruction: *"If the context does not contain sufficient information to answer the question, say: I don't know based on the provided PDF."*

---

## 4. Full-Stack Application Orchestration

Shipping a complete RAG product requires solving systemic architecture and user experience challenges:

- **End-to-End Orchestration:** Orchestrating text extraction, chunking, embedding generation, vector indexing, retrieval, and LLM inference inside an event-driven framework.
- **Decoupled Architecture:** Keeping `app.py` purely focused on UI and session state while isolating extraction in `pdf_utils.py` and vector retrieval in `rag_utils.py` makes the codebase maintainable, testable, and production-ready.
- **Streamlit Reactive Lifecycle:** Streamlit re-executes the entire script on every user interaction. Without careful session management via `st.session_state` and caching, models would reload and documents would re-index on every keystroke, rendering the app unusable.

---

## 5. Technical Challenges Faced and Resolved

1. **Page-Level Source Attribution:**
   - *Problem:* Standard chunkers split an entire document string continuously, losing page boundaries. This prevents accurate page citations.
   - *Solution:* In `pdf_utils.py`, page-aware extraction extracts text per page first, tracks 1-indexed page numbers, and performs overlapping sliding-window chunking within each page boundary. This guarantees that every retrieved chunk has an exact, verifiable page number.

2. **Per-Document Session & Vector Store Isolation:**
   - *Problem:* If ChromaDB uses a single shared collection, uploading a new PDF or asking questions in a new session could retrieve chunks from previously uploaded documents.
   - *Solution:* Implemented SHA-256 document hashing (`doc_hash`). Each uploaded PDF is indexed into its own isolated collection namespace (`pdf_col_<hash[:32]>`). When switching documents or clicking "New Chat / Upload New PDF", the active namespace resets, eliminating cross-document contamination.

3. **Dynamic Groq Model Availability & Fallback:**
   - *Problem:* Remote LLM providers periodically update, rotate, or rate-limit model endpoints. Relying on a single hardcoded model string can cause runtime failures.
   - *Solution:* Built dynamic model discovery via `client.models.list()` paired with a multi-model fallback cascade in `rag_utils.py`. The app inspects available models, displays compatible options in the UI, and automatically tries secondary models if the primary model is temporarily unavailable.

4. **Handling Scanned PDFs and Corrupted Files:**
   - *Problem:* Users frequently upload image-only PDFs (scanned documents) or password-protected files.
   - *Solution:* Created explicit custom exceptions (`NoTextExtractedError`, `EncryptedPDFError`, `CorruptedPDFError`). The system inspects total extracted word count before attempting chunking, clearly explaining that OCR is not supported rather than silently failing or indexing empty collections.

---

## 6. Why Grounding and Source References Matter

In enterprise AI and high-stakes fields (legal, healthcare, finance, engineering), an ungrounded LLM is a liability. A confident hallucination is far more dangerous than an admission of ignorance.

1. **Auditability & Verifiability:** Providing page numbers and exact verbatim excerpts allows human reviewers to fact-check answers in seconds. If an assistant cites "Page 3", the user can cross-reference the physical page immediately.
2. **User Trust:** When users see the exact excerpts alongside similarity scores, they understand *how* the model reached its conclusion.
3. **Prompt Injection Defense:** Strict grounding prompts explicitly instruct the model to treat document text as data rather than instructions, mitigating indirect prompt injection attacks embedded inside uploaded PDFs.

---

## 7. Future Improvements for Production

To scale this application further for high-load enterprise environments:

1. **Hybrid Search (BM25 + Dense Vectors):** Dense embeddings excel at high-level semantic meaning but struggle with exact alphanumeric strings (like part numbers, error codes, or RFC numbers). Combining BM25 sparse keyword search with dense vectors via Reciprocal Rank Fusion (RRF) would maximize retrieval accuracy.
2. **Local / Cloud OCR Integration:** Integrate Tesseract or AWS Textract to extract text from scanned and image-heavy documents, removing the text-only limitation.
3. **Streaming Responses:** Use `st.write_stream` with Groq's streaming API to deliver tokens in real time, drastically improving the perceived time-to-first-token (TTFT).
4. **Multi-Document Comparison Mode:** Allow users to upload multiple PDFs simultaneously and ask comparative questions (e.g., comparing contractual differences between two legal agreements).

---

## Conclusion

This project represents an end-to-end understanding of embeddings, vector databases, LLM alignment, grounding constraints, and user-centered interface design. By coupling dense local embeddings with isolated persistent indexing and strict context-bound generation, **Chat With Your PDF** delivers dependable, verifiable document intelligence.
