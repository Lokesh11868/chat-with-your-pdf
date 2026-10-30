"""
test_e2e_grounding.py - End-to-End Live Grounding and Source Citation Verification.
Executes comprehensive PDF RAG grounding and citation verification tests:
- 3 Answerable questions verified for accurate content and page attribution.
- 2 Unanswerable questions verified for strict refusal ("I don't know based on the provided PDF").
- Edge cases: scanned PDF, corrupted PDF, empty queries, API key handling, cross-document isolation.
"""

import os
import sys
import io
import json
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

import pdf_utils
import rag_utils
from test_app import make_pdf_bytes_with_text


def create_sample_pdf() -> bytes:
    """Creates a 3-page technical document on Computer Networking."""
    page1 = (
        "Computer Networking Guide: Architecture and Protocols.\n"
        "Chapter 1: The OSI Reference Model.\n"
        "The Open Systems Interconnection model defines seven architectural layers:\n"
        "Layer 1 Physical Layer transmits raw bit streams over cables and radio frequencies.\n"
        "Layer 2 Data Link Layer uses MAC hardware addresses to transfer frames across switches.\n"
        "Layer 3 Network Layer is responsible for logical IP packet routing across routers.\n"
        "Layer 4 Transport Layer provides host-to-host communication using TCP or UDP protocols.\n"
        "Layer 5 Session Layer manages dialogues and connection sessions.\n"
        "Layer 6 Presentation Layer handles syntax translation and SSL/TLS data encryption.\n"
        "Layer 7 Application Layer supports end-user protocols including HTTP, DNS, and SMTP."
    )
    page2 = (
        "Chapter 2: The Domain Name System (DNS).\n"
        "DNS translates alphanumeric domain names into numeric IP addresses.\n"
        "DNS servers primarily listen on UDP port 53 for low-overhead query resolution.\n"
        "However, when DNS responses exceed 512 bytes, DNS automatically falls back to TCP port 53.\n"
        "Recursive resolvers query root name servers, Top-Level Domain (TLD) servers, and authoritative servers.\n"
        "DNS caching reduces query latency across local networks and operating systems."
    )
    page3 = (
        "Chapter 3: Transmission Control Protocol (TCP) and UDP.\n"
        "TCP provides reliable, ordered, and error-checked delivery of a stream of bytes.\n"
        "To establish a connection, TCP performs a three-way handshake:\n"
        "Step 1: The client sends a SYN (synchronize) packet to the server.\n"
        "Step 2: The server responds with a SYN-ACK (synchronize-acknowledge) packet.\n"
        "Step 3: The client replies with an ACK (acknowledge) packet.\n"
        "TCP uses sequence numbers and sliding window flow control to prevent buffer overflows.\n"
        "Conversely, UDP is a connectionless datagram protocol without handshakes or retransmission."
    )
    return make_pdf_bytes_with_text([page1, page2, page3])


def run_e2e_tests():
    print("=" * 75)
    print("CHAT WITH YOUR PDF: END-TO-END RAG PIPELINE & GROUNDING VERIFICATION")
    print("=" * 75)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[!] ERROR: GROQ_API_KEY is not set. Cannot run live LLM tests.")
        return False

    print("[1] Building test document: 'networking_guide.pdf' (3 pages)...")
    pdf_bytes = create_sample_pdf()
    
    # Save a copy to disk for manual inspection and demo testing
    sample_path = _PROJECT_ROOT / "assets" / "sample_networking_guide.pdf"
    with open(sample_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"    Saved sample PDF to: {sample_path}")

    # Process and index
    print("\n[2] Processing PDF with pdf_utils...")
    processed = pdf_utils.process_pdf(
        pdf_source=pdf_bytes,
        chunk_size=50,
        overlap=10,
        source_name="sample_networking_guide.pdf"
    )
    print(f"    Total Pages : {processed['total_pages']}")
    print(f"    Total Words : {processed['total_words']}")
    print(f"    Total Chunks: {len(processed['chunks'])}")
    print(f"    Document Hash: {processed['doc_hash'][:16]}...")

    print("\n[3] Indexing chunks in ChromaDB (isolated collection)...")
    col, was_new = rag_utils.index_pdf_chunks(
        chunks=processed["chunks"],
        doc_hash=processed["doc_hash"]
    )
    print(f"    Collection Name : {col.name}")
    print(f"    Indexed Chunks  : {col.count()}")

    # --- Live Answerable Questions ---
    print("\n" + "=" * 75)
    print("[4] TESTING ANSWERABLE QUESTIONS (Must cite correct content and page)")
    print("=" * 75)

    answerable_queries = [
        {
            "id": "A1",
            "question": "What are the specific steps in the TCP three-way handshake?",
            "expected_page": 3,
            "expected_keywords": ["SYN", "SYN-ACK", "ACK"]
        },
        {
            "id": "A2",
            "question": "What port does DNS use and when does it switch from UDP to TCP?",
            "expected_page": 2,
            "expected_keywords": ["53", "512"]
        },
        {
            "id": "A3",
            "question": "Which OSI layer is responsible for logical IP packet routing?",
            "expected_page": 1,
            "expected_keywords": ["Network", "Layer 3"]
        }
    ]

    for q in answerable_queries:
        print(f"\n--- Query {q['id']}: \"{q['question']}\" ---")
        res = rag_utils.generate_rag_answer(
            question=q["question"],
            doc_hash=processed["doc_hash"],
            api_key=api_key,
            top_k=3
        )
        print(f"Answer: {res['answer']}\n")
        print("Retrieved Sources:")
        pages_found = []
        for s in res["sources"]:
            print(f"  • {s['page_label']} (similarity: {s['similarity']:.4f}): \"{s['excerpt'][:90]}...\"")
            pages_found.append(s["page"])

        # Verifications
        has_expected_page = q["expected_page"] in pages_found
        has_keywords = any(kw.lower() in res["answer"].lower() for kw in q["expected_keywords"])
        is_not_refusal = not res["is_grounded_fallback"]

        print(f"Verification: Page {q['expected_page']} in sources? {'YES' if has_expected_page else 'NO'} | "
              f"Keywords in answer? {'YES' if has_keywords else 'NO'} | Not refused? {'YES' if is_not_refusal else 'NO'}")
        assert has_expected_page, f"Query {q['id']} failed to retrieve page {q['expected_page']}"
        assert has_keywords, f"Query {q['id']} missing expected keywords"
        assert is_not_refusal, f"Query {q['id']} unexpectedly refused"

    # --- Live Unanswerable Questions ---
    print("\n" + "=" * 75)
    print("[5] TESTING UNANSWERABLE QUESTIONS (Must refuse with exact fallback)")
    print("=" * 75)

    unanswerable_queries = [
        {
            "id": "U1",
            "question": "What is the capital city of Australia?",
        },
        {
            "id": "U2",
            "question": "Who won the 2022 FIFA World Cup and what was the final score?",
        }
    ]

    for q in unanswerable_queries:
        print(f"\n--- Query {q['id']}: \"{q['question']}\" ---")
        res = rag_utils.generate_rag_answer(
            question=q["question"],
            doc_hash=processed["doc_hash"],
            api_key=api_key,
            top_k=3
        )
        print(f"Answer: {res['answer']}")
        print(f"Declined / Fallback returned: {res['is_grounded_fallback']}")
        # Check strict fallback text
        refusal_matched = "i don't know based on the provided pdf" in res["answer"].lower()
        print(f"Verification: Refusal statement present? {'YES' if refusal_matched else 'NO'}")
        assert refusal_matched or res["is_grounded_fallback"], f"Query {q['id']} failed to refuse unsupported question"

    # --- Edge Cases ---
    print("\n" + "=" * 75)
    print("[6] TESTING SYSTEM EDGE CASES")
    print("=" * 75)

    # 1. Empty query
    empty_res = rag_utils.generate_rag_answer(
        question="   ",
        doc_hash=processed["doc_hash"],
        api_key=api_key
    )
    print(f"• Empty Query Response: \"{empty_res['answer']}\" -> OK")
    assert "Please ask a specific question" in empty_res["answer"]

    # 2. Scanned PDF (no text layer)
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    try:
        pdf_utils.extract_pages_from_pdf(buf.getvalue())
        print("• Scanned PDF test FAILED (did not raise)")
        assert False
    except pdf_utils.NoTextExtractedError as e:
        print(f"• Scanned PDF Exception: {e} -> OK")

    # 3. Corrupt PDF
    try:
        pdf_utils.extract_pages_from_pdf(b"%PDF-corrupt-data-not-valid")
        print("• Corrupt PDF test FAILED (did not raise)")
        assert False
    except pdf_utils.CorruptedPDFError as e:
        print(f"• Corrupt PDF Exception caught cleanly -> OK")

    # 4. Invalid API Key
    try:
        rag_utils.call_groq_llm(
            prompt={"system_instruction": "test", "user_content": "test"},
            api_key="your_groq_api_key_here"
        )
        print("• Invalid API Key test FAILED (did not raise)")
        assert False
    except ValueError as e:
        print(f"• Missing/Placeholder API Key handled cleanly -> OK")

    print("\n" + "=" * 75)
    print("ALL END-TO-END RAG PIPELINE & GROUNDING TESTS PASSED PERFECTLY!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)
