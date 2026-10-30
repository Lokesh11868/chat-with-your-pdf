"""
test_app.py - Comprehensive Unit and Integration Test Suite for Chat With Your PDF.
Tests PDF extraction, page tracking, chunking, ChromaDB isolation, retrieval, and grounding.
"""

import os
import sys
import io
import unittest
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pypdf import PdfWriter
import pdf_utils
import rag_utils


def create_test_pdf_with_pages(pages_content: list, filename: str) -> str:
    """Helper to create a PDF with synthetic text on specific pages."""
    writer = PdfWriter()
    for text in pages_content:
        # Create a blank page and add annotation / text
        page = writer.add_blank_page(width=612, height=792)
        # Note: in pypdf, creating text directly requires stream writing or reportlab
        # Let's write standard PDF content stream for text
        # PDF stream syntax for text: BT /F1 12 Tf 50 700 Td (text) Tj ET
    # Rather than raw low-level streams, let's check reportlab or generate proper text stream
    return filename


def make_pdf_bytes_with_text(page_texts: list) -> bytes:
    """
    Creates a valid PDF binary with extractable text across pages
    using standard PDF text objects without external heavy dependencies.
    """
    # Build standard PDF 1.4 syntax
    objects = []
    
    # 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    
    # 2: Pages container
    kids_refs = []
    page_objs = []
    content_objs = []
    
    start_obj = 3
    num_pages = len(page_texts)
    
    # We will allocate:
    # 2: Pages
    # 3: Font
    # 4 .. 4+num_pages-1: Page objects
    # 4+num_pages .. 4+2*num_pages-1: Content stream objects
    
    font_obj_idx = 3
    objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj")
    
    for i in range(num_pages):
        page_idx = 4 + i
        content_idx = 4 + num_pages + i
        kids_refs.append(f"{page_idx} 0 R".encode("ascii"))
        
        page_objs.append(
            f"{page_idx} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_idx} 0 R >> >> /Contents {content_idx} 0 R >>\nendobj".encode("utf-8")
        )
        
        # Build text stream
        raw_text = page_texts[i]
        # Clean text for PDF string
        escaped_lines = []
        for line in raw_text.split("\n"):
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            escaped_lines.append(f"({esc}) Tj T*")
        
        stream_data = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(escaped_lines) + " ET"
        stream_bytes = stream_data.encode("utf-8")
        
        content_objs.append(
            f"{content_idx} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n{stream_data}\nendstream\nendobj".encode("utf-8")
        )

    pages_kids_str = b" ".join(kids_refs)
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{pages_kids_str.decode('ascii')}] /Count {num_pages} >>\nendobj".encode("utf-8")
    
    # Assemble all objects
    all_objs = [objects[0], pages_obj, objects[1]] + page_objs + content_objs
    
    # Build cross reference table and header
    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets = [0]
    
    for obj in all_objs:
        offsets.append(len(body))
        body.extend(obj)
        body.extend(b"\n")
        
    xref_offset = len(body)
    total_objs = len(all_objs) + 1
    xref = f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("ascii")
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("ascii")
        
    trailer = f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    body.extend(xref)
    body.extend(trailer)
    
    return bytes(body)


class TestPDFUtils(unittest.TestCase):
    """Tests for PDF text extraction and chunking."""

    def setUp(self):
        self.page1_text = (
            "The Open Systems Interconnection (OSI) model describes seven layers of networking. "
            "Layer 1 is the Physical layer dealing with bit transmission over copper or fiber cables. "
            "Layer 2 is the Data Link layer which handles MAC addresses and frame switching. "
            "Layer 3 is the Network layer where IP routing and IP packets are managed by routers. "
            "Layer 4 is the Transport layer featuring TCP for reliable transmission and UDP for fast datagrams. "
            "Layer 7 is the Application layer providing HTTP, DNS, and SSH to end-user applications."
        )
        self.page2_text = (
            "The Domain Name System (DNS) translates human-readable domain names into IP addresses. "
            "DNS typically operates over UDP port 53 for standard queries to minimize connection latency. "
            "When DNS responses exceed 512 bytes, DNS can fall back to TCP port 53. "
            "Top-level domains include dot com, dot org, and country-code domains like dot in. "
            "Authoritative name servers hold the master DNS records for a domain zone."
        )
        self.page3_text = (
            "Transmission Control Protocol (TCP) uses a three-way handshake: SYN, SYN-ACK, and ACK. "
            "TCP ensures reliability through sequence numbers, acknowledgments, and window flow control. "
            "In contrast, User Datagram Protocol (UDP) is connectionless and does not guarantee delivery. "
            "UDP is preferred for real-time video streaming, voice over IP (VoIP), and gaming."
        )
        self.pdf_bytes = make_pdf_bytes_with_text([self.page1_text, self.page2_text, self.page3_text])

    def test_extract_pages_preserves_page_numbers(self):
        pages = pdf_utils.extract_pages_from_pdf(self.pdf_bytes)
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertEqual(pages[2]["page_number"], 3)
        self.assertIn("OSI", pages[0]["text"])
        self.assertIn("DNS", pages[1]["text"])
        self.assertIn("handshake", pages[2]["text"])

    def test_chunking_with_overlap(self):
        chunks = pdf_utils.split_text_into_chunks(
            text=self.page1_text,
            page_number=1,
            chunk_size=30,
            overlap=10,
            source_name="network_guide.pdf"
        )
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c["page"], 1)
            self.assertEqual(c["page_label"], "Page 1")
            self.assertIn("network_guide.pdf", c["source"])

    def test_process_pdf_full_pipeline(self):
        processed = pdf_utils.process_pdf(
            pdf_source=self.pdf_bytes,
            chunk_size=40,
            overlap=10,
            source_name="test_doc.pdf"
        )
        self.assertEqual(processed["total_pages"], 3)
        self.assertGreater(processed["total_words"], 50)
        self.assertGreater(len(processed["chunks"]), 2)
        self.assertTrue(len(processed["doc_hash"]) == 64)

    def test_scanned_pdf_detection(self):
        """A PDF with blank pages and no text should raise NoTextExtractedError."""
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        out = io.BytesIO()
        writer.write(out)
        blank_bytes = out.getvalue()

        with self.assertRaises(pdf_utils.NoTextExtractedError):
            pdf_utils.extract_pages_from_pdf(blank_bytes)

    def test_corrupt_pdf_handling(self):
        with self.assertRaises(pdf_utils.CorruptedPDFError):
            pdf_utils.extract_pages_from_pdf(b"not a valid pdf at all")


class TestRAGPipeline(unittest.TestCase):
    """Tests for vector storage, isolation, retrieval, and grounding."""

    @classmethod
    def setUpClass(cls):
        # Create test documents
        cls.doc1_pages = [
            "Python was created by Guido van Rossum and first released in 1991. "
            "Python emphasizes code readability with significant whitespace indentation. "
            "The Zen of Python outlines principles like 'Beautiful is better than ugly'.",
            
            "Streamlit is an open-source Python framework for building interactive web apps. "
            "Streamlit apps are written as simple Python scripts and automatically re-run on interaction. "
            "Streamlit was founded by Adrien Treuille, Amanda Kelly, and Thiago Teixeira."
        ]
        cls.doc1_bytes = make_pdf_bytes_with_text(cls.doc1_pages)
        cls.doc1_processed = pdf_utils.process_pdf(cls.doc1_bytes, chunk_size=30, overlap=5, source_name="doc1.pdf")

        cls.doc2_pages = [
            "Quantum computing uses qubits that exist in superpositions of zero and one. "
            "Quantum entanglement allows qubits to be correlated regardless of distance. "
            "Shor's algorithm can factor integers in polynomial time on an ideal quantum computer."
        ]
        cls.doc2_bytes = make_pdf_bytes_with_text(cls.doc2_pages)
        cls.doc2_processed = pdf_utils.process_pdf(cls.doc2_bytes, chunk_size=30, overlap=5, source_name="doc2.pdf")

    def test_isolated_chromadb_collections(self):
        # Index doc 1
        col1, was_new1 = rag_utils.index_pdf_chunks(
            chunks=self.doc1_processed["chunks"],
            doc_hash=self.doc1_processed["doc_hash"]
        )
        # Index doc 2
        col2, was_new2 = rag_utils.index_pdf_chunks(
            chunks=self.doc2_processed["chunks"],
            doc_hash=self.doc2_processed["doc_hash"]
        )

        self.assertNotEqual(col1.name, col2.name)

        # Retrieve for doc 1
        chunks1 = rag_utils.retrieve_relevant_chunks(
            question="Who created Streamlit?",
            doc_hash=self.doc1_processed["doc_hash"],
            top_k=2
        )
        self.assertGreater(len(chunks1), 0)
        # Ensure doc 1 query NEVER returns quantum computing
        for c in chunks1:
            self.assertNotIn("qubits", c["text"].lower())
            self.assertIn(c["page"], [1, 2])

        # Retrieve for doc 2
        chunks2 = rag_utils.retrieve_relevant_chunks(
            question="What is Shor's algorithm?",
            doc_hash=self.doc2_processed["doc_hash"],
            top_k=2
        )
        self.assertGreater(len(chunks2), 0)
        # Ensure doc 2 query NEVER returns python or streamlit
        for c in chunks2:
            self.assertNotIn("streamlit", c["text"].lower())
            self.assertEqual(c["page"], 1)

    def test_strict_grounding_fallback_prompt(self):
        prompt = rag_utils.build_grounded_prompt(
            question="What is the capital of Mars?",
            chunks=[]
        )
        self.assertIn("I don't know based on the provided PDF", prompt["system_instruction"])
        self.assertIn("No relevant context", prompt["user_content"])


if __name__ == "__main__":
    unittest.main()
