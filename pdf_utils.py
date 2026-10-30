"""
pdf_utils.py - PDF Text Extraction and Overlapping Chunking Module.

Handles PDF parsing, page-level text extraction with page number preservation,
usable text validation, and overlapping word chunking for the RAG pipeline.
"""

import io
import hashlib
from typing import List, Dict, Any, Union, BinaryIO
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass


class EncryptedPDFError(PDFProcessingError):
    """Raised when a PDF is password protected and cannot be read."""
    pass


class CorruptedPDFError(PDFProcessingError):
    """Raised when a PDF file is corrupted or not a valid PDF."""
    pass


class NoTextExtractedError(PDFProcessingError):
    """Raised when a PDF contains no extractable text layer (e.g. scanned images)."""
    pass


def get_pdf_hash(file_bytes: bytes) -> str:
    """Generates a SHA-256 hash string for the given PDF bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def extract_pages_from_pdf(
    pdf_source: Union[bytes, BinaryIO, io.BytesIO, str]
) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF while preserving page numbers.

    Args:
        pdf_source: PDF content as bytes, BytesIO stream, file-like object, or file path.

    Returns:
        List of dictionaries with page metadata:
        [
            {
                "page_number": 1,
                "text": "...",
                "word_count": 120,
                "char_count": 780
            },
            ...
        ]

    Raises:
        EncryptedPDFError: If the document is password-protected.
        CorruptedPDFError: If the file is invalid or cannot be parsed.
        NoTextExtractedError: If no usable text is extracted from any page.
    """
    try:
        if isinstance(pdf_source, str):
            stream = open(pdf_source, "rb")
        elif isinstance(pdf_source, bytes):
            stream = io.BytesIO(pdf_source)
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            content = pdf_source.read()
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            stream = io.BytesIO(content)

        reader = PdfReader(stream)

        if reader.is_encrypted:
            # Attempt empty password decryption if possible
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    raise EncryptedPDFError(
                        "The uploaded PDF is password-protected. "
                        "Please provide an unprotected PDF file."
                    )
            except Exception:
                raise EncryptedPDFError(
                    "The uploaded PDF is password-protected. "
                    "Please provide an unprotected PDF file."
                )

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise CorruptedPDFError("The uploaded PDF has 0 pages.")

        pages_data = []
        total_extracted_words = 0

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1  # 1-indexed for human readability
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                page_text = ""

            cleaned_text = page_text.strip()
            words = cleaned_text.split()
            word_count = len(words)
            char_count = len(cleaned_text)
            total_extracted_words += word_count

            pages_data.append({
                "page_number": page_num,
                "text": cleaned_text,
                "word_count": word_count,
                "char_count": char_count
            })

        if total_extracted_words == 0:
            raise NoTextExtractedError(
                "No extractable text was found in this PDF. "
                "The file appears to be a scanned document or contains only images. "
                "Note: Optical Character Recognition (OCR) is not supported."
            )

        return pages_data

    except (EncryptedPDFError, NoTextExtractedError):
        raise
    except PdfReadError as e:
        raise CorruptedPDFError(f"Failed to parse PDF: {str(e)}")
    except Exception as e:
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            raise EncryptedPDFError("The uploaded PDF is password-protected.")
        raise CorruptedPDFError(f"Error processing PDF file: {str(e)}")


def split_text_into_chunks(
    text: str,
    page_number: int,
    chunk_size: int = 250,
    overlap: int = 50,
    source_name: str = "document.pdf"
) -> List[Dict[str, Any]]:
    """
    Splits text from a specific page into overlapping word chunks.
    Preserves page number and metadata for source attribution.

    Args:
        text: Text string to split.
        page_number: 1-indexed page number.
        chunk_size: Target words per chunk (default: 250).
        overlap: Overlap in words between consecutive chunks (default: 50).
        source_name: Name of the source PDF.

    Returns:
        List of chunk dictionaries with text, page number, and metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        # Merge tiny remainder chunk (< 15 words or < half overlap) with preceding chunk
        if len(chunk_words) < max(15, overlap // 2) and chunks:
            chunks[-1]["text"] += " " + chunk_text
            chunks[-1]["word_count"] = len(chunks[-1]["text"].split())
            chunks[-1]["char_count"] = len(chunks[-1]["text"])
            chunks[-1]["end_word_idx"] = end
            break

        c_idx = len(chunks)
        chunk_id = f"p{page_number}_c{c_idx:02d}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "page": page_number,
            "page_label": f"Page {page_number}",
            "chunk_index_in_page": c_idx,
            "word_count": len(chunk_words),
            "char_count": len(chunk_text),
            "start_word_idx": start,
            "end_word_idx": end,
            "source": source_name
        })

        if end >= total_words:
            break

        start += step

    return chunks


def chunk_extracted_pages(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 250,
    overlap: int = 50,
    source_name: str = "document.pdf"
) -> List[Dict[str, Any]]:
    """
    Chunks all pages in the document while preserving page attribution.

    Args:
        pages_data: Output of extract_pages_from_pdf.
        chunk_size: Words per chunk (default: 250).
        overlap: Overlap in words (default: 50).
        source_name: Name of the source PDF.

    Returns:
        List of all chunks across all pages with global and page-level metadata.
    """
    all_chunks = []
    global_idx = 0

    for p in pages_data:
        page_num = p["page_number"]
        page_text = p["text"]
        if not page_text:
            continue

        page_chunks = split_text_into_chunks(
            text=page_text,
            page_number=page_num,
            chunk_size=chunk_size,
            overlap=overlap,
            source_name=source_name
        )

        for c in page_chunks:
            c["global_chunk_index"] = global_idx
            # Make chunk id globally unique within this document
            c["id"] = f"{source_name}_p{page_num}_c{c['chunk_index_in_page']:02d}"
            all_chunks.append(c)
            global_idx += 1

    return all_chunks


def process_pdf(
    pdf_source: Union[bytes, BinaryIO, io.BytesIO, str],
    chunk_size: int = 250,
    overlap: int = 50,
    source_name: str = "document.pdf"
) -> Dict[str, Any]:
    """
    Full pipeline to extract and chunk a PDF document.

    Returns:
        Dictionary with:
        - "source_name": filename
        - "doc_hash": sha256 hash of PDF bytes
        - "total_pages": int
        - "total_words": int
        - "chunks": List[Dict]
        - "pages": List[Dict]
    """
    if isinstance(pdf_source, str):
        with open(pdf_source, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(pdf_source, bytes):
        raw_bytes = pdf_source
    else:
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
        raw_bytes = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)

    doc_hash = get_pdf_hash(raw_bytes)
    pages = extract_pages_from_pdf(raw_bytes)
    total_words = sum(p["word_count"] for p in pages)
    chunks = chunk_extracted_pages(
        pages_data=pages,
        chunk_size=chunk_size,
        overlap=overlap,
        source_name=source_name
    )

    return {
        "source_name": source_name,
        "doc_hash": doc_hash,
        "total_pages": len(pages),
        "total_words": total_words,
        "chunks": chunks,
        "pages": pages
    }
