#!/usr/bin/env python3
"""
pdf_to_pages.py

Extracts text from the first PDF in the `books/` directory and writes each page to a separate .txt file
inside a directory named after the PDF (without extension). Default behavior:

- Looks in './books' for the first PDF.
- Creates './book_txt/<book_name>/' and writes '1.txt', '2.txt', ... for each page.

Usage:
    python pdf_to_pages.py                  # find first pdf in 'books/' and process it
    python pdf_to_pages.py --pdf path/to/book.pdf --output ./books

This script uses PyMuPDF (pymupdf) for good text extraction. It will fall back to PyPDF2 if PyMuPDF is unavailable.

"""

import argparse
import os
import sys
import pathlib
import re
from typing import Optional


def _sanitize_name(name: str) -> str:
    # Replace forbidden path chars and trim
    return re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()


def find_first_pdf(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    for entry in os.listdir(directory):
        if entry.lower().endswith(".pdf"):
            return os.path.join(directory, entry)
    return None


def extract_with_pymupdf(pdf_path: str):
    try:
        import fitz  # pymupdf
    except Exception:
        raise

    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
    doc.close()
    return pages_text


def extract_with_pypdf2(pdf_path: str):
    # PyPDF2 fallback
    try:
        import PyPDF2
    except Exception:
        raise

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages_text = []
        for p in reader.pages:
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            pages_text.append(text)
    return pages_text


def extract_text_per_page(pdf_path: str):
    # Try pymupdf first
    try:
        return extract_with_pymupdf(pdf_path)
    except Exception:
        try:
            return extract_with_pypdf2(pdf_path)
        except Exception as e:
            raise RuntimeError("Neither PyMuPDF nor PyPDF2 extraction succeeded. Install one of them. Error: " + str(e))


def write_pages_out(pages_text, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, text in enumerate(pages_text, start=1):
        filename = os.path.join(out_dir, f"{i}.txt")
        # Use UTF-8 and newline normalization
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            f.write(text or "")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract PDF pages to individual text files.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to a single PDF file to process.")
    parser.add_argument("--input-dir", type=str, default="books", help="Directory to search for PDFs when --pdf isn't provided (default: books)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write output. Defaults to 'book_txt' folder.")

    args = parser.parse_args(argv)

    pdf_path = args.pdf
    if pdf_path is None:
        pdf_path = find_first_pdf(args.input_dir)
        if pdf_path is None:
            print(f"No PDF found in '{args.input_dir}'", file=sys.stderr)
            sys.exit(2)

    if not os.path.isfile(pdf_path):
        print(f"PDF file '{pdf_path}' not found.", file=sys.stderr)
        sys.exit(2)

    # Use book_txt folder for output if not specified
    output_root = args.output_dir or "book_txt"

    base_name = pathlib.Path(pdf_path).stem
    safe_name = _sanitize_name(base_name)
    output_dir = os.path.join(output_root, safe_name)

    print(f"Processing: {pdf_path}")
    print(f"Output: {output_dir}")

    try:
        pages_text = extract_text_per_page(pdf_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    write_pages_out(pages_text, output_dir)
    print(f"Extracted {len(pages_text)} pages.")


if __name__ == "__main__":
    main()
