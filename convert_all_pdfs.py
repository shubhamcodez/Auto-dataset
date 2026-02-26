#!/usr/bin/env python3
"""
convert_all_pdfs.py

Converts all PDF files in the books/ directory to page-wise text files
in the book_txt/ directory using pdf_to_pages.py functionality.
"""

import os
import sys
import pathlib
import re
from typing import List

# Import functions from pdf_to_pages
from pdf_to_pages import extract_text_per_page, write_pages_out, _sanitize_name


def find_all_pdfs(directory: str) -> List[str]:
    """Find all PDF files in the given directory."""
    if not os.path.isdir(directory):
        return []
    pdfs = []
    for entry in os.listdir(directory):
        if entry.lower().endswith(".pdf"):
            pdfs.append(os.path.join(directory, entry))
    return pdfs


def convert_pdf_to_pages(pdf_path: str, output_root: str = "book_txt"):
    """Convert a single PDF to page-wise text files."""
    base_name = pathlib.Path(pdf_path).stem
    safe_name = _sanitize_name(base_name)
    output_dir = os.path.join(output_root, safe_name)
    
    print(f"Processing: {pdf_path}")
    print(f"Output: {output_dir}")
    
    try:
        pages_text = extract_text_per_page(pdf_path)
        write_pages_out(pages_text, output_dir)
        print(f"Extracted {len(pages_text)} pages.\n")
        return len(pages_text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}", file=sys.stderr)
        return 0


def main():
    books_dir = "books"
    output_dir = "book_txt"
    
    if not os.path.isdir(books_dir):
        print(f"Directory '{books_dir}' not found.", file=sys.stderr)
        sys.exit(1)
    
    pdfs = find_all_pdfs(books_dir)
    
    if not pdfs:
        print(f"No PDF files found in '{books_dir}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(pdfs)} PDF file(s) to process.\n")
    
    total_pages = 0
    for pdf_path in pdfs:
        pages = convert_pdf_to_pages(pdf_path, output_dir)
        total_pages += pages
    
    print(f"\nCompleted! Processed {len(pdfs)} PDF(s) with {total_pages} total pages.")


if __name__ == "__main__":
    main()

