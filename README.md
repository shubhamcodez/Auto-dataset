# PDF to Pages

A small Python utility to extract a PDF into per-page `.txt` files.

Default behavior:
- Finds the first `.pdf` inside the `books/` directory
- Creates `books/<book-name>/1.txt`, `2.txt`, ... where `<book-name>` is the PDF filename without the extension

Usage:

```bash
# Process the first PDF found inside 'books/' and write text files into the book folder
python pdf_to_pages.py

# Process a specific PDF and place output in a different directory
python pdf_to_pages.py --pdf books/my_book.pdf --output-dir output

# Specify another input directory to search
python pdf_to_pages.py --input-dir ./book_txt --output-dir ./out
```

Dependencies
- `pymupdf` (PyMuPDF) for best extraction
- Falls back to `PyPDF2` if pymupdf is not installed

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Notes:
- This script focuses on text-based PDFs; scanned-image PDFs may require OCR (not included here).
- The script writes UTF-8 text and normalizes newlines to `\n`.
