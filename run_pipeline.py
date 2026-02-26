#!/usr/bin/env python3
"""
run_pipeline.py

Master script that:
1. Converts all PDFs in books/ to page-wise text files in book_txt/
2. Processes 3-page windows and generates datasets
"""

import sys
import subprocess
import os


def check_openai_api():
    """Check if OpenAI API key is configured."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", "")
    # Also check if it's set in generate_datasets.py
    if not api_key:
        try:
            from generate_datasets import OPENAI_API_KEY
            api_key = OPENAI_API_KEY
        except:
            pass
    
    if api_key:
        print("✓ OpenAI API key is configured")
        return True
    else:
        print("⚠ Warning: OpenAI API key not detected.")
        print("  Set OPENAI_API_KEY or OPENAI_KEY environment variable or configure in generate_datasets.py")
        print("  Get your API key from: https://platform.openai.com/api-keys")
        return False


def main():
    print("=" * 60)
    print("PDF to Dataset Pipeline")
    print("=" * 60)
    
    # Step 1: Convert PDFs
    print("\n[Step 1/2] Converting PDFs to page-wise text files...")
    print("-" * 60)
    try:
        from convert_all_pdfs import main as convert_main
        convert_main()
    except Exception as e:
        print(f"Error in PDF conversion: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Step 2: Check OpenAI API
    print("\n[Step 2/3] Checking OpenAI API configuration...")
    print("-" * 60)
    check_openai_api()
    
    # Step 3: Generate datasets
    print("\n[Step 3/3] Generating datasets from text files...")
    print("-" * 60)
    try:
        from generate_datasets import main as generate_main
        generate_main()
    except Exception as e:
        print(f"Error in dataset generation: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

