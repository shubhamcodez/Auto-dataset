#!/usr/bin/env python3
"""
Test script to process only page 20 from the book.
"""

import os
import sys

# Set API key if not in environment (you can set it here for testing)
if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_KEY"):
    # Uncomment and add your API key here for testing:
    os.environ["OPENAI_API_KEY"] = ""

from generate_datasets import (
    read_pages, detect_content_categories, generate_multiple_instruction_tuning_datasets,
    generate_reasoning_tool_dataset
)
import json

def test_page_20():
    """Test processing page 20 only."""
    # Check if book_txt directory exists, if not, suggest running convert_all_pdfs.py
    book_txt_base = "book_txt"
    book_name = "Practical Guide to Quant finance interviews"
    
    # Sanitize the book name to match what pdf_to_pages.py would create
    import re
    safe_name = re.sub(r"[\\/:*?\"<>|]+", "_", book_name).strip()
    book_dir = os.path.join(book_txt_base, safe_name)
    
    if not os.path.exists(book_txt_base):
        print(f"Error: '{book_txt_base}' directory not found.")
        print("Please run: python convert_all_pdfs.py")
        return
    
    if not os.path.exists(book_dir):
        print(f"Error: Book directory '{book_dir}' not found.")
        print(f"Expected directory: {book_dir}")
        print("Please run: python convert_all_pdfs.py")
        return
    
    output_dir = "dataset"
    
    # Create book-specific output directory
    book_output_dir = os.path.join(output_dir, book_name)
    os.makedirs(book_output_dir, exist_ok=True)
    
    # Read pages 18-20 (3-page window centered on 20)
    start_page = 24
    print(f"Testing pages {start_page}-{start_page+2} (centered on page 20)...")
    
    pages_text = read_pages(book_dir, start_page, 3)
    if not pages_text:
        print(f"Error: Could not read pages {start_page}-{start_page+2}")
        return
    
    page_range = f"{start_page}-{start_page+2}"
    
    print(f"\n{'='*60}")
    print(f"Content from pages {page_range}:")
    print(f"{'='*60}")
    print(pages_text[:500] + "..." if len(pages_text) > 500 else pages_text)
    print(f"{'='*60}\n")
    
    # Detect content categories
    print("Categorizing content...")
    categories = detect_content_categories(pages_text)
    
    instruction_tuning = categories.get("instruction_tuning", False)
    reasoning = categories.get("reasoning", False)
    tool_use = categories.get("tool_use", False)
    
    print(f"Categories detected:")
    print(f"  - Instruction-tuning: {instruction_tuning}")
    print(f"  - Reasoning: {reasoning}")
    print(f"  - Tool-use: {tool_use}\n")
    
    if not (instruction_tuning or reasoning or tool_use):
        print("No relevant content categories detected.")
        return
    
    # Generate datasets based on detected categories
    dataset_count = 0
    
    # 1. Instruction-tuning datasets
    if instruction_tuning:
        print("Generating instruction-tuning datasets (extracting multiple Q&A pairs)...")
        datasets = generate_multiple_instruction_tuning_datasets(pages_text, book_name, page_range, num_iterations=3)
        if datasets:
            # Save each Q&A pair as a separate JSON file
            for idx, dataset in enumerate(datasets, 1):
                dataset_count += 1
                output_file = os.path.join(book_output_dir, f"instruction_p{start_page}_{dataset_count}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Saved instruction-tuning: {output_file}")
            print(f"  ✓ Generated {len(datasets)} instruction-tuning dataset(s)")
        else:
            print("  ✗ Failed to generate instruction-tuning datasets")
    
    # 2. Reasoning datasets
    if reasoning:
        print("Generating reasoning dataset...")
        dataset = generate_reasoning_tool_dataset(pages_text, book_name, page_range)
        if dataset:
            dataset_count += 1
            # Test the tool code and mark success
            from generate_datasets import test_tool_code
            test_id = f"{book_name.replace(' ', '_')}_reasoning_p{start_page}_{dataset_count}"
            success, test_file = test_tool_code(dataset, test_id)
            dataset["success"] = "yes" if success else "no"
            
            if success:
                print(f"  ✓ Test passed: {test_file}")
            else:
                print(f"  ✗ Test failed: {test_file}")
            
            output_file = os.path.join(book_output_dir, f"reasoning_p{start_page}_{dataset_count}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved reasoning: {output_file} (success: {dataset['success']})")
        else:
            print("  ✗ Failed to generate reasoning dataset")
    
    # 3. Tool-use datasets
    if tool_use:
        print("Generating tool-use dataset...")
        dataset = generate_reasoning_tool_dataset(pages_text, book_name, page_range)
        if dataset:
            dataset_count += 1
            # Test the tool code and mark success
            from generate_datasets import test_tool_code
            test_id = f"{book_name.replace(' ', '_')}_tool_use_p{start_page}_{dataset_count}"
            success, test_file = test_tool_code(dataset, test_id)
            dataset["success"] = "yes" if success else "no"
            
            if success:
                print(f"  ✓ Test passed: {test_file}")
            else:
                print(f"  ✗ Test failed: {test_file}")
            
            output_file = os.path.join(book_output_dir, f"tool_use_p{start_page}_{dataset_count}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved tool-use: {output_file} (success: {dataset['success']})")
        else:
            print("  ✗ Failed to generate tool-use dataset")
    
    print(f"\n{'='*60}")
    print(f"Test complete! Generated {dataset_count} dataset(s).")
    print(f"Check output in: {book_output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_page_20()

