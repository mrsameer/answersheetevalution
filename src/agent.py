import argparse
import os

# It's good practice to handle potential import errors if these modules are part of the same package
# and the script might be run in different contexts.
try:
    from pdf_processing import extract_text_from_pdf
    from image_processing import extract_text_from_image
    # from scorer import score_answers # Removed as per new requirements
    from answer_extractor import extract_answers_with_ollama
except ImportError:
    # Fallback for direct execution
    try:
        from src.pdf_processing import extract_text_from_pdf
        from src.image_processing import extract_text_from_image
        # from src.scorer import score_answers # Removed
        from src.answer_extractor import extract_answers_with_ollama
    except ImportError as e:
        print(f"Error importing modules: {e}. Ensure the 'src' directory is in your PYTHONPATH or run from the project root.")
        exit(1)

import json # Added for JSON output formatting


def get_text_from_file(file_path: str) -> str | None:
    """
    Extracts text from a file based on its extension.

    Supports PDF, common image formats (PNG, JPG, JPEG), TXT, and JSON files.

    Args:
        file_path: The path to the file.

    Returns:
        The extracted text as a string, or None if the file is not found,
        the file type is unsupported, or an error occurs during extraction.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == '.pdf':
        print(f"Extracting text from PDF: {file_path}")
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        print(f"Extracting text from image: {file_path}")
        return extract_text_from_image(file_path)
    elif file_extension == '.txt' or file_extension == '.json': # Added .json
        print(f"Reading text from {file_extension.upper()} file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text/json file '{file_path}': {e}")
            return None
    else:
        print(f"Error: Unsupported file type '{file_extension}' for file '{file_path}'.")
        print("Supported types: .pdf, .png, .jpg, .jpeg, .txt, .json")
        return None


def main():
    """
    Main function to parse arguments, process files, and extract answers using Ollama.
    """
    parser = argparse.ArgumentParser(description="Extract answers from student sheets using Ollama.")
    parser.add_argument("student_sheet_path", help="Path to the student's answer sheet (PDF, PNG, JPG, JPEG, or TXT for OCR).")
    parser.add_argument("question_paper_path", help="Path to the question paper JSON file.")
    parser.add_argument("--ollama-url", required=True, help="Base URL for the Ollama API (e.g., http://localhost:11434)")
    parser.add_argument("--ollama-model", default="gemma3:12b", help="Name of the Ollama model to use (default: gemma3:12b)")

    args = parser.parse_args()

    print("Processing student answer sheet...")
    student_text = get_text_from_file(args.student_sheet_path)
    if student_text is None: # Check for None, empty string might be valid for OCR if sheet is blank
        print(f"Could not extract text from student sheet '{args.student_sheet_path}'. Exiting.")
        return
    if student_text.strip() == "":
        print(f"Warning: Student sheet '{args.student_sheet_path}' yielded empty text after OCR/reading.")
        # Continue, as an empty answer sheet might be valid input for the LLM

    print("\nReading question paper JSON...")
    question_paper_json_string = get_text_from_file(args.question_paper_path)
    if question_paper_json_string is None:
        print(f"Could not read question paper JSON from '{args.question_paper_path}'. Exiting.")
        return
    if question_paper_json_string.strip() == "":
        print(f"Error: Question paper JSON file '{args.question_paper_path}' is empty. Exiting.")
        return

    # Validate if question_paper_json_string is valid JSON before sending to Ollama
    try:
        json.loads(question_paper_json_string) # Try parsing to check validity
        print("Question paper JSON successfully read and parsed.")
    except json.JSONDecodeError as e:
        print(f"Error: Question paper '{args.question_paper_path}' is not valid JSON. Details: {e}. Exiting.")
        return

    print("\nExtracting answers with Ollama...")
    extraction_result = extract_answers_with_ollama(
        question_paper_json=question_paper_json_string, # Ensure kwarg name matches function def
        answer_script_ocr=student_text,                 # Ensure kwarg name matches function def
        ollama_base_url=args.ollama_url,
        ollama_model_name=args.ollama_model
    )

    print("\n--- Results ---")
    if "error" in extraction_result:
        print(f"Answer extraction failed: {extraction_result['error']}")
        if "details" in extraction_result:
            print(f"Details: {extraction_result['details']}")
        if "raw_model_content" in extraction_result: # from answer_extractor
            print(f"Raw Model Content (if available):\n{extraction_result['raw_model_content']}")
        elif "raw_response" in extraction_result: # from ollama_client
             print(f"Raw Ollama Client Response (if available):\n{extraction_result['raw_response']}")
    else:
        # Successfully extracted data, print as formatted JSON
        print("Extracted data:")
        print(json.dumps(extraction_result, indent=4))


if __name__ == '__main__':
    main()
