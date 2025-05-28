import argparse
import os

# It's good practice to handle potential import errors if these modules are part of the same package
# and the script might be run in different contexts.
try:
    from pdf_processing import extract_text_from_pdf
    from image_processing import extract_text_from_image
    from answer_extractor import extract_answers_with_ollama
    from question_paper_json_generator import generate_question_paper_json_from_text # Added
except ImportError:
    # Fallback for direct execution
    try:
        from src.pdf_processing import extract_text_from_pdf
        from src.image_processing import extract_text_from_image
        from src.answer_extractor import extract_answers_with_ollama
        from src.question_paper_json_generator import generate_question_paper_json_from_text # Added
    except ImportError as e:
        print(f"Error importing modules: {e}. Ensure the 'src' directory is in your PYTHONPATH or run from the project root.")
        exit(1)

import json


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
    Main function to parse arguments, process files, and orchestrate the two-stage answer extraction using Ollama.
    """
    parser = argparse.ArgumentParser(description="Extracts answers from student sheets using a two-stage Ollama process.")
    parser.add_argument("student_sheet_path", help="Path to the student's answer sheet (PDF, PNG, JPG, JPEG, or TXT for OCR).")
    parser.add_argument("question_paper_path", help="Path to the question paper with answers (key) file (PDF, image, or TXT for OCR).")
    parser.add_argument("--ollama-url", required=True, help="Base URL for the Ollama API (e.g., http://localhost:11434)")
    parser.add_argument("--ollama-model", default="gemma3:12b", help="Name of the Ollama model to use (default: gemma3:12b)")

    args = parser.parse_args()

    # Step A: Process Student Answer Sheet
    print("Step A: Processing student answer sheet...")
    student_text = get_text_from_file(args.student_sheet_path)
    if not student_text: # Handles None or empty string after strip more generally
        if student_text is None: # Specifically if file not found or major error
             print(f"Could not extract text from student sheet '{args.student_sheet_path}'. Exiting.")
             return
        else: # If file was read but OCR yielded empty text
            print(f"Warning: Student sheet '{args.student_sheet_path}' yielded empty text after OCR/reading. Processing will continue with empty student answers.")
            # student_text will be "" which is acceptable for extract_answers_with_ollama

    # Step B: Process "Question Paper with Answers (Key)" File
    print("\nStep B: Processing question paper with answers (key) file...")
    question_paper_ocr_text = get_text_from_file(args.question_paper_path)
    if not question_paper_ocr_text: # Handles None or empty string
        print(f"Could not extract text from question paper file '{args.question_paper_path}' or file is empty. Exiting.")
        return

    print("\nGenerating structured Question Paper JSON from its OCR text...")
    generated_qp_dict = generate_question_paper_json_from_text(
        ocr_text_of_question_paper_with_key=question_paper_ocr_text,
        ollama_base_url=args.ollama_url,
        ollama_model_name=args.ollama_model
    )

    if "error" in generated_qp_dict:
        print(f"Failed to generate Question Paper JSON: {generated_qp_dict.get('error')}")
        if "details" in generated_qp_dict:
            print(f"Details: {generated_qp_dict.get('details')}")
        if "raw_model_content" in generated_qp_dict:
            print(f"Raw Model Content for QP JSON generation (if available):\n{generated_qp_dict.get('raw_model_content', 'N/A')}")
        return

    # Convert the generated QP dict to a JSON string for the next step
    try:
        question_paper_json_string = json.dumps(generated_qp_dict)
        print("Successfully generated and serialized Question Paper JSON.")
    except TypeError as e:
        print(f"Error serializing generated Question Paper JSON: {e}. Exiting.")
        return


    # Step C: Extract Answers using Generated Question Paper JSON
    print("\nStep C: Extracting answers from student script using generated Question Paper JSON...")
    final_extraction_result = extract_answers_with_ollama(
        question_paper_json=question_paper_json_string,
        answer_script_ocr=student_text if student_text is not None else "", # Ensure student_text is not None
        ollama_base_url=args.ollama_url,
        ollama_model_name=args.ollama_model
    )

    # Step D: Output Handling
    print("\n--- Final Extracted Answers ---")
    if "error" in final_extraction_result:
        print(f"Answer extraction failed: {final_extraction_result.get('error')}")
        if "details" in final_extraction_result:
            print(f"Details: {final_extraction_result.get('details')}")
        # Check for different raw content keys from the two possible error sources
        if "raw_model_content" in final_extraction_result:
            print(f"Raw Model Content for Answer Extraction (if available):\n{final_extraction_result.get('raw_model_content', 'N/A')}")
        elif "raw_response" in final_extraction_result: # from ollama_client if it errored before model response
             print(f"Raw Ollama Client Response for Answer Extraction (if available):\n{final_extraction_result.get('raw_response', 'N/A')}")
    else:
        print(json.dumps(final_extraction_result, indent=4))


if __name__ == '__main__':
    main()
