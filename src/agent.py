import argparse
import os

# It's good practice to handle potential import errors if these modules are part of the same package
# and the script might be run in different contexts.
try:
    from pdf_processing import extract_text_from_pdf
    from image_processing import extract_text_from_image
    from scorer import score_answers
except ImportError:
    # Fallback for direct execution if modules are in the same directory
    # and not installed as a package.
    # This assumes src/ is the current working directory or in PYTHONPATH
    try:
        from src.pdf_processing import extract_text_from_pdf
        from src.image_processing import extract_text_from_image
        from src.scorer import score_answers
    except ImportError as e:
        print(f"Error importing modules: {e}. Ensure the 'src' directory is in your PYTHONPATH or run from the project root.")
        exit(1)


def get_text_from_file(file_path: str) -> str | None:
    """
    Extracts text from a file based on its extension.

    Supports PDF, common image formats (PNG, JPG, JPEG), and TXT files.

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
    elif file_extension == '.txt':
        print(f"Reading text from TXT file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file '{file_path}': {e}")
            return None
    else:
        print(f"Error: Unsupported file type '{file_extension}' for file '{file_path}'.")
        print("Supported types: .pdf, .png, .jpg, .jpeg, .txt")
        return None


def main():
    """
    Main function to parse arguments, process files, and score answers.
    """
    parser = argparse.ArgumentParser(description="Grade student answer sheets using Gemini API.")
    parser.add_argument("student_sheet_path", help="Path to the student's answer sheet (PDF, PNG, JPG, JPEG, or TXT).")
    parser.add_argument("answer_key_path", help="Path to the answer key (PDF, PNG, JPG, JPEG, or TXT).")
    parser.add_argument("api_key", help="Your Gemini API key.")

    args = parser.parse_args()

    print("Processing student answer sheet...")
    student_text = get_text_from_file(args.student_sheet_path)
    if student_text is None or student_text.strip() == "":
        print(f"Could not extract text from student sheet '{args.student_sheet_path}' or sheet is empty. Exiting.")
        return # Exit if student text extraction failed or is empty

    print("\nProcessing answer key...")
    answer_key_text = get_text_from_file(args.answer_key_path)
    if answer_key_text is None or answer_key_text.strip() == "":
        print(f"Could not extract text from answer key '{args.answer_key_path}' or key is empty. Exiting.")
        return # Exit if answer key extraction failed or is empty

    print("\nScoring answers...")
    # Ensure that empty strings (after stripping whitespace) are handled by score_answers or here.
    # The current implementation of get_text_from_file returns "" on some errors, not None.
    # Adding explicit checks for empty strings after extraction.

    scoring_result = score_answers(student_text, answer_key_text, args.api_key)

    print("\n--- Results ---")
    if "error" in scoring_result:
        print(f"Scoring failed: {scoring_result['error']}")
        if "details" in scoring_result:
            print(f"Details: {scoring_result['details']}")
        if "raw_response" in scoring_result:
            print(f"Raw Model Response (if available):\n{scoring_result['raw_response']}")
    else:
        score = scoring_result.get("score", "N/A")
        feedback = scoring_result.get("feedback", "No feedback provided.")
        print(f"Score: {score}%" if isinstance(score, (int, float)) else f"Score: {score}")
        print(f"Feedback:\n{feedback}")


if __name__ == '__main__':
    main()
