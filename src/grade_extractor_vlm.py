import argparse
import json
import os
import sys # For sys.exit and sys.stderr

from src.pdf_processing import convert_pdf_to_images_base64
from src.image_processing import load_image_as_base64
from src.vlm_extractor import extract_answers_with_vlm

def main():
    parser = argparse.ArgumentParser(description="Extract answers from student scripts using VLM.")
    parser.add_argument("question_paper_path", help="Path to the JSON file containing the question paper.")
    parser.add_argument("answer_script_path", help="Path to the student's answer script (PDF or image file like PNG, JPG).")
    parser.add_argument("output_path", help="Path to save the resulting JSON output.")
    parser.add_argument("--ollama_base_url", default="http://localhost:11434", help="Base URL for the Ollama API.")
    parser.add_argument("--ollama_model_name", default="gemma3:12b", help="Name of the Ollama model to use.")
    parser.add_argument("--ollama_options", default=None, help="JSON string for Ollama options (e.g., '{\"temperature\": 0.5}').")

    args = parser.parse_args()

    # 1. Load Question Paper
    question_paper_json_str = None
    try:
        with open(args.question_paper_path, 'r') as f:
            content = f.read()
            # Try to parse to validate and ensure it's a proper JSON string for the VLM
            parsed_content = json.loads(content)
            if isinstance(parsed_content, dict) or isinstance(parsed_content, list):
                question_paper_json_str = json.dumps(parsed_content)
            else: # Should ideally be a JSON object or array
                question_paper_json_str = content # Assume it's already a valid JSON string
    except FileNotFoundError:
        print(f"Error: Question paper file not found at '{args.question_paper_path}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in question paper file '{args.question_paper_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not read or process question paper file '{args.question_paper_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Process Answer Script
    images_base64 = []
    file_ext = os.path.splitext(args.answer_script_path)[1].lower()

    if file_ext == ".pdf":
        print(f"Processing PDF file: {args.answer_script_path}")
        images_base64 = convert_pdf_to_images_base64(args.answer_script_path)
        if not images_base64:
            print(f"Error: Could not extract any images from PDF '{args.answer_script_path}'. It might be empty, corrupted, or unreadable.", file=sys.stderr)
            sys.exit(1)
    elif file_ext in [".png", ".jpg", ".jpeg"]:
        print(f"Processing image file: {args.answer_script_path}")
        single_image_base64 = load_image_as_base64(args.answer_script_path)
        if single_image_base64:
            images_base64 = [single_image_base64]
        else:
            print(f"Error: Could not load image '{args.answer_script_path}'. It might be corrupted or an unsupported format.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Unsupported answer script file type '{file_ext}'. Please provide a PDF, PNG, JPG, or JPEG file.", file=sys.stderr)
        sys.exit(1)

    if not images_base64:
        print(f"Error: No images were obtained from the answer script '{args.answer_script_path}'.", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully obtained {len(images_base64)} image(s) from the answer script.")

    # 3. Parse Ollama Options
    parsed_ollama_options = None
    if args.ollama_options:
        try:
            parsed_ollama_options = json.loads(args.ollama_options)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON string for --ollama_options: {e}", file=sys.stderr)
            sys.exit(1)

    # 4. Call VLM Extractor
    print(f"Sending request to VLM model '{args.ollama_model_name}' at '{args.ollama_base_url}'...")
    extraction_result = extract_answers_with_vlm(
        question_paper_json_str=question_paper_json_str,
        images_base64=images_base64,
        ollama_base_url=args.ollama_base_url,
        ollama_model_name=args.ollama_model_name,
        ollama_options=parsed_ollama_options
    )

    # 5. Save Output
    if "error" in extraction_result:
        print(f"Error from VLM extractor: {extraction_result['error']}", file=sys.stderr)
        # Optionally, print more details if available
        if "raw_response" in extraction_result:
            print(f"Raw VLM response: {extraction_result['raw_response']}", file=sys.stderr)
        if "raw_model_content" in extraction_result:
             print(f"Raw model content: {extraction_result['raw_model_content']}", file=sys.stderr)
        # Decide if to write the error to file or just exit
        # For now, we'll still write the error to the output file as per typical API error responses
    
    try:
        with open(args.output_path, 'w') as f:
            json.dump(extraction_result, f, indent=2)
        print(f"Successfully wrote VLM output to '{args.output_path}'")
    except IOError as e:
        print(f"Error: Could not write output to file '{args.output_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while writing output: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
