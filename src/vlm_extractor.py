import json
from src.ollama_client import send_request_to_ollama

def extract_answers_with_vlm(
    question_paper_json_str: str,
    images_base64: list[str],
    ollama_base_url: str,
    ollama_model_name: str = "gemma3:12b",
    ollama_options: dict = None
) -> dict:
    """
    Extracts answers from student answer script images using a VLM,
    guided by a question paper JSON.

    Args:
        question_paper_json_str: JSON string of the question paper.
        images_base64: List of base64 encoded image strings of the answer script.
        ollama_base_url: Base URL for the Ollama API.
        ollama_model_name: Name of the Ollama model to use.
        ollama_options: Optional dictionary of Ollama-specific options.

    Returns:
        A dictionary containing the extracted answers and metadata, or an error dictionary.
    """

    system_prompt_content = '''**Task:** Extract all possible answers from the student's answer script images and map them to the corresponding question numbers in the provided question paper. Also, extract any relevant metadata.

        **Important:** Include all questions (including OR variants) as they appear in the question paper. If the student did not answer a question or its OR variant, mark it as "No Answer". If OCR fails for a given question, mark that question as "OCR Failed". Do not omit any question or OR question variation.

        **Detailed Steps:**
        1. **Text Extraction:** Consider all text from the provided answer sheet images.
        2. **Answer Mapping:** For every question and OR variant in the question paper, attempt to locate and extract the student's corresponding answer from the answer script.
           - If multiple pages or images are involved, combine all parts of the answer.
           - If the answer is missing, explicitly mark it as "No Answer".
           - If OCR fails, use "OCR Failed".
           - Include all questions and their OR variants in the final output.
        3. **Metadata Extraction:**
           - Extract metadata (e.g., student name, roll number, exam_date, subject) from the script if present.
           - If any piece of metadata is missing, leave it empty or as 'N/A' if instructed.
        4. **Organization:**
           - Return metadata under the `exam` field.
           - Return a fully structured list of questions (including all OR variants) and sub-questions with their answers in `data`.
           - Maintain the order and structure given by the question paper. If questions or OR variants are absent or not answered, mark them accordingly.

        **Challenges and Notes:**
        - Student answers might not follow the question order.
        - Some answers might be split across multiple pages.
        - DO NOT add information that is not present in the answer script. No interpretation or paraphrasing.
        - Extract diagram text if readable and associate it with the correct question.'''

    user_prompt_content = f'''
            **Question Paper (For Reference Only, Do Not Include Answers):**
            ```{question_paper_json_str}```

            [The VLM should use the provided images to extract answers based on the question paper.]
            '''

    # Combine system and user prompts into a single string for send_request_to_ollama
    combined_prompt = f"System Prompt:\n{system_prompt_content}\n\nUser Prompt:\n{user_prompt_content}"

    response_dict = send_request_to_ollama(
        prompt=combined_prompt,
        base_url=ollama_base_url,
        model_name=ollama_model_name,
        options=ollama_options,
        images=images_base64
    )

    if "error" in response_dict:
        return response_dict  # Return error from client or API request

    model_content_text = response_dict.get("response")

    if not model_content_text or not isinstance(model_content_text, str):
        return {
            "error": "No valid 'response' field in Ollama output",
            "raw_response": response_dict
        }

    try:
        parsed_output = json.loads(model_content_text)
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to decode JSON response from VLM: {e}",
            "raw_model_content": model_content_text
        }

    if not isinstance(parsed_output, dict):
        return {
            "error": "VLM output is not a JSON object (dictionary)",
            "raw_model_content": model_content_text,
            "parsed_type": type(parsed_output).__name__
        }

    if "data" not in parsed_output or "exam" not in parsed_output:
        return {
            "error": "VLM output JSON does not contain required 'data' and 'exam' keys",
            "missing_keys": [k for k in ["data", "exam"] if k not in parsed_output],
            "raw_model_content": model_content_text
        }

    return parsed_output


if __name__ == '__main__':
    # This is a conceptual example.
    # To run this, you would need:
    # 1. A running Ollama instance with a VLM model (e.g., "gemma3:12b" if it's multimodal or a specific VLM like llava).
    # 2. Actual base64 encoded images.
    # 3. A sample question paper JSON string.

    print("Example usage of extract_answers_with_vlm (conceptual):")

    sample_question_paper_str = json.dumps({
        "exam_name": "Midterm Exam",
        "questions": [
            {"question_number": "1", "text": "What is 2+2?", "marks": 5},
            {"question_number": "2", "text": "Explain photosynthesis.", "marks": 10},
            {
                "question_number": "3",
                "text": "Solve for x:",
                "sub_questions": [
                    {"question_number": "3.a", "text": "x + 5 = 10", "marks": 3},
                    {"question_number": "3.b", "text": "2x - 8 = 0", "marks": 3}
                ]
            }
        ]
    })

    # Dummy base64 image data (replace with actual image data)
    sample_images_base64 = [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # Minimal valid PNG
    ]
    
    ollama_api_base_url = "http://localhost:11434" # Replace if your Ollama runs elsewhere
    # Ensure the model used supports vision and follows instructions for JSON output.
    # For this example, we'll use a placeholder model name.
    # The actual model name might be something like "llava" or a multimodal gemma variant.
    vlm_model_name = "llava" # Or your specific VLM model

    print(f"Simulating call to extract_answers_with_vlm with model: {vlm_model_name}")
    print("Note: This example will likely fail if a VLM model is not running or if the model "
          "does not return the expected JSON structure without actual images and prompting.")

    # Mocking the send_request_to_ollama function for this example
    # to avoid actual API calls during this illustrative run.
    original_send_request = send_request_to_ollama
    def mock_send_request_to_ollama(prompt, base_url, model_name, options, images):
        print(f"Mocked send_request_to_ollama called for model {model_name} at {base_url}")
        print(f"Prompt length: {len(prompt)}")
        print(f"Number of images: {len(images)}")
        # Simulate a successful VLM response that is a JSON string
        mock_response_content = {
            "exam": {
                "student_name": "John Doe",
                "roll_number": "JD123",
                "subject": "Sample Subject",
                "exam_date": "2024-07-30"
            },
            "data": [
                {"question_number": "1", "answer": "4"},
                {"question_number": "2", "answer": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create their own food."},
                {"question_number": "3.a", "answer": "x = 5"},
                {"question_number": "3.b", "answer": "x = 4"},
                {"question_number": "4 (OR)", "answer": "No Answer"} 
            ]
        }
        return {"response": json.dumps(mock_response_content), "done": True}

    # Replace the actual function with the mock for the purpose of this example
    globals()['send_request_to_ollama'] = mock_send_request_to_ollama

    extracted_data = extract_answers_with_vlm(
        question_paper_json_str=sample_question_paper_str,
        images_base64=sample_images_base64,
        ollama_base_url=ollama_api_base_url,
        ollama_model_name=vlm_model_name,
        ollama_options={"temperature": 0.1} # Example option
    )

    # Restore the original function
    globals()['send_request_to_ollama'] = original_send_request

    if "error" in extracted_data:
        print(f"\nError extracting answers: {extracted_data['error']}")
        if "raw_model_content" in extracted_data:
            print(f"Raw model content was: {extracted_data['raw_model_content']}")
        if "raw_response" in extracted_data:
            print(f"Raw response was: {extracted_data['raw_response']}")
    else:
        print("\nSuccessfully extracted data (mocked):")
        print(json.dumps(extracted_data, indent=2))

    # Example of a simulated error case: VLM returns non-JSON string
    def mock_send_request_bad_json(prompt, base_url, model_name, options, images):
        return {"response": "This is not JSON.", "done": True}
    
    globals()['send_request_to_ollama'] = mock_send_request_bad_json
    extracted_data_bad_json = extract_answers_with_vlm(
        sample_question_paper_str, sample_images_base64, ollama_api_base_url, vlm_model_name
    )
    print("\nSimulating VLM returning non-JSON string:")
    print(json.dumps(extracted_data_bad_json, indent=2))
    globals()['send_request_to_ollama'] = original_send_request

    # Example of a simulated error case: VLM returns JSON but wrong structure
    def mock_send_request_wrong_structure(prompt, base_url, model_name, options, images):
        return {"response": json.dumps({"metadata": {}, "answers": []}), "done": True} # Missing 'data' and 'exam'
    
    globals()['send_request_to_ollama'] = mock_send_request_wrong_structure
    extracted_data_wrong_structure = extract_answers_with_vlm(
        sample_question_paper_str, sample_images_base64, ollama_api_base_url, vlm_model_name
    )
    print("\nSimulating VLM returning JSON with wrong structure:")
    print(json.dumps(extracted_data_wrong_structure, indent=2))
    globals()['send_request_to_ollama'] = original_send_request

    # Example of a simulated error case: Ollama client returns an error
    def mock_send_request_ollama_error(prompt, base_url, model_name, options, images):
        return {"error": "Ollama server not found"}
    
    globals()['send_request_to_ollama'] = mock_send_request_ollama_error
    extracted_data_ollama_error = extract_answers_with_vlm(
        sample_question_paper_str, sample_images_base64, ollama_api_base_url, vlm_model_name
    )
    print("\nSimulating Ollama client error:")
    print(json.dumps(extracted_data_ollama_error, indent=2))
    globals()['send_request_to_ollama'] = original_send_request

    print("\nConceptual example finished.")
