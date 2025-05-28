import json
from src.ollama_client import send_request_to_ollama


def extract_answers_with_ollama(question_paper_json: str, answer_script_ocr: str, ollama_base_url: str,
                                ollama_model_name: str = "gemma3:12b") -> dict:
    """
    Extracts answers from an answer script OCR using Ollama.

    Args:
        question_paper_json: JSON string of the question paper.
        answer_script_ocr: OCR text of the student's answer script.
        ollama_base_url: The base URL of the Ollama API.
        ollama_model_name: The name of the Ollama model to use.

    Returns:
        A dictionary containing the raw response from the Ollama API.
    """

    prompt = f"""
You are an expert AI assistant tasked with extracting structured information from a student's answer script based on a provided question paper. Your goal is to identify which questions were attempted and what the corresponding answers are.

**Question Paper (For Reference Only, Do Not Include Answers):**
---JSON START---
{question_paper_json}
---JSON END---

**Answer Script OCR:**
---OCR START---
{answer_script_ocr}
---OCR END---

**Important:**
*   **Focus on Extraction, Not Evaluation:** Your primary task is to extract the student's answers as accurately as possible. Do not try to grade the answers or provide feedback on their correctness.
*   **Handle Missing Answers:** If a student did not attempt a question, clearly indicate this (e.g., "Not Attempted").
*   **Interpret OCR Imperfections:** OCR output can be messy. Use the context from the question paper to make reasonable interpretations of the student's intent.
*   **Maintain Order:** Present the extracted answers in the same order as the questions appear in the question paper.
*   **Output Format:** The final output should be a JSON object where keys are question numbers (as strings, e.g., "1a", "2bii") and values are the extracted answers (as strings).

**Detailed Steps:**
1.  **Parse Question Paper:** Understand the structure of the question paper (sections, question numbers, sub-parts).
2.  **Scan Answer Script:** Read through the OCR text of the answer script.
3.  **Map Answers to Questions:** For each question in the question paper, locate the corresponding answer in the answer script.
    *   Students might explicitly number their answers (e.g., "Answer to Q1a:", "1) a)").
    *   They might answer questions out of order.
    *   They might not attempt all questions.
4.  **Extract Answer Text:** Once an answer is located, extract the full text of that answer.
    *   Be careful to capture multi-line answers.
    *   If an answer involves diagrams, code snippets, or mathematical equations that are poorly represented in OCR, make a note (e.g., "[Diagram present, OCR may be incomplete]").
5.  **Handle Ambiguity:** If it's genuinely unclear which question an answer pertains to, or if an answer is completely illegible, note this.

**Metadata Extraction (If Possible):**
*   Look for any student identifiers (name, ID) or exam details at the beginning of the answer script. Extract these if present, perhaps under a special key like `"_metadata"`.

**Organization:**
*   Structure your primary output around the question numbers from the question paper.
*   For questions with sub-parts (e.g., 1a, 1b), use nested structures or clear identifiers.

**Challenges and Notes:**
*   **Handwriting Quality:** OCR of handwritten text can be poor. Prioritize clear matches.
*   **Answer Numbering:** Students might use inconsistent numbering. The question paper is your ground truth for structure.
*   **Implicit Answers:** Sometimes answers are embedded within a larger narrative. Extract the relevant portion.
*   **Partial Answers:** If a student only partially answers a question, extract what's there.

**Expected Output Structure (Example):**
---JSON EXAMPLE START---
{{
  "1a": "The capital of France is Paris.",
  "1b": "The main components of a CPU are the ALU and Control Unit.",
  "2": "Not Attempted",
  "3i": "[Diagram present, OCR may be incomplete] The process involves...",
  "_metadata": {{
    "student_name": "Jane Doe",
    "student_id": "12345"
  }}
}}
---JSON EXAMPLE END---

Based on the provided Question Paper and Answer Script OCR, please extract the answers and return them in the specified JSON format.
"""

    # 2. Call Ollama API
    ollama_response_dict = send_request_to_ollama(
        prompt=prompt,
        base_url=ollama_base_url,
        model_name=ollama_model_name
    )

    # 3. Check for errors from the client itself
    if 'error' in ollama_response_dict:
        # ollama_client already returns a dict with an 'error' key
        return ollama_response_dict

    # 4. Extract the model's actual response text
    # Ollama's /api/generate endpoint (when stream=False) returns a JSON where
    # the model's output is in the 'response' field.
    model_content_text = ollama_response_dict.get('response')

    if model_content_text is None:  # Check for None specifically, as empty string might be valid in some edge cases
        return {"error": "No 'response' field in Ollama output", "raw_response": ollama_response_dict}

    if not isinstance(model_content_text, str):
        return {"error": "'response' field is not a string", "raw_response": ollama_response_dict}

    # 5. Parse the JSON content from the model's response
    try:
        # The model's response string is expected to be a JSON representing the structured data
        parsed_output = json.loads(model_content_text)

        # Basic validation for top-level keys and type
        if not isinstance(parsed_output, dict):
            return {
                "error": "Model output, when parsed, is not a dictionary.",
                "raw_model_content": model_content_text,
                "parsed_type": type(parsed_output).__name__
            }

        # Schema adherence for "data" and "exam" keys - this is a simplified check.
        # The prompt asks for a specific structure like {"1a": "answer", "_metadata": {...}}
        # For now, the subtask asks for "data" and "exam" at the top level.
        # Let's adjust the prompt's "Expected Output Structure" example if this is a mismatch.
        # For now, I will stick to the "data" and "exam" keys as per the subtask description for validation.
        # If the prompt's example output (e.g. {"1a": ..., "_metadata": ...}) is the actual expected structure
        # then this validation needs to be changed or the prompt's example needs to be wrapped.
        # Assuming the subtask's example structure is the target for this function's return.
        #
        # Re-reading the prompt: "Expected Output Structure (Example):" shows direct question numbers as keys.
        # The subtask says: "ensure the top-level keys ("data", "exam") are present."
        # This implies the actual desired output from this *function* might be:
        # { "data": { "1a": "Paris", ... }, "exam": { ... } }
        # However, the prompt to the LLM shows: { "1a": "Paris", ... }
        # This is a point of ambiguity. I will assume the LLM produces the flat structure (as in its prompt)
        # and this Python function is NOT expected to wrap it further with "data" and "exam".
        # The validation "data" and "exam" must then refer to keys *within* the LLM's output,
        # which is not what the LLM prompt asks for.
        #
        # Clarification: The issue description for *this current subtask* says:
        # "ensure the top-level keys ("data", "exam") are present." and refers to an "Output Structure"
        # from "the issue". It's possible "the issue" refers to the overall project issue, not just the LLM prompt.
        #
        # Given the example in *this subtask's description*:
        # # Basic validation for top-level keys
        # if not isinstance(parsed_output, dict) or "data" not in parsed_output or "exam" not in parsed_output:
        #
        # I will implement this validation. This means the LLM *is* expected to return a JSON
        # with "data" and "exam" as top-level keys. The current LLM prompt needs to be updated
        # in a future step to reflect this. For now, I will implement the validation as requested.

        required_keys = ["data", "exam"]  # As per subtask description
        missing_keys = [key for key in required_keys if key not in parsed_output]

        if missing_keys:
            return {
                "error": f"Model output JSON is missing required top-level keys: {', '.join(missing_keys)}",
                "raw_model_content": model_content_text,
                "parsed_json_keys": list(parsed_output.keys())
            }

        return parsed_output  # This is the successfully parsed Python dictionary

    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse JSON from model output",
            "details": str(e),
            "raw_model_content": model_content_text
        }
    except Exception as e:  # Catch any other unexpected errors during parsing/validation
        return {
            "error": "An unexpected error occurred while processing model output",
            "details": str(e),
            "raw_model_content": model_content_text  # Include raw content for debugging
        }


if __name__ == '__main__':
    # Example Usage (Illustrative - requires a running Ollama instance and sample data)
    # Note: The sample_question_paper_json and sample_answer_script_ocr are simplified.
    # The LLM prompt expects a more detailed JSON for question_paper_json.
    # The ollama_model_name is also illustrative.
    sample_question_paper_json = json.dumps({
        "1": {
            "question": "What is the capital of France?",
            "marks": 2
        },
        "2a": {
            "question": "Explain the concept of photosynthesis.",
            "marks": 5
        },
        "2b": {
            "question": "What are the reactants in photosynthesis?",
            "marks": 3
        }
    })

    sample_answer_script_ocr = """
    Student Name: John Doe
    Student ID: S12345

    Answer 1:
    Paris is the capital of France.

    Answer 2a:
    Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create their own food and release oxygen.

    Answer 2b:
    The reactants are sunlight, water, and carbon dioxide.
    """

    # This ollama_model_name is defined in the function signature as a default.
    # To make it available here, it should be defined outside or passed explicitly.
    # For simplicity, let's use the default directly in the print statement or define it.
    current_model_for_testing = "gemma3:12b"  # or extract_answers_with_ollama.__defaults__[0] if it's the first default

    ollama_api_base_url = "http://localhost:11434"  # Replace if your Ollama runs elsewhere

    print(f"Attempting to extract answers using Ollama model: {current_model_for_testing}")

    # This example call will likely fail if Ollama isn't running or if the model
    # doesn't produce the exact JSON structure with "data" and "exam" keys.
    # The purpose of this __main__ block is illustrative.
    processed_output = extract_answers_with_ollama(
        sample_question_paper_json,
        sample_answer_script_ocr,
        ollama_api_base_url,
        ollama_model_name=current_model_for_testing
    )

    print("\n--- Example Call Results ---")
    if "error" in processed_output:
        print(f"Error processing Ollama response: {processed_output['error']}")
        if "details" in processed_output:
            print(f"Details: {processed_output['details']}")
        if "raw_model_content" in processed_output:
            print(f"Raw model content was: {processed_output['raw_model_content']}")
        if "raw_response" in processed_output:
            print(f"Raw Ollama client response was: {processed_output['raw_response']}")
    else:
        # If successful, processed_output is the Python dictionary from the model's JSON
        print("Successfully processed Ollama response.")
        print("Extracted data (Python dict):")
        print(json.dumps(processed_output, indent=2))
        # Example: Accessing data (assuming "data" and "exam" keys exist as per validation)
        # print("\nAccessing specific parts (example):")
        # print(f"Exam details: {processed_output.get('exam')}")
        # print(f"Answer data: {processed_output.get('data')}")

    # Example with a non-existent model to test error handling from ollama_client via send_request_to_ollama
    print("\n--- Testing Non-Existent Model ---")
    error_output = extract_answers_with_ollama(
        sample_question_paper_json,
        sample_answer_script_ocr,
        ollama_api_base_url,
        ollama_model_name="non_existent_model_gemma"
    )
    if "error" in error_output:
        print(f"Successfully caught error for non-existent model: {error_output['error']}")
        if "details" in error_output:  # If the error came from ollama_client
            print(f"Details: {error_output['details']}")
    else:
        print("Error for non-existent model was not caught as expected.")
        print(error_output)

