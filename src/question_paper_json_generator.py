import json
from src.ollama_client import send_request_to_ollama
import re  # For basic JSON cleanup


def generate_question_paper_json_from_text(ocr_text_of_question_paper_with_key: str, ollama_base_url: str,
                                           ollama_model_name: str = "gemma3:12b") -> dict:
    """
    Generates a structured JSON for a question paper from its OCR text (including answers) using Ollama.

    Args:
        ocr_text_of_question_paper_with_key: Raw OCR text of the question paper, including answers.
        ollama_base_url: The base URL of the Ollama API.
        ollama_model_name: The name of the Ollama model to use.

    Returns:
        A Python dictionary representing the structured question paper, or an error dictionary.
    """

    prompt_template = """
You are an expert AI assistant tasked with converting a raw OCR text dump of a question paper (which includes answers) into a structured JSON format. This JSON will be used by other AI systems to understand the questions, their structure, associated marks, and the correct answers.

**Input:**
You will receive a single block of text which is the OCR output of a question paper that also contains the answers to the questions.

**Output Structure:**
The output MUST be a single JSON object. This JSON object must have two top-level keys:
1.  `"exam"`: An array containing a single object with metadata about the exam.
2.  `"data"`: An array of objects, where each object represents a main question.

**Field Definitions:**

**1. `exam` (Array, contains one object):**
   *   `id`: (String) A unique identifier for this exam. UUID format preferred.
   *   `exam_name`: (String) The name of the exam (e.g., "Midterm Exam", "Class Test"). Extract if available, otherwise use "N/A".
   *   `subject`: (String) Subject of the exam (e.g., "Mathematics", "Physics"). Extract if available, otherwise use "N/A".
   *   `standard`: (String) Class or grade level (e.g., "10th Grade", "Class V"). Extract if available, otherwise use "N/A".
   *   `total_marks`: (Float/Integer) Total marks for the exam. Extract if available, otherwise use 0.0.
   *   `duration`: (String) Duration of the exam (e.g., "2 hours", "90 minutes"). Extract if available, otherwise use "N/A".
   *   `school_name`: (String) Name of the school or institution. Extract if available, otherwise use "N/A".
   *   `exam_instructions`: (String) General instructions for the exam. Extract if available, otherwise use "N/A".
   *   `exam_date`: (String) Date of the exam. Extract if available, otherwise use "N/A".
   *   `class_section`: (String) Section of the class, if applicable. Extract if available, otherwise use "N/A".
   *   `exam_set`: (String) Exam set identifier (e.g., "Set A", "Set 1"). Extract if available, otherwise use "N/A".
   *   `total_questions`: (Integer) The total number of main questions in the paper. Calculate this by counting the top-level items in the "data" array.

**2. `data` (Array of question objects):**
   Each object in this array represents a main question (e.g., Q1, Q2a, Question 3).
   *   `id`: (String) A unique identifier for this main question. UUID format preferred.
   *   `question_number`: (String) The designation of the question (e.g., "1", "2a", "Section A Q1"). Try to capture this as accurately as it appears.
   *   `question_section`: (String) The section of the paper this question belongs to (e.g., "Section A", "Part 1"). If not specified, use "N/A".
   *   `supporting_text`: (String) Any introductory text or passage that applies to this main question but precedes its sub-questions (if any). If none, use an empty string "".
   *   `marks`: (Float/Integer) Marks allocated to this main question (sum of its sub-questions if applicable, or marks for the question itself if no sub-questions). If not specified, use 0.0.
   *   `contains_OR_questions`: (Boolean) Set to `true` if this main question offers an internal choice (e.g., "Answer Q3a OR Q3b"). Otherwise, `false`.
   *   `questions`: (Array of sub-question objects) Contains the actual question text(s). If a main question has no sub-parts (e.g. "Q1. What is X?"), this array will still contain one object for that question.
      *   `id`: (String) A unique identifier for this specific question/sub-question. UUID format preferred.
      *   `subquestion_number`: (String) The sub-question number (e.g., "a", "i", "part 1"). If the main question has no sub-numbering, use an empty string "".
      *   `question_text`: (String) The full text of the question.
      *   `question_instruction`: (String) Any specific instruction for this question (e.g., "Fill in the blank", "Choose the correct option", "Max 100 words"). If none, use an empty string "".
      *   `question_diagram_description`: (String) A brief description if a diagram is associated with/part of the question text. E.g., "Diagram shows a circuit." If none, use an empty string "".
      *   `answer_text`: (String) The **correct answer** to this question/sub-question. This is critical. Extract this from the provided text. If the answer is not found or is ambiguous, use "Answer not found".

**Important Considerations:**
*   **Uniqueness of IDs:** Ensure all `id` fields (for exam, main questions, and sub-questions) are unique strings. UUID format is preferred but any unique string is acceptable.
*   **Accuracy:** Extract all information as accurately as possible from the provided text.
*   **Completeness:** Try to populate all fields. If information for a field is not available in the text, use the specified default values (e.g., "N/A", 0.0, "", false).
*   **Hierarchical Structure:** Pay close attention to the hierarchy of questions and sub-questions. A main question in the `data` array can have multiple sub-questions in its `questions` array.
*   **"OR" Questions:** If a question offers a choice (e.g., "Q1. Do X OR Do Y"), the `contains_OR_questions` field for the parent question in `data` should be `true`. Both options (X and Y) should then be listed as separate sub-questions within the `questions` array, each with its own ID, text, and answer.
*   **Marks Allocation:**
    *   If a main question has sub-questions, its `marks` field should ideally be the sum of the marks of its sub-questions. If only total marks for the main question are given, try to distribute them or note it. If marks are only specified per sub-question, sum them up for the parent.
    *   If marks are not specified anywhere, use 0.0.
*   **Answer Extraction:** This is a key part. The input text contains answers. Ensure the `answer_text` field for each question/sub-question is correctly populated with its corresponding answer from the text.
*   **JSON Validity:** The final output MUST be a valid JSON. Do not include any text or explanations outside the main JSON object. Ensure all strings are properly escaped.

**Input OCR Text:**
---OCR TEXT START---
{ocr_text_of_question_paper_with_key}
---OCR TEXT END---

**Generate the structured JSON output based on the above instructions and the provided OCR text.**
"""

    full_prompt = prompt_template.format(ocr_text_of_question_paper_with_key=ocr_text_of_question_paper_with_key)

    ollama_response = send_request_to_ollama(
        prompt=full_prompt,
        base_url=ollama_base_url,
        model_name=ollama_model_name
    )

    if 'error' in ollama_response:
        return {
            "error": "Ollama API call failed for JSON generation",
            "details": ollama_response.get('error')
        }

    model_content_text = ollama_response.get('response')
    if not model_content_text or not model_content_text.strip():
        return {
            "error": "No content in Ollama response for JSON generation",
            "raw_response": ollama_response
        }

    # Basic cleanup: try to find the JSON block.
    # This looks for the first '{' and the last '}'
    try:
        json_start_index = model_content_text.index('{')
        json_end_index = model_content_text.rindex('}') + 1
        potential_json_str = model_content_text[json_start_index:json_end_index]
    except ValueError:
        potential_json_str = model_content_text  # Fallback to using the whole string if '{' or '}' not found

    try:
        parsed_output = json.loads(potential_json_str)
    except json.JSONDecodeError as e:
        # Try another cleanup: remove common markdown code block delimiters if present
        cleaned_text = re.sub(r'^```json\s*|\s*```$', '', potential_json_str.strip(), flags=re.MULTILINE)
        try:
            parsed_output = json.loads(cleaned_text)
        except json.JSONDecodeError as e2:
            return {
                "error": "Failed to parse JSON from model output after cleanup",
                "details": str(e2),
                "original_json_error": str(e),
                "raw_model_content": model_content_text
            }

    if not isinstance(parsed_output, dict):
        return {
            "error": "Parsed output is not a dictionary.",
            "raw_model_content": model_content_text,
            "parsed_type": type(parsed_output).__name__
        }

    required_keys = ["data", "exam"]
    missing_keys = [key for key in required_keys if key not in parsed_output]
    if missing_keys:
        return {
            "error": f"Parsed JSON is missing required top-level keys: {', '.join(missing_keys)}",
            "raw_model_content": model_content_text,
            "parsed_json_keys": list(parsed_output.keys())
        }

    return parsed_output


if __name__ == '__main__':
    sample_ocr_text = """
    Midterm Exam - Physics - Class X
    Total Marks: 100 Duration: 3 hours
    School: Future Scholars Academy Date: 2024-08-15
    Instructions: Attempt all questions.

    Section A
    Q1. What is the speed of light? (5 marks)
    Answer: The speed of light is 3 x 10^8 m/s.

    Q2. Define force. (5 marks)
    Answer: Force is a push or pull upon an object resulting from the object's interaction with another object.

    Section B
    Q3. Explain Newton's first law of motion. (10 marks)
    Answer: Newton's first law states that an object will remain at rest or in uniform motion in a straight line unless acted upon by an external force.

    Q3a. Give an example of inertia. (5 marks)
    Answer: A person standing in a bus tends to fall backward when the bus suddenly starts.
    (This should be under Q3's questions array)
    """

    # This URL is for local testing and won't work in the tool's environment
    # without a running Ollama instance accessible to it.
    sample_ollama_url = "http://localhost:11434"
    sample_model = "gemma3:12b"  # Example model

    print(f"--- Attempting to generate Question Paper JSON (Example) ---")
    print(f"Using Ollama URL: {sample_ollama_url} (Note: This will likely fail if Ollama is not running or accessible)")

    # Simulate a call, expecting it to fail gracefully if Ollama isn't available
    # or return a structured error from the function itself.
    generated_json = generate_question_paper_json_from_text(
        ocr_text_of_question_paper_with_key=sample_ocr_text,
        ollama_base_url=sample_ollama_url,
        ollama_model_name=sample_model
    )

    print("\n--- Result from generate_question_paper_json_from_text ---")
    if "error" in generated_json:
        print(f"Error: {generated_json['error']}")
        if "details" in generated_json:
            print(f"Details: {generated_json['details']}")
        if "raw_model_content" in generated_json:
            print(
                f"Raw Model Content (if available for debugging): {generated_json['raw_model_content'][:500]}...")  # Print first 500 chars
    else:
        print("Successfully generated JSON (or received a structured response from LLM to be parsed as JSON):")
        print(json.dumps(generated_json, indent=4))

    print("\n--- Note for testing ---")
    print("The above example call in `if __name__ == '__main__':` is illustrative.")
    print("For actual generation, ensure your Ollama instance is running and accessible at the specified URL,")
    print("and the model is available. The function is designed to call out to this service.")
    print("If the Ollama service is not reachable, an error related to the HTTP request will be shown.")


