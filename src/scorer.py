import google.generativeai as genai
import re # For parsing the response

def score_answers(student_text: str, answer_key_text: str, api_key: str) -> dict:
    """
    Compares student answers to an answer key using the Gemini API
    and returns a score and feedback.

    Args:
        student_text: A string containing the student's answers.
        answer_key_text: A string containing the correct answers.
        api_key: The API key for the Gemini API.

    Returns:
        A dictionary containing the 'score' (int or float) and 'feedback' (str),
        or an error message if the API call or parsing fails.
    """
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        # This typically won't raise an exception, but good to be aware
        print(f"Error configuring Gemini API: {e}")
        return {"error": "API configuration failed", "details": str(e)}

    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""**Instruction:** You are an academic evaluator. Your task is to grade the following student's answer sheet based on the provided answer key.
Provide a numerical score as a percentage (0-100) and a brief qualitative feedback summary.
Format your response as:
Score: [score_value]
Feedback: [qualitative_feedback]

**Answer Key:**
---
{answer_key_text}
---

**Student's Answers:**
---
{student_text}
---

**Evaluation:**
"""

    try:
        response = model.generate_content(prompt)
    except genai.types.generation_types.BlockedPromptException as e:
        print(f"Error: The prompt was blocked by the API. Details: {e}")
        return {"error": "API call failed due to blocked prompt", "details": str(e)}
    except Exception as e: # Catches other API call errors like PermissionDenied, DeadlineExceeded, etc.
        # Check for common specific exceptions if needed, e.g., from google.api_core.exceptions
        # For instance, google.api_core.exceptions.PermissionDenied for invalid API key
        # from google.api_core import exceptions as google_exceptions
        # if isinstance(e, google_exceptions.PermissionDenied):
        # print(f"API Error: Permission denied. Please check your API key. Details: {e}")
        # return {"error": "API call failed - Permission Denied", "details": str(e)}
        print(f"Error during Gemini API call: {e}")
        return {"error": "API call failed", "details": str(e)}

    try:
        model_response_text = response.text
        # Using regex to parse the score and feedback
        score_match = re.search(r"Score:\s*(\d+\.?\d*)", model_response_text, re.IGNORECASE)
        feedback_match = re.search(r"Feedback:\s*(.*)", model_response_text, re.DOTALL | re.IGNORECASE)

        if score_match and feedback_match:
            try:
                score_value_str = score_match.group(1)
                # Try converting to float first, then to int if it's a whole number
                try:
                    score_value = float(score_value_str)
                    if score_value.is_integer():
                        score_value = int(score_value)
                except ValueError:
                    print(f"Warning: Could not convert score '{score_value_str}' to a number. Storing as string.")
                    score_value = score_value_str # Keep as string if conversion fails
            except IndexError: # Should not happen if score_match is successful
                 return {"error": "Failed to parse score from model response", "raw_response": model_response_text}
            
            qualitative_feedback = feedback_match.group(1).strip()
            return {"score": score_value, "feedback": qualitative_feedback}
        else:
            print("Error: Could not parse score and/or feedback from the model's response.")
            print(f"Raw response:\n{model_response_text}")
            return {"error": "Failed to parse model response", "raw_response": model_response_text}

    except AttributeError:
        # This can happen if response.text is not available (e.g. if response.parts is empty or not text)
        # or if the response object itself doesn't conform to expectations (e.g. if it's None or parts is empty)
        # Check for response.candidates and prompt_feedback for more details if needed
        error_details = "Unknown error parsing response."
        if response and response.prompt_feedback:
            error_details = f"Prompt feedback: {response.prompt_feedback}"
        elif response and not response.parts:
             error_details = "Response was empty."
        
        print(f"Error: Failed to extract text from model response. {error_details}")
        return {"error": "Failed to extract text from model response", "details": error_details}
    except Exception as e:
        print(f"An unexpected error occurred during response parsing: {e}")
        return {"error": "Unexpected error parsing model response", "details": str(e), "raw_response": model_response_text if 'model_response_text' in locals() else "Unavailable"}

if __name__ == '__main__':
    # This example requires a valid API_KEY environment variable to be set.
    import os
    api_key = os.environ.get("API_KEY") # Replace with your actual API key if not using env var

    if not api_key:
        print("API_KEY environment variable not set. Please set it to run this example.")
        # You could also hardcode a key here for testing, but be careful not to commit it.
        # api_key = "YOUR_API_KEY_HERE" 
    else:
        sample_student_text = """
        Question 1: What is the capital of France?
        Answer: Paris

        Question 2: What is 2 + 2?
        Answer: 4

        Question 3: Who wrote Hamlet?
        Answer: Shakespeare
        """

        sample_answer_key = """
        Question 1: The capital of France is Paris.
        Question 2: The sum of 2 + 2 is 4.
        Question 3: Hamlet was written by William Shakespeare.
        """
        
        # Test case 1: Valid input
        print("--- Test Case 1: Valid Input ---")
        result = score_answers(sample_student_text, sample_answer_key, api_key)
        print(f"Result: {result}")
        if "score" in result:
            print(f"Score Type: {type(result['score'])}")

        # Test case 2: Invalid API Key (replace with a clearly invalid key for testing)
        print("\n--- Test Case 2: Invalid API Key ---")
        invalid_api_key = "THIS_IS_NOT_A_VALID_API_KEY"
        result_invalid_key = score_answers(sample_student_text, sample_answer_key, invalid_api_key)
        print(f"Result (Invalid Key): {result_invalid_key}")

        # Test case 3: Empty student text (model might handle this, or we might want specific error)
        print("\n--- Test Case 3: Empty Student Text ---")
        result_empty_student = score_answers("", sample_answer_key, api_key)
        print(f"Result (Empty Student Text): {result_empty_student}")
        
        # Test case 4: Empty answer key (model might handle this)
        print("\n--- Test Case 4: Empty Answer Key ---")
        result_empty_key = score_answers(sample_student_text, "", api_key)
        print(f"Result (Empty Answer Key): {result_empty_key}")

        # Test case 5: Prompt that might be blocked (example, adjust if needed)
        # This is highly dependent on the safety settings of the model and API provider
        print("\n--- Test Case 5: Potentially Blocked Prompt Content ---")
        potentially_problematic_text = "This is some text that might be considered harmful or against policy."
        # Note: The Gemini API might still process this if it's not severe enough to trigger blocking.
        # This test is illustrative.
        result_blocked_prompt = score_answers(potentially_problematic_text, sample_answer_key, api_key)
        print(f"Result (Potentially Blocked Prompt): {result_blocked_prompt}")
