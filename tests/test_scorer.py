import unittest
from unittest.mock import patch, MagicMock
import os
# Adjust import if src is not directly in PYTHONPATH or to reflect project structure
try:
    from src.scorer import score_answers
except ImportError:
    # This allows running tests directly from the 'tests' directory or project root
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.scorer import score_answers


class TestScorer(unittest.TestCase):

    @patch('src.scorer.genai.GenerativeModel')
    def test_score_answers_success(self, mock_generative_model):
        """Test successful scoring and parsing."""
        # Configure the mock model instance
        mock_model_instance = MagicMock()
        # Configure the generate_content method of the instance
        mock_response = MagicMock()
        mock_response.text = "Score: 85\nFeedback: Good effort."
        mock_model_instance.generate_content.return_value = mock_response
        # The patch decorates the class, so it returns the mock class itself
        mock_generative_model.return_value = mock_model_instance

        student_text = "This is the student's answer."
        answer_key_text = "This is the answer key."
        api_key = "fake_api_key"

        expected_result = {"score": 85, "feedback": "Good effort."}
        actual_result = score_answers(student_text, answer_key_text, api_key)
        
        self.assertEqual(actual_result, expected_result)
        mock_model_instance.generate_content.assert_called_once()

    @patch('src.scorer.genai.GenerativeModel')
    def test_score_answers_success_float_score(self, mock_generative_model):
        """Test successful scoring with a float score."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Score: 75.5\nFeedback: Decent attempt."
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        expected_result = {"score": 75.5, "feedback": "Decent attempt."}
        actual_result = score_answers("student text", "key text", "fake_api_key")
        
        self.assertEqual(actual_result, expected_result)

    @patch('src.scorer.genai.GenerativeModel')
    def test_score_answers_api_error(self, mock_generative_model):
        """Test handling of API errors during content generation."""
        mock_model_instance = MagicMock()
        # Configure generate_content to raise a generic exception simulating an API error
        mock_model_instance.generate_content.side_effect = Exception("API Communication Error")
        mock_generative_model.return_value = mock_model_instance

        student_text = "Student's submission."
        answer_key_text = "Correct answers."
        api_key = "fake_api_key_for_api_error"

        result = score_answers(student_text, answer_key_text, api_key)
        
        self.assertIn("error", result, "Result dictionary should contain an 'error' key.")
        self.assertEqual(result["error"], "API call failed")
        self.assertIn("API Communication Error", result.get("details", ""), "Error details should mention the API error.")

    @patch('src.scorer.genai.GenerativeModel')
    def test_score_answers_parsing_error_malformed_score(self, mock_generative_model):
        """Test handling of parsing errors when score is malformed."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        # Malformed score: "eighty" instead of a number
        mock_response.text = "Score: eighty\nFeedback: The student tried."
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        student_text = "Some answers."
        answer_key_text = "Some key."
        api_key = "fake_api_key_for_parsing_error"

        result = score_answers(student_text, answer_key_text, api_key)
        
        self.assertIn("score", result) # score_answers now attempts to store string score
        self.assertEqual(result["score"], "eighty")
        self.assertEqual(result["feedback"], "The student tried.")
        # The current implementation of score_answers tries to convert score to float/int,
        # but if it fails, it prints a warning and stores the score as a string.
        # No "error" key is returned in this specific scenario by the current code.
        # If strict error on parsing is desired, the scorer.py would need modification.

    @patch('src.scorer.genai.GenerativeModel')
    def test_score_answers_parsing_error_missing_feedback(self, mock_generative_model):
        """Test handling of parsing errors when feedback is missing."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Score: 90" # Feedback part is missing
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        result = score_answers("student", "key", "fake_api_key")

        self.assertIn("error", result, "Result should contain an 'error' key for missing feedback.")
        self.assertEqual(result["error"], "Failed to parse model response")
        self.assertIn("Score: 90", result.get("raw_response", ""), "Raw response should be included in error details.")

    @patch('src.scorer.genai.GenerativeModel')
    def test_score_answers_empty_response_text(self, mock_generative_model):
        """Test handling when the model returns an empty or non-text response."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "" # Empty text
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        result = score_answers("student", "key", "fake_api_key")

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to parse model response")

    @patch('src.scorer.genai.configure') # Mock configure to avoid actual API key actions
    @patch('src.scorer.genai.GenerativeModel')
    def test_score_answers_blocked_prompt(self, mock_generative_model, mock_configure):
        """Test handling of a blocked prompt exception from the API."""
        mock_model_instance = MagicMock()
        # Simulate a BlockedPromptException
        # Need to import it if it's a specific exception type from the library
        # For now, using a general Exception and checking message might be enough
        # or using the actual exception if available in the test environment
        # from google.generativeai.types import BlockedPromptException (if available)
        # mock_model_instance.generate_content.side_effect = BlockedPromptException("Prompt blocked")
        # Using a general exception and checking type/message in scorer.py is an alternative
        
        # To mock specific exception from genai.types
        class MockBlockedPromptException(Exception):
            pass

        # Ensure the 'types' submodule and 'generation_types' are correctly mocked if they exist
        # This can get complex if the library structure is deep.
        # For this example, let's assume BlockedPromptException is accessible for mocking.
        # If not, the scorer.py should catch a more general API error.
        # We'll assume the scorer.py catches `genai.types.generation_types.BlockedPromptException`
        
        # The scorer.py code uses 'genai.types.generation_types.BlockedPromptException'
        # We need to make sure this path is mockable.
        # One way is to ensure 'genai.types.generation_types' exists in the mocked 'genai'
        with patch('src.scorer.genai.types.generation_types.BlockedPromptException', create=True) as mock_blocked_exception:
            mock_blocked_exception.side_effect = Exception("Simulated Blocked Prompt") # make the side_effect the exception itself
            mock_model_instance.generate_content.side_effect = mock_blocked_exception("Prompt was blocked by safety settings.")
            mock_generative_model.return_value = mock_model_instance

            result = score_answers("problematic student text", "key text", "fake_api_key")

            self.assertIn("error", result)
            self.assertEqual(result["error"], "API call failed due to blocked prompt")
            self.assertIn("Prompt was blocked", result.get("details", ""))

if __name__ == '__main__':
    unittest.main()
