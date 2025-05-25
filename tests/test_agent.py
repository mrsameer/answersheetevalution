import unittest
from unittest.mock import patch, mock_open, MagicMock
import argparse
import os
import io
import sys

# Adjust import if src is not directly in PYTHONPATH or to reflect project structure
try:
    from src.agent import main as agent_main, get_text_from_file
    # If main is directly imported, other functions from agent might also be needed directly
    # or they are called via main.
except ImportError:
    # This allows running tests directly from the 'tests' directory or project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.agent import main as agent_main, get_text_from_file


class TestAgent(unittest.TestCase):

    def setUp(self):
        # Create dummy files that get_text_from_file might try to access
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.test_data_dir = os.path.join(self.base_dir, 'data')
        os.makedirs(self.test_data_dir, exist_ok=True)

        self.student_pdf_path = os.path.join(self.test_data_dir, 'student_test.pdf')
        self.key_txt_path = os.path.join(self.test_data_dir, 'key_test.txt')

        # Create simple dummy files for testing purposes
        # A minimal PDF
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50,72), "Student PDF content.")
            doc.save(self.student_pdf_path)
            doc.close()
        except Exception as e:
            # Fallback if PyMuPDF is not available to create a file,
            # get_text_from_file will be mocked anyway.
            with open(self.student_pdf_path, 'wb') as f:
                f.write(b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>>>endobj xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF")


        with open(self.key_txt_path, 'w') as f:
            f.write("Answer key content.")

    def tearDown(self):
        # Clean up dummy files
        if os.path.exists(self.student_pdf_path):
            os.remove(self.student_pdf_path)
        if os.path.exists(self.key_txt_path):
            os.remove(self.key_txt_path)
        # Attempt to remove data directory if it's empty, handle error if not.
        try:
            if os.path.isdir(self.test_data_dir) and not os.listdir(self.test_data_dir):
                 # Check if it's the original 'data' dir or a test-specific one.
                 # Be cautious if self.test_data_dir is same as 'data/' used by other tests.
                 # For this setup, it refers to 'data/', so only remove if sure it's safe.
                 # Given setUp creates it if not existing, and other tests might also use 'data/',
                 # it's safer to leave 'data/' and only remove files created by this test.
                 pass
        except OSError:
            pass # Directory not empty or other issue


    @patch('src.agent.score_answers')
    @patch('src.agent.get_text_from_file') # Patching the helper function in agent.py
    @patch('sys.stdout', new_callable=io.StringIO) # Capture stdout
    def test_agent_happy_path_pdf_and_txt(self, mock_stdout, mock_get_text, mock_score_answers):
        """Test the agent's main function with PDF and TXT inputs."""
        
        # Configure mocks for get_text_from_file
        # It will be called twice: first for student sheet, then for answer key.
        mock_get_text.side_effect = [
            "Extracted student text from PDF.", 
            "Extracted answer key text from TXT."
        ]
        
        # Configure mock for score_answers
        mock_score_answers.return_value = {"score": 90, "feedback": "Excellent work"}

        # Simulate command line arguments
        # Use paths created in setUp for student_sheet_path and answer_key_path
        # to ensure that os.path.exists passes within get_text_from_file if not fully mocked.
        # However, since get_text_from_file is fully mocked here, the actual file existence
        # for these paths within the agent's main logic (before get_text_from_file is called)
        # is not strictly necessary *for this specific test structure*.
        # But using real-like paths is good practice.
        
        test_args = [
            'src/agent.py', # Program name, part of sys.argv
            self.student_pdf_path, 
            self.key_txt_path,
            'fake_api_key123'
        ]

        with patch('sys.argv', test_args):
            agent_main()

        # Assertions
        # Check that get_text_from_file was called correctly
        self.assertEqual(mock_get_text.call_count, 2)
        mock_get_text.assert_any_call(self.student_pdf_path)
        mock_get_text.assert_any_call(self.key_txt_path)
        
        # Check that score_answers was called with the text returned by the mocked get_text_from_file
        mock_score_answers.assert_called_once_with(
            "Extracted student text from PDF.",
            "Extracted answer key text from TXT.",
            'fake_api_key123'
        )

        # Check stdout for results
        output = mock_stdout.getvalue()
        self.assertIn("Score: 90%", output)
        self.assertIn("Feedback:\nExcellent work", output)


    @patch('src.agent.get_text_from_file')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_agent_file_extraction_fails_student(self, mock_stdout, mock_get_text):
        """Test agent behavior when student sheet extraction fails."""
        mock_get_text.return_value = None # Simulate extraction failure

        test_args = ['src/agent.py', 'dummy_student.pdf', 'dummy_key.txt', 'key']
        with patch('sys.argv', test_args):
            agent_main()
        
        output = mock_stdout.getvalue()
        self.assertIn("Could not extract text from student sheet", output)
        self.assertIn("Exiting.", output)

    @patch('src.agent.score_answers')
    @patch('src.agent.get_text_from_file')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_agent_scoring_api_error(self, mock_stdout, mock_get_text, mock_score_answers):
        """Test agent behavior when score_answers returns an API error."""
        mock_get_text.side_effect = ["student content", "key content"]
        mock_score_answers.return_value = {"error": "API Error", "details": "Connection failed"}

        test_args = ['src/agent.py', 's.pdf', 'k.txt', 'key']
        with patch('sys.argv', test_args):
            agent_main()

        output = mock_stdout.getvalue()
        self.assertIn("Scoring failed: API Error", output)
        self.assertIn("Details: Connection failed", output)

    # Test for get_text_from_file utility function (part of agent.py)
    @patch('src.agent.extract_text_from_pdf')
    def test_get_text_from_file_pdf_calls_extract_pdf(self, mock_extract_pdf):
        mock_extract_pdf.return_value = "pdf text"
        # Create a dummy PDF file for this specific test
        test_file_path = os.path.join(self.test_data_dir, "test.pdf")
        with open(test_file_path, "w") as f: # Simple text file, but with .pdf extension
            f.write("dummy pdf content for os.path.exists")

        result = get_text_from_file(test_file_path)
        mock_extract_pdf.assert_called_once_with(test_file_path)
        self.assertEqual(result, "pdf text")
        os.remove(test_file_path)

    @patch('src.agent.extract_text_from_image')
    def test_get_text_from_file_png_calls_extract_image(self, mock_extract_image):
        mock_extract_image.return_value = "image text"
        test_file_path = os.path.join(self.test_data_dir, "test.png")
        with open(test_file_path, "w") as f:
             f.write("dummy image content for os.path.exists") # Not a real PNG

        result = get_text_from_file(test_file_path)
        mock_extract_image.assert_called_once_with(test_file_path)
        self.assertEqual(result, "image text")
        os.remove(test_file_path)

    def test_get_text_from_file_txt_reads_correctly(self):
        test_file_path = os.path.join(self.test_data_dir, "test.txt")
        expected_content = "Hello from the text file."
        with open(test_file_path, "w", encoding='utf-8') as f:
            f.write(expected_content)
        
        result = get_text_from_file(test_file_path)
        self.assertEqual(result, expected_content)
        os.remove(test_file_path)

    def test_get_text_from_file_unsupported_extension(self):
        test_file_path = os.path.join(self.test_data_dir, "test.doc")
        with open(test_file_path, "w") as f:
            f.write("Some Word document content.")
        # Capture print output for this specific call if desired
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            result = get_text_from_file(test_file_path)
            self.assertIsNone(result)
            self.assertIn("Unsupported file type '.doc'", mock_stdout.getvalue())
        os.remove(test_file_path)

    def test_get_text_from_file_not_found(self):
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            result = get_text_from_file("non_existent_file.pdf")
            self.assertIsNone(result)
            self.assertIn("File not found", mock_stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
