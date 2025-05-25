import unittest
import os
# Adjust import if src is not directly in PYTHONPATH or to reflect project structure
try:
    from src.pdf_processing import extract_text_from_pdf
except ImportError:
    # This allows running tests directly from the 'tests' directory or project root
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.pdf_processing import extract_text_from_pdf

class TestPdfProcessing(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Determine the base directory (project root)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.sample_pdf_path = os.path.join(self.base_dir, 'data', 'sample_student_sheet.pdf')
        self.non_existent_pdf_path = os.path.join(self.base_dir, 'data', 'non_existent.pdf')

        # Create a dummy PDF for testing if it doesn't exist, to ensure the test can run
        # In a real scenario, this sample PDF should be part of the repo.
        # For this exercise, we'll assume it exists or create a very simple one.
        if not os.path.exists(self.sample_pdf_path):
            # Try to create a dummy one if it's missing (basic PDF structure)
            os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
            try:
                import fitz # PyMuPDF
                doc = fitz.open()
                page = doc.new_page()
                page.insert_text((50, 72), "This is a sample PDF for testing.")
                doc.save(self.sample_pdf_path)
                doc.close()
                print(f"Created dummy PDF: {self.sample_pdf_path}")
            except Exception as e:
                print(f"Could not create dummy PDF for testing: {e}. Please ensure sample_student_sheet.pdf exists in data/ directory.")


    def test_extract_text_sample_pdf(self):
        """Test text extraction from a sample PDF."""
        if not os.path.exists(self.sample_pdf_path):
            self.skipTest(f"Sample PDF not found at {self.sample_pdf_path}, skipping test.")
            return

        extracted_text = extract_text_from_pdf(self.sample_pdf_path)
        self.assertIsNotNone(extracted_text, "Extracted text should not be None.")
        self.assertIsInstance(extracted_text, str, "Extracted text should be a string.")
        self.assertTrue(len(extracted_text) > 0, "Extracted text should not be empty.")
        # A more specific check, if the content of sample_student_sheet.pdf is known and stable:
        # For the dummy PDF created in setUp:
        self.assertIn("This is a sample PDF for testing.", extracted_text, 
                      "Extracted text should contain known snippet.")

    def test_extract_text_non_existent_pdf(self):
        """Test text extraction from a non-existent PDF."""
        # Ensure the file truly doesn't exist before testing
        if os.path.exists(self.non_existent_pdf_path):
            os.remove(self.non_existent_pdf_path)
            
        extracted_text = extract_text_from_pdf(self.non_existent_pdf_path)
        self.assertEqual(extracted_text, "", 
                         "Function should return an empty string for non-existent files.")

    def test_extract_text_corrupted_pdf(self):
        """Test text extraction from a corrupted or invalid PDF."""
        corrupted_pdf_path = os.path.join(self.base_dir, 'data', 'corrupted.pdf')
        with open(corrupted_pdf_path, 'w') as f:
            f.write("This is not a valid PDF content.")
        
        extracted_text = extract_text_from_pdf(corrupted_pdf_path)
        self.assertEqual(extracted_text, "", 
                         "Function should return an empty string for corrupted PDF files.")
        
        os.remove(corrupted_pdf_path) # Clean up

if __name__ == '__main__':
    unittest.main()
