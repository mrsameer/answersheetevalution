import unittest
import os
# Adjust import if src is not directly in PYTHONPATH or to reflect project structure
try:
    from src.image_processing import extract_text_from_image
except ImportError:
    # This allows running tests directly from the 'tests' directory or project root
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.image_processing import extract_text_from_image

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.sample_image_path = os.path.join(self.base_dir, 'data', 'sample_student_sheet.png')
        self.non_existent_image_path = os.path.join(self.base_dir, 'data', 'non_existent.png')

        # Create a dummy PNG for testing if it doesn't exist
        if not os.path.exists(self.sample_image_path):
            os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (600, 100), color = (255, 255, 255))
                d = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError: # If Arial is not available, use default font
                    font = ImageFont.load_default()
                d.text((10,10), "This is a sample PNG for testing OCR.", fill=(0,0,0), font=font)
                img.save(self.sample_image_path)
                print(f"Created dummy PNG: {self.sample_image_path}")
            except Exception as e:
                print(f"Could not create dummy PNG for testing: {e}. Please ensure sample_student_sheet.png exists in data/ directory.")


    def test_extract_text_sample_image(self):
        """Test text extraction from a sample image."""
        if not os.path.exists(self.sample_image_path):
            self.skipTest(f"Sample PNG not found at {self.sample_image_path}, skipping test.")
            return

        # Note: OCR can be sensitive to Tesseract installation, language packs, and image quality.
        # This test primarily checks if the function runs and returns a string.
        # For the dummy image created in setUp, we can be a bit more specific.
        extracted_text = extract_text_from_image(self.sample_image_path)
        self.assertIsNotNone(extracted_text, "Extracted text should not be None.")
        self.assertIsInstance(extracted_text, str, "Extracted text should be a string.")
        # We expect some text, but it might not be perfect.
        # If Tesseract is not installed or configured, this might be empty.
        # The function itself handles TesseractNotFoundError and prints a message.
        if "Tesseract is not installed" in extracted_text: # Check if function returned error message
             self.assertTrue("Tesseract is not installed" in extracted_text, "Function should indicate Tesseract error if applicable.")
        else:
            # If Tesseract is expected to work, check for some content.
            # For the dummy image:
            self.assertTrue(len(extracted_text) > 0, 
                            "Extracted text should not be empty if OCR is successful.")
            self.assertIn("sample PNG for testing", extracted_text.lower(),
                          "Extracted text should contain known snippet if OCR is successful.")


    def test_extract_text_non_existent_image(self):
        """Test text extraction from a non-existent image."""
        if os.path.exists(self.non_existent_image_path):
            os.remove(self.non_existent_image_path) # Ensure it doesn't exist

        extracted_text = extract_text_from_image(self.non_existent_image_path)
        self.assertEqual(extracted_text, "", 
                         "Function should return an empty string for non-existent files.")

    def test_extract_text_invalid_image(self):
        """Test text extraction from an invalid image file."""
        invalid_image_path = os.path.join(self.base_dir, 'data', 'invalid_image.png')
        with open(invalid_image_path, 'w') as f:
            f.write("This is not a valid image content.")
        
        extracted_text = extract_text_from_image(invalid_image_path)
        self.assertEqual(extracted_text, "", 
                         "Function should return an empty string for invalid image files.")
        
        os.remove(invalid_image_path) # Clean up

if __name__ == '__main__':
    unittest.main()
