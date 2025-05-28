from PIL import Image
import pytesseract
import base64
import io

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image file using Tesseract OCR.

    Args:
        image_path: The path to the image file.

    Returns:
        A string containing the extracted text, or an empty string
        if an error occurs (e.g., file not found, Tesseract not installed,
        invalid image format).
    """
    try:
        # Attempt to open the image file
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
        return ""
    except Exception as e:  # Catches other PIL errors like UnidentifiedImageError
        print(f"Error: Could not open or read image file '{image_path}'. It might be corrupted or an unsupported format. Details: {e}")
        return ""

    try:
        # Attempt to extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not found in your PATH.")
        print("Please install Tesseract OCR and ensure it's added to your system's PATH.")
        print("For more information, see: https://github.com/tesseract-ocr/tesseract")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during OCR processing for '{image_path}': {e}")
        return ""
    
    return extracted_text

def load_image_as_base64(image_path: str) -> str | None:
    """
    Loads an image file, converts it to PNG in memory,
    and returns it as a base64 encoded string.

    Args:
        image_path: The path to the image file.

    Returns:
        A base64 encoded string of the image in PNG format,
        or None if an error occurs.
    """
    try:
        img = Image.open(image_path)
        
        # Convert to PNG in memory
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        
        base64_encoded_string = base64.b64encode(img_bytes).decode('utf-8')
        return base64_encoded_string
        
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
        return None
    except Exception as e:  # Catches other PIL errors like UnidentifiedImageError and potential issues during save/encode
        print(f"Error: Could not process image file '{image_path}'. It might be corrupted, an unsupported format, or another issue occurred. Details: {e}")
        return None

if __name__ == '__main__':
    # Example usage (optional, for testing purposes)
    # This requires an image file (e.g., 'test_image.png') in the same directory.
    # You might need to create a dummy image or use an actual one.
    
    # Create a dummy png for testing
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            img = Image.new('RGB', (400, 100), color = (255, 255, 255))
            d = ImageDraw.Draw(img)
            # Attempt to load a font, fall back to a default if not found
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            d.text((10,10), "Hello World\nThis is a test image.", fill=(0,0,0), font=font)
            img.save("test_image.png")

            # Test with the created image
            text_from_image = extract_text_from_image('test_image.png')
            print(f"Text extracted from 'test_image.png':\n---\n{text_from_image}\n---")

            # Test with a non-existent file
            text_non_existent = extract_text_from_image('non_existent_image.png')
            print(f"Text from non_existent_image.png: '{text_non_existent}'") # Expected: Error message and empty string

            # Test with a file that is not an image (e.g. a text file)
            not_an_image_file = "not_an_image.txt"
            with open(not_an_image_file, "w") as f:
                f.write("This is a text file, not an image.")
            text_not_image = extract_text_from_image(not_an_image_file)
            print(f"Text from {not_an_image_file}: '{text_not_image}'") # Expected: Error message and empty string

            print("\n--- Testing load_image_as_base64 ---")
            # Test with the created image
            base64_str = load_image_as_base64('test_image.png')
            if base64_str:
                print(f"Base64 string from 'test_image.png' (first 30 chars): {base64_str[:30]}...")
            else:
                print("Failed to load 'test_image.png' as base64.")

            # Test with a non-existent file
            base64_non_existent = load_image_as_base64('non_existent_image.png')
            if base64_non_existent is None:
                print("Correctly handled non_existent_image.png (returned None).")
            else:
                print("Error: non_existent_image.png test failed.")
            
            # Test with a file that is not an image
            base64_not_image = load_image_as_base64(not_an_image_file)
            if base64_not_image is None:
                print(f"Correctly handled {not_an_image_file} (returned None).")
            else:
                print(f"Error: {not_an_image_file} test failed.")

        finally:
            # Clean up dummy files
            import os
            if os.path.exists("test_image.png"):
                os.remove("test_image.png")
            if os.path.exists(not_an_image_file):
                os.remove(not_an_image_file)
                
    except ImportError:
        print("Pillow (PIL) is not installed. This example requires Pillow to create a test image.")
    except Exception as e:
        print(f"An error occurred in the example usage block: {e}")
