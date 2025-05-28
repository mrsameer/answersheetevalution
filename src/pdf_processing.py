import fitz  # PyMuPDF
import base64

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A string containing all text extracted from the PDF,
        or an empty string if an error occurs.
    """
    extracted_text = ""
    try:
        # Attempt to open the PDF document
        with fitz.open(pdf_path) as doc:
            # Iterate through each page in the document
            for page in doc:
                # Extract text from the current page
                extracted_text += page.get_text()
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        return ""
    except fitz.fitz.RuntimeException as e:
        print(f"Error: Could not open or read PDF '{pdf_path}'. It might be corrupted or not a valid PDF. Details: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while processing '{pdf_path}': {e}")
        return ""
    return extracted_text

def convert_pdf_to_images_base64(pdf_path: str) -> list[str]:
    """
    Converts each page of a PDF file into a list of base64 encoded image strings.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A list of base64 encoded PNG image strings,
        or an empty list if an error occurs.
    """
    base64_images = []
    try:
        # Attempt to open the PDF document
        with fitz.open(pdf_path) as doc:
            # Iterate through each page in the document
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render page to a pixmap (image)
                pix = page.get_pixmap()
                # Convert pixmap to PNG image bytes
                img_bytes = pix.tobytes("png")
                # Encode image bytes to base64 string
                base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(base64_encoded_image)
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        return []
    except fitz.fitz.RuntimeException as e:
        print(f"Error: Could not open or read PDF '{pdf_path}'. It might be corrupted or not a valid PDF. Details: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while converting PDF to images '{pdf_path}': {e}")
        return []
    return base64_images

if __name__ == '__main__':
    # Example usage (optional, for testing purposes)
    # Create a dummy PDF for testing if you don't have one readily available.
    # This part requires a PDF file named 'dummy.pdf' in the same directory.
    # You can create one or replace 'dummy.pdf' with an actual PDF path.
    dummy_pdf_path = "dummy.pdf"
    corrupted_pdf_path = "corrupted.pdf"
    non_existent_pdf_path = "non_existent_file.pdf"

    try:
        # Create a dummy PDF for testing
        with open(dummy_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF")
        
        print("--- Testing extract_text_from_pdf ---")
        print(f"Content of {dummy_pdf_path}: {extract_text_from_pdf(dummy_pdf_path)}")
        text_non_existent = extract_text_from_pdf(non_existent_pdf_path)
        print(f"Text from {non_existent_pdf_path}: '{text_non_existent}'")
        
        with open(corrupted_pdf_path, "w") as f:
            f.write("This is not a PDF.")
        text_corrupted = extract_text_from_pdf(corrupted_pdf_path)
        print(f"Text from {corrupted_pdf_path}: '{text_corrupted}'")

        print("\n--- Testing convert_pdf_to_images_base64 ---")
        base64_images_dummy = convert_pdf_to_images_base64(dummy_pdf_path)
        print(f"Number of images from {dummy_pdf_path}: {len(base64_images_dummy)}")
        if base64_images_dummy:
            print(f"First image's (first 30 chars): {base64_images_dummy[0][:30]}...")

        base64_images_non_existent = convert_pdf_to_images_base64(non_existent_pdf_path)
        print(f"Number of images from {non_existent_pdf_path}: {len(base64_images_non_existent)}")

        base64_images_corrupted = convert_pdf_to_images_base64(corrupted_pdf_path)
        print(f"Number of images from {corrupted_pdf_path}: {len(base64_images_corrupted)}")

    finally:
        # Clean up dummy files
        import os
        if os.path.exists(dummy_pdf_path):
            os.remove(dummy_pdf_path)
        if os.path.exists(corrupted_pdf_path):
            os.remove(corrupted_pdf_path)
