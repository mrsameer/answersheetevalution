import fitz  # PyMuPDF

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

if __name__ == '__main__':
    # Example usage (optional, for testing purposes)
    # Create a dummy PDF for testing if you don't have one readily available.
    # This part requires a PDF file named 'dummy.pdf' in the same directory.
    # You can create one or replace 'dummy.pdf' with an actual PDF path.
    try:
        with open("dummy.pdf", "wb") as f: # Create a dummy pdf for testing
            f.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF")
        print(f"Content of dummy.pdf: {extract_text_from_pdf('dummy.pdf')}")
        text = extract_text_from_pdf('non_existent_file.pdf')
        print(f"Text from non_existent_file.pdf: '{text}'") # Expected: Error message and empty string
        
        # Test with a potentially corrupted or invalid PDF file (e.g., an empty file or a text file)
        with open("corrupted.pdf", "w") as f:
            f.write("This is not a PDF.")
        text_corrupted = extract_text_from_pdf('corrupted.pdf')
        print(f"Text from corrupted.pdf: '{text_corrupted}'") # Expected: Error message and empty string

    finally:
        # Clean up dummy files
        import os
        if os.path.exists("dummy.pdf"):
            os.remove("dummy.pdf")
        if os.path.exists("corrupted.pdf"):
            os.remove("corrupted.pdf")
