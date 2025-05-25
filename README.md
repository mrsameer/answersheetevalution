# AI-Powered Answer Sheet Grader

## Overview

This project is an AI-powered command-line application designed to automate the grading of student answer sheets. It leverages the capabilities of Google's Gemini language model to compare student-provided answers against a predefined answer key and generate a score along with qualitative feedback.

The system supports processing student answer sheets provided as PDF or common image files (PNG, JPG/JPEG). The answer key can be supplied as a PDF, an image file, or a plain TXT file.

## Features

*   **Text Extraction:** Extracts text from PDF documents and various image formats (PNG, JPG, JPEG) using Optical Character Recognition (OCR).
*   **AI-Driven Evaluation:** Utilizes the Gemini Pro model for intelligent comparison of student answers with the answer key, providing a percentage score and textual feedback.
*   **Flexible Input:** Supports multiple file formats for both student sheets and answer keys.
*   **Command-Line Interface:** Easy to use via a command-line interface for processing and receiving results.
*   **Modular Design:** Code is structured into modules for PDF processing, image processing, and AI scoring.
*   **Test Suite:** Includes unit tests for verifying the functionality of different components.

## Setup and Installation

### Prerequisites

1.  **Python:** Python 3.7 or newer is required. You can download it from [python.org](https://www.python.org/downloads/).
2.  **Tesseract-OCR:** This is essential for extracting text from image files.
    *   Installation instructions can be found on the official Tesseract documentation: [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html). Please follow the instructions specific to your operating system.
    *   After installation, ensure that the Tesseract executable (`tesseract` or `tesseract.exe`) is in your system's PATH. For some systems or Python environments, you might need to configure the path to the Tesseract executable within your Python script or environment if `pytesseract` cannot find it automatically. Refer to the `pytesseract` documentation for more details if you encounter issues. The `requirements.txt` file includes a comment with an example of how to set this path in Python if needed.

### Dependencies

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_name> 
    ```
    (Replace `<repository_url>` and `<repository_name>` with the actual URL and project directory name)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Install Python dependencies:**
    Navigate to the project's root directory (where `requirements.txt` is located) and run:
    ```bash
    pip install -r requirements.txt
    ```

### Gemini API Key

1.  **Obtain an API Key:** This project requires a Gemini API key to interact with the language model. You can obtain one from Google AI Studio:
    *   Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    *   Sign in with your Google account and create an API key.
2.  **Usage:** The API key is passed to the script as a command-line argument when you run the agent. **Do not hardcode your API key directly into the script.**

## How to Run

The script is run from the command line, providing paths to the student's answer sheet, the answer key, and your Gemini API key.

**Command Syntax:**

```bash
python src/agent.py <student_sheet_path> <answer_key_path> <your_gemini_api_key>
```

**Arguments:**

*   `student_sheet_path`: The file path to the student's answer sheet.
    *   Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`.
*   `answer_key_path`: The file path to the answer key.
    *   Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`.
*   `your_gemini_api_key`: Your personal Gemini API key.

**Examples using sample files from the `data/` directory:**

1.  **Using a PDF student sheet and a TXT answer key:**
    ```bash
    python src/agent.py data/sample_student_sheet.pdf data/sample_answer_key.txt YOUR_GEMINI_API_KEY
    ```

2.  **Using a PNG student sheet and a TXT answer key:**
    ```bash
    python src/agent.py data/sample_student_sheet.png data/sample_answer_key.txt YOUR_GEMINI_API_KEY
    ```

Replace `YOUR_GEMINI_API_KEY` with your actual Gemini API key.

**Expected Output:**

The script will print the processing steps to the console, followed by the final grading result:

```
Processing student answer sheet...
Extracting text from PDF: data/sample_student_sheet.pdf
Processing answer key...
Reading text from TXT file: data/sample_answer_key.txt
Scoring answers...

--- Results ---
Score: XX% 
Feedback:
[Qualitative feedback from the Gemini model]
```
The score and feedback will vary based on the content of the files and the model's evaluation.

## File Structure

*   `src/`: Contains the core Python modules for the application.
    *   `agent.py`: The main command-line interface script.
    *   `pdf_processing.py`: Handles text extraction from PDF files.
    *   `image_processing.py`: Handles text extraction from image files.
    *   `scorer.py`: Interacts with the Gemini API to score answers.
*   `data/`: Includes sample files for testing and demonstration.
    *   `sample_student_sheet.pdf`: A sample PDF answer sheet.
    *   `sample_student_sheet.png`: A sample image answer sheet.
    *   `sample_answer_key.txt`: A sample text-based answer key.
*   `tests/`: Contains unit tests for the project.
    *   `test_*.py`: Individual test files for each module.
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `README.md`: This file.

## Limitations

*   **OCR Accuracy:** The accuracy of text extraction from images (and some PDFs) depends heavily on the input quality (e.g., clarity of text, font style, image resolution, layout complexity). Handwriting is generally not well supported by the current OCR setup.
*   **Gemini Model Performance:** The quality of the scoring and feedback is dependent on the Gemini model's understanding of the provided text, the clarity of the student's answers, the comprehensiveness of the answer key, and the effectiveness of the internal prompt used.
*   **Answer Key Format:** While text-based answer keys (`.txt`) are most reliable, PDF and image keys are subject to OCR accuracy. For best results, answer keys should have clear, machine-readable text.
*   **No Graphical User Interface (GUI):** This is a command-line tool.
*   **API Costs:** Use of the Google Gemini API may incur costs depending on your usage and Google's pricing model. Please refer to Google's API pricing documentation.
*   **Error Handling:** While basic error handling is implemented (e.g., file not found, API issues), there might be edge cases not fully covered.

## Running Tests

The project includes a suite of unit tests to verify the functionality of its components.

To run all tests, navigate to the project root directory and use the `unittest` discovery feature:

```bash
python -m unittest discover -s tests
```

Alternatively, you can run individual test files:

```bash
python tests/test_pdf_processing.py
python tests/test_image_processing.py
python tests/test_scorer.py
python tests/test_agent.py
```

Ensure that you have installed all dependencies from `requirements.txt` before running tests, as some tests might rely on these libraries (e.g., `PyMuPDF` for creating dummy PDFs in tests).
