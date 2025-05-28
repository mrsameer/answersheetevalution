# AI-Powered Answer Sheet Processing

## Overview

This project provides command-line tools for AI-powered processing of student answer sheets. It currently offers two main functionalities:

1.  **Ollama-based Answer Extraction:** This feature uses a local or remote Ollama instance in a two-stage process:
    *   First, it takes a "question paper with answers (key)" file (PDF, image, or TXT), performs OCR, and uses an LLM to convert this text into a structured JSON representation of the question paper.
    *   Second, using this generated JSON question paper, it extracts structured answers and metadata from the student's answer script (PDF, image, or TXT).
    *   The final output is a JSON object mapping extracted answers to question numbers and including metadata.
2.  **Gemini-based Automated Grading:** This original feature leverages Google's Gemini language model to compare student-provided answers against a predefined answer key and generate a score along with qualitative feedback.

The system supports processing student answer sheets provided as PDF or common image files (PNG, JPG/JPEG). The answer key for the Gemini-based grading can be supplied as a PDF, an image file, or a plain TXT file.

## Features

*   **Text Extraction:** Extracts text from PDF documents and various image formats (PNG, JPG, JPEG) using Optical Character Recognition (OCR).
*   **Ollama-based Answer Extraction:**
    *   **Two-Stage LLM Process:**
        1.  Processes a "question paper with answers (key)" file (PDF, image, TXT) by performing OCR and then using an Ollama model to convert the OCR text into a structured JSON representation of the question paper (including questions, answers, marks, etc.).
        2.  Uses this LLM-generated JSON question paper to guide a second Ollama model call that extracts answers and metadata from the student's answer script.
    *   Outputs a structured JSON object mapping extracted answers to question numbers and including metadata.
*   **AI-Driven Evaluation (Gemini):** Utilizes the Gemini Pro model for intelligent comparison of student answers with the answer key, providing a percentage score and textual feedback.
*   **Flexible Input:** Supports multiple file formats for student sheets, question papers, and answer keys.
*   **Command-Line Interface:** Easy to use via a command-line interface for processing and receiving results.
*   **Modular Design:** Code is structured into modules for PDF processing, image processing, AI interaction (Ollama and Gemini), and answer extraction.
*   **Test Suite:** Includes unit tests for verifying the functionality of different components.

## Setup and Installation

### Prerequisites

1.  **Python:** Python 3.7 or newer is required. You can download it from [python.org](https://www.python.org/downloads/).
2.  **Ollama (for answer extraction feature):** If using the answer extraction feature, you need a running Ollama instance accessible at the URL you provide. Refer to [Ollama.ai](https://ollama.ai/) for installation instructions.
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

### API Keys and Endpoints

*   **Ollama API Endpoint:** For the answer extraction feature, you'll need the base URL of your running Ollama instance (e.g., `http://localhost:11434`). This is passed via the `--ollama-url` argument.
*   **Gemini API Key (for grading feature):**
    1.  **Obtain an API Key:** The Gemini-based grading feature requires a Gemini API key. You can obtain one from Google AI Studio:
        *   Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
        *   Sign in with your Google account and create an API key.
    2.  **Usage:** The API key is passed to the script as a command-line argument when you run the grading agent. **Do not hardcode your API key directly into the script.**

## How to Run

The `src/agent.py` script serves as the entry point for both functionalities, determined by the provided arguments.

### Running the Ollama Answer Extractor

This mode uses a two-stage LLM process. First, it generates a structured JSON representation of the question paper from an input "question paper with answers (key)" file. Second, it uses this generated JSON to extract answers from the student's script.

**Command Syntax:**

```bash
python src/agent.py <student_sheet_path> <question_paper_path> --ollama-url <ollama_api_base_url> [--ollama-model <model_name>]
```

**Arguments:**

*   `student_sheet_path`: Path to the student's answer sheet (PDF, PNG, JPG, JPEG, or TXT for OCR).
*   `question_paper_path`: Path to the "question paper with answers (key)" file. This file can be a PDF, image (PNG, JPG, JPEG), or TXT file. It will be OCR'd, and its text content will be used by an LLM to generate a structured JSON representation of the question paper (including questions, answers, marks, etc.). This generated JSON is then used in the subsequent answer extraction step.
*   `--ollama-url`: (Required) Base URL for the Ollama API (e.g., `http://localhost:11434`).
*   `--ollama-model`: (Optional) Name of the Ollama model to use for both stages (default: `gemma3:12b`).

**Example Command:**

*(Note: You'll need to have a `sample_question_paper_with_key.pdf` or similar file in the `data/` directory for this example. This file should contain the questions and their corresponding correct answers, as it will be used to generate the structured question paper.)*
```bash
python src/agent.py data/sample_student_sheet.pdf data/sample_question_paper_with_key.pdf --ollama-url http://localhost:11434
```

**Expected Output (Ollama Answer Extractor):**

The script will print processing steps for both stages (Question Paper JSON generation and Answer Extraction) and then output a final JSON structure containing the extracted answers from the student's sheet. An example snippet of the final output:

```json
{
    "data": [
        {
            "id": "1-e125a4de-fcce-4988-9a88-9b32d89ebca3",
            "question_number": "Q1",
            "answers": [
                {
                    "id": "2-e125a4de-fcce-4988-9a88-9b32d89ebca3",
                    "student_answer_text": "in"
                }
            ]
        }
    ],
    "exam": [
        {
            "school_name": "",
            "name": "Avantilka"
        }
    ]
}
```

### Running the Gemini Answer Grader

This mode scores a student's answer sheet against an answer key using the Gemini API.

**Command Syntax:**

```bash
python src/agent.py <student_sheet_path> <answer_key_path> <your_gemini_api_key>
```

**Arguments:**

*   `student_sheet_path`: The file path to the student's answer sheet.
    *   Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`.
*   `answer_key_path`: The file path to the answer key.
    *   Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`.
*   `your_gemini_api_key`: Your personal Gemini API key.

**Examples (Gemini Grader):**

1.  **Using a PDF student sheet and a TXT answer key:**
    ```bash
    python src/agent.py data/sample_student_sheet.pdf data/sample_answer_key.txt YOUR_GEMINI_API_KEY
    ```

2.  **Using a PNG student sheet and a TXT answer key:**
    ```bash
    python src/agent.py data/sample_student_sheet.png data/sample_answer_key.txt YOUR_GEMINI_API_KEY
    ```

Replace `YOUR_GEMINI_API_KEY` with your actual Gemini API key.

**Expected Output (Gemini Grader):**

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
    *   `agent.py`: The main command-line interface script, routing to different functionalities based on arguments.
    *   `pdf_processing.py`: Handles text extraction from PDF files.
    *   `image_processing.py`: Handles text extraction from image files.
    *   `ollama_client.py`: Handles communication with the Ollama API.
    *   `answer_extractor.py`: Implements logic for extracting structured answers using Ollama.
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
*   **LLM Performance (General):** The quality of results for both answer extraction (Ollama) and grading (Gemini) depends significantly on:
    *   The capabilities of the specific LLM used (e.g., `gemma3:12b` for Ollama, Gemini Pro).
    *   The clarity and quality of the OCR'd student script.
    *   For the Ollama feature:
        *   The clarity and quality of the OCR'd "question paper with answers (key)" file.
        *   The LLM's ability to accurately convert this OCR text into the required structured JSON format.
        *   The quality of the LLM-generated question paper JSON.
    *   For the Gemini feature: The detail and correctness of the provided answer key.
    *   The effectiveness of the prompts used internally for all LLM calls.
*   **Ollama Setup:** The Ollama-based extraction requires a correctly configured and accessible Ollama instance. Performance can vary based on the model used and the hardware running Ollama.
*   **Multi-Stage LLM Dependency (Ollama):** The success of the Ollama answer extraction is now dependent on two sequential LLM calls:
    1.  Generation of the question paper JSON from the provided key file.
    2.  Extraction of answers from the student script based on the generated JSON.
    Errors or poor quality in the first stage will directly impact the second stage.
*   **Answer Key Format (Gemini):** While text-based answer keys (`.txt`) are most reliable for the Gemini grader, PDF and image keys are subject to OCR accuracy. For best results, answer keys should have clear, machine-readable text.
*   **No Graphical User Interface (GUI):** This is a command-line tool.
*   **API Costs:** Use of the Google Gemini API may incur costs. Local Ollama usage depends on your own hardware.
*   **Error Handling:** While basic error handling is implemented, there might be edge cases not fully covered.

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
