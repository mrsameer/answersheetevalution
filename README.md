# AI Grading Agent

## Description
This project aims to build an intelligent agent that automates the grading of student answer sheets. The agent will be able to process answer sheets in PDF or image format, compare them against a provided answer key, and provide detailed scoring.

## Planned Features
- Process PDF answer sheets.
- Process image-based answer sheets.
- Compare student answers with a digital answer key.
- Calculate scores and provide a breakdown.
- Allow for different question types (e.g., multiple choice, short answer - future).
- Configuration for grading rubrics (future).

## Setup and Usage
(Instructions will be updated as development progresses)

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests:**
   ```bash
   pytest
   ```
5. **Run the agent (example):**
   ```bash
   python src/agent.py --sheet data/sample_student_sheet.pdf --key data/sample_answer_key.txt 
   ```
