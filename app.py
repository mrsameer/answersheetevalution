import streamlit as st
import google.generativeai as genai
from PIL import Image
import pdf2image
import io
import json
import time
from datetime import datetime, timedelta
import hashlib
import os

# Initialize session state
if 'token_history' not in st.session_state:
    st.session_state.token_history = []
if 'teacher_answers' not in st.session_state:
    st.session_state.teacher_answers = None
if 'teacher_file_hash' not in st.session_state:
    st.session_state.teacher_file_hash = None
if 'cached_content' not in st.session_state:
    st.session_state.cached_content = None
if 'current_grading_results' not in st.session_state: # Initialize for clarity, though logic handles its absence
    st.session_state.current_grading_results = None
if 'last_evaluation_feedback' not in st.session_state:
    st.session_state.last_evaluation_feedback = None
if 'last_evaluation_stats' not in st.session_state:
    st.session_state.last_evaluation_stats = None

# Constants
USD_TO_INR = 83.0  # Exchange rate

# Pricing constants per 1M tokens in USD
# IMPORTANT: Prices for gemini-1.5-pro and gemini-1.0-pro are ILLUSTRATIVE EXAMPLES.
# Always refer to official Google Cloud documentation for current pricing.
PRICING_USD = {
    "gemini-1.5-flash": {
        'input': {'standard': 0.075, 'long': 0.15},    # ≤128k / >128k tokens
        'output': {'standard': 0.30, 'long': 0.60},   # ≤128k / >128k tokens
        'cached_input': {'standard': 0.01875, 'long': 0.0375}, # 75% discount
        'cache_storage_hourly': 0.00016667 # Example: $1.20/month for 1GB, approx per hour for typical cache object size. Placeholder.
                                         # Actual cache storage pricing is more complex (per GB-month).
    },
    "gemini-1.5-pro": { # EXAMPLE PRICING - VERIFY WITH OFFICIAL DOCS
        'input': {'standard': 0.5, 'long': 1.0},      # Example: Higher than Flash
        'output': {'standard': 1.5, 'long': 3.0},     # Example: Higher than Flash
        'cached_input': {'standard': 0.125, 'long': 0.25}, # Example: 75% discount
        'cache_storage_hourly': 0.00020 # Example placeholder
    },
    "gemini-1.0-pro": { # EXAMPLE PRICING - VERIFY WITH OFFICIAL DOCS
        'input': {'standard': 0.1},                   # Example: Simpler model, no long context distinction
        'output': {'standard': 0.3},                  # Example: Simpler model
        # 'cached_input': {}, # No distinct cached input pricing for this example
        # 'cache_storage_hourly': 0.0 # No distinct cache storage for this example
    }
}

def calculate_file_hash(file_bytes):
    """Calculate SHA256 hash of file bytes"""
    return hashlib.sha256(file_bytes).hexdigest()

def calculate_cost(tokens, model_name, token_type='input', cached=False, prompt_length=0):
    """
    Calculate the estimated cost of a GenAI API call based on token count, model, and type.

    Args:
        tokens (int): The number of tokens.
        model_name (str): The name of the Gemini model used (e.g., "gemini-1.5-flash").
        token_type (str, optional): Type of tokens, either 'input' or 'output'. 
                                    Defaults to 'input'.
        cached (bool, optional): Whether the input tokens were from a cached source. 
                                 Defaults to False. Affects input token cost only.
        prompt_length (int, optional): The total length of the prompt (typically input tokens)
                                       to determine if long-context pricing applies. 
                                       Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - cost_usd (float): Estimated cost in USD.
            - cost_inr (float): Estimated cost in INR.
    """
    model_pricing = PRICING_USD.get(model_name)
    if not model_pricing:
        st.warning(f"Pricing not found for model: {model_name}. Using default (Flash) rates.")
        model_pricing = PRICING_USD.get("gemini-1.5-flash")

    # Determine if it's a long context (>128k tokens for models that support it)
    # This threshold might vary by model, but 128k is a common one for Flash 1.5.
    # For models without 'long' pricing, it will default to 'standard'.
    is_long = prompt_length > 128000 
    rate_type = 'long' if is_long else 'standard'

    rate = 0
    if token_type == 'input':
        price_category = 'cached_input' if cached else 'input'
        category_rates = model_pricing.get(price_category)
        if category_rates:
            rate = category_rates.get(rate_type, category_rates.get('standard', 0))
        else: # Fallback if 'cached_input' or 'input' is not defined for the model
            st.warning(f"'{price_category}' pricing not defined for {model_name}. Using standard input rate.")
            rate = model_pricing.get('input', {}).get('standard', 0)
            
    elif token_type == 'output':
        category_rates = model_pricing.get('output')
        if category_rates:
            rate = category_rates.get(rate_type, category_rates.get('standard', 0))
        else: # Fallback if 'output' is not defined
            st.warning(f"'output' pricing not defined for {model_name}. Rate will be 0.")
            rate = 0
            
    if rate == 0 and tokens > 0:
        st.warning(
            f"Could not determine rate for {model_name}, type: {token_type}, "
            f"cached: {cached}, long_context: {is_long}. Cost will be 0."
        )
    
    # Convert to cost (rates are per 1M tokens)
    cost_usd = (tokens / 1_000_000) * rate
    cost_inr = cost_usd * USD_TO_INR
    
    return cost_usd, cost_inr

# Placeholder Router Function
def route_page_to_model(page_image, available_models):
    """
    Determines which GenAI model to use for processing a given page.
    
    This is currently a placeholder. In a future implementation, this function 
    would incorporate logic, potentially using Adobe Document Intelligence or 
    other heuristics, to select the most appropriate and cost-effective model 
    based on the page's content (e.g., text density, presence of images, diagrams).

    Args:
        page_image (PIL.Image.Image): The image of the page to analyze. 
                                     (Currently unused by placeholder logic).
        available_models (list): A list of available GenAI model names.

    Returns:
        str: The name of the selected GenAI model.
    """
    st.info("Router Agent: Analyzing page content to select optimal model... (Placeholder - using default model for now)")
    # TODO: Implement actual routing logic, possibly using Adobe Document Intelligence
    #       or other heuristics based on page_image content.
    if not available_models:
        st.warning("Router Agent: No available models provided. Defaulting to 'gemini-1.5-flash'.")
        return "gemini-1.5-flash" # Fallback if list is empty
    
    # Placeholder logic: always return the first model in the list.
    selected_model = available_models[0]
    # st.info(f"Router Agent: Selected model '{selected_model}' for this page. (Placeholder logic)")
    return selected_model

# Page-level Evaluation Prompt
page_evaluation_prompt_template = """
You are an AI assistant tasked with analyzing a single page from a teacher's answer key and a corresponding single page from a student's submission.
Your goal is to extract information and evaluate the student's work for THIS PAGE ONLY.

Instructions:
1.  Extract all questions and their corresponding answers from the teacher's page image.
2.  Extract all questions and their corresponding answers from the student's page image.
3.  For each question found on the student's page, compare it to the teacher's answer key.
4.  Provide a verdict ("Correct", "Incorrect", "Partially Correct") and a score (0.0 to 1.0) for each student answer.
5.  Briefly explain your reasoning for each evaluation.

Evaluation Guidelines:
*   Focus ONLY on content visible on the provided page images. Do not assume context from other pages.
*   Answers do not need to be word-for-word identical to be "Correct". Focus on conceptual understanding.
*   Mathematical expressions can be equivalent even if written differently.
*   Award partial credit if a student's answer is partially correct or shows some understanding.

Output Format:
Return ONLY a single JSON object starting with { and ending with }. Do not include any text before or after the JSON object.
The JSON object must have the following exact structure. Ensure all top-level keys ("teacher_answers_page", "student_answers_page", "evaluations_page") are present.
If no questions or answers are identifiable on a page, the value for the corresponding key should be an empty list ([]).

{
    "teacher_answers_page": [
        {
            "question_number_page": "e.g., 1a or Q1", // Identifier for the question on this page
            "question_text_page": "Full text of the question, if visible",
            "answer_page": "Teacher's answer for this question on this page"
        }
        // ... more teacher answers from this page
    ],
    "student_answers_page": [
        {
            "question_number_page": "e.g., 1a or Q1", // Identifier for the question on this page
            "question_text_page": "Full text of the question, if visible",
            "answer_page": "Student's answer for this question on this page"
        }
        // ... more student answers from this page
    ],
    "evaluations_page": [
        {
            "question_number_page": "e.g., 1a or Q1", // Identifier for the question being evaluated
            "verdict_page": "Correct" or "Incorrect" or "Partially Correct",
            "score_page": 0.0, // Score for this specific question on this page (float between 0.0 and 1.0)
            "explanation_page": "Brief explanation for the verdict and score on this page's question"
        }
        // ... more evaluations from this page
    ]
}
"""

def convert_pdf_to_images(pdf_bytes):
    """Convert PDF bytes to list of PIL images"""
    try:
        images = pdf2image.convert_from_bytes(pdf_bytes)
        return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return None

def process_file(uploaded_file):
    """Process uploaded file (PDF or image) and return PIL image(s)"""
    if uploaded_file is None:
        return None
    
    file_bytes = uploaded_file.read()
    
    if uploaded_file.type == "application/pdf":
        images = convert_pdf_to_images(file_bytes)
        return images
    else:
        # Handle image files
        image = Image.open(io.BytesIO(file_bytes))
        return [image]

# def extract_and_evaluate_with_caching(model, teacher_images, student_images, use_cache=False): # Original signature
def extract_and_evaluate_with_caching(main_model_placeholder, teacher_images, student_images, available_models_list, use_cache=False): # New signature
    """
    Processes teacher and student answer sheets page by page.

    For each pair of pages, it uses a router agent (currently a placeholder) to select 
    an appropriate GenAI model. It then calls the selected model to extract answers and 
    evaluate the student's work against the teacher's key for that specific page.
    The results from all pages are aggregated into a single JSON object.

    Document-level caching is currently disabled in this per-page processing flow.

    Args:
        main_model_placeholder (genai.GenerativeModel): A pre-initialized model instance, 
                                                       currently used as a fallback or for general config.
                                                       (Note: actual page processing uses per-page models).
        teacher_images (list of PIL.Image.Image): List of images for the teacher's answer key.
        student_images (list of PIL.Image.Image): List of images for the student's answer sheet.
        available_models_list (list): A list of model names available for the router agent.
        use_cache (bool, optional): Flag to indicate if caching should be used. 
                                    Currently ignored as caching is disabled in this function.

    Returns:
        tuple: A tuple containing:
            - aggregated_results (dict): A JSON-like dictionary containing all extracted 
                                         teacher answers, student answers, and evaluations, 
                                         aggregated across all processed pages.
            - overall_stats (dict): A dictionary with statistics about the entire 
                                    per-page processing operation (tokens, cost, time, etc.),
                                    including a 'page_details' list with stats for each page.
                                    Returns (None, None) if critical errors occur early.
    """
    st.warning("Note: Document-level caching is temporarily disabled due to the per-page routing structure. Per-page model selection is active.")
    
    aggregated_results = {
        "teacher_answers": [],
        "student_answers": [],
        "evaluations": []
    }
    all_page_stats = []
    
    # Determine the number of pages to process (min of teacher/student pages)
    num_pages = min(len(teacher_images), len(student_images))
    if len(teacher_images) != len(student_images):
        st.warning(
            f"Teacher and Student sheets have different page counts "
            f"({len(teacher_images)} vs {len(student_images)}). "
            f"Processing up to the minimum ({num_pages}) pages."
        )

    overall_start_time = time.time()
    total_input_tokens_overall = 0
    total_output_tokens_overall = 0
    total_cost_usd_overall = 0
    total_cost_inr_overall = 0
    
    # This is a simplified approach for question numbering. 
    # A more robust solution might involve the model returning global question numbers 
    # or using more sophisticated merging logic based on unique question identifiers.
    question_number_offset = 0 

    for i in range(num_pages):
        teacher_page = teacher_images[i]
        student_page = student_images[i]
        current_page_number_display = i + 1
        
        st.info(f"Processing Page {current_page_number_display} of {num_pages}...")
        
        # 1. Route page to model (placeholder)
        selected_model_for_page_name = route_page_to_model(student_page, available_models_list)
        
        try:
            # 2. Initialize GenerativeModel for this page
            page_model = genai.GenerativeModel(selected_model_for_page_name)
            
            # 3. Prepare content for API call
            content_for_page = [
                page_evaluation_prompt_template,
                f"TEACHER PAGE (Image {current_page_number_display}):",
                teacher_page,
                f"STUDENT PAGE (Image {current_page_number_display}):",
                student_page
            ]
            
            page_start_time = time.time()
            # 4. Call model.generate_content
            response_page = page_model.generate_content(content_for_page)
            page_end_time = time.time()
            
            if not response_page or not response_page.text:
                st.error(f"No response or empty response received from API for page {current_page_number_display} using model {selected_model_for_page_name}.")
                # Optionally, add a placeholder error entry for this page in results
                all_page_stats.append({
                    'page_number': current_page_number_display, 'model_used': selected_model_for_page_name, 
                    'status': 'Error - No API response', 'input_tokens': 0, 'output_tokens': 0, 
                    'total_tokens': 0, 'cost_usd': 0, 'cost_inr': 0, 'processing_time': page_end_time - page_start_time
                })
                continue

            # 5. Extract token counts and calculate costs for this page
            token_metadata_page = response_page.usage_metadata
            input_tokens_page = token_metadata_page.prompt_token_count if token_metadata_page else 0
            output_tokens_page = token_metadata_page.candidates_token_count if token_metadata_page else 0
            
            total_input_tokens_overall += input_tokens_page
            total_output_tokens_overall += output_tokens_page

            input_cost_usd_page, input_cost_inr_page = calculate_cost(
                input_tokens_page, selected_model_for_page_name, 
                token_type='input', cached=False, prompt_length=input_tokens_page
            )
            output_cost_usd_page, output_cost_inr_page = calculate_cost(
                output_tokens_page, selected_model_for_page_name, 
                token_type='output', prompt_length=input_tokens_page # Using input_tokens_page for context length of output
            )
            
            total_cost_usd_page = input_cost_usd_page + output_cost_usd_page
            total_cost_inr_page = input_cost_inr_page + output_cost_inr_page
            total_cost_usd_overall += total_cost_usd_page
            total_cost_inr_overall += total_cost_inr_page

            page_stats = {
                'page_number': current_page_number_display,
                'model_used': selected_model_for_page_name,
                'input_tokens': input_tokens_page,
                'output_tokens': output_tokens_page,
                'total_tokens': input_tokens_page + output_tokens_page,
                'cost_usd': total_cost_usd_page,
                'cost_inr': total_cost_inr_page,
                'processing_time': page_end_time - page_start_time,
                'status': 'Processed'
            }
            all_page_stats.append(page_stats)
            
            # 6. Parse page-level JSON response
            response_text_page = response_page.text
            
            # --- TEMPORARY DEBUGGING OUTPUT for '0 questions' issue ---
            # These UI elements display raw and parsed data from the LLM for each page.
            # They are intended to help diagnose inconsistencies in LLM output that might lead to
            # the "0 questions found" issue. Review for removal once the issue is confirmed resolved.
            st.text_area(f"Raw JSON Response (Page {current_page_number_display}, Model: {selected_model_for_page_name})", response_text_page, height=150, key=f"raw_resp_p{current_page_number_display}")
            # --- END TEMPORARY DEBUGGING ---

            page_result_data = None # Initialize before try block
            try:
                # Attempt to find the start and end of the JSON block
                json_start_page = response_text_page.find('{')
                json_end_page = response_text_page.rfind('}') + 1
                
                if json_start_page != -1 and json_end_page > json_start_page: # ensure end is after start
                    json_str_page = response_text_page[json_start_page:json_end_page]
                    page_result_data = json.loads(json_str_page)
                    
                    # --- TEMPORARY DEBUGGING OUTPUT for '0 questions' issue ---
                    if page_result_data:
                        st.json(page_result_data, expanded=False, key=f"parsed_json_p{current_page_number_display}")
                    # --- END TEMPORARY DEBUGGING ---
                else:
                    st.error(f"Could not find valid JSON structure in response for page {current_page_number_display}. Raw response snippet: {response_text_page[:500]}...")
                    page_stats['status'] = 'Error - JSON structure not found' 
                    page_stats['raw_response_snippet'] = response_text_page[:200] # Store snippet for history
                    continue 
            except json.JSONDecodeError as json_err:
                st.error(f"Failed to parse JSON response for page {current_page_number_display}. Error: {json_err}. Raw response snippet: {response_text_page[:500]}...")
                page_stats['status'] = 'Error - JSON parsing failed'
                page_stats['raw_response_snippet'] = response_text_page[:200] # Store snippet for history
                continue 
            
            if not page_result_data: # If page_result_data is None after attempts
                st.error(f"page_result_data is None for page {current_page_number_display}, skipping aggregation for this page.")
                page_stats['status'] = 'Error - Parsed data is None'
                continue

            # 7. Aggregate results
            # Ensure the keys exist in page_result_data, use .get() for safety
            teacher_answers_on_page = page_result_data.get("teacher_answers_page", [])
            student_answers_on_page = page_result_data.get("student_answers_page", [])
            evaluations_on_page = page_result_data.get("evaluations_page", [])

            if not isinstance(teacher_answers_on_page, list) or \
               not isinstance(student_answers_on_page, list) or \
               not isinstance(evaluations_on_page, list):
                st.error(f"Invalid data structure in parsed JSON for page {current_page_number_display}. Expected lists for main keys. Skipping aggregation for this page.")
                page_stats['status'] = 'Error - Invalid JSON data structure'
                page_stats['parsed_data_snippet'] = str(page_result_data)[:200]
                continue
            
            if not evaluations_on_page: # If there are no evaluations, it might indicate an issue or an empty page
                st.warning(f"No evaluations found in the response for page {current_page_number_display}. This page might contribute 0 questions to the total.")
                # This is not necessarily an error to stop processing, but good to note.
                # The page_stats will reflect 0 tokens for this page's LLM call if it truly returned nothing useful.

            max_q_num_on_this_page = 0

                def get_page_q_num_val(q_num_str):
                    # Extracts numeric part for offset calculation if possible
                    if isinstance(q_num_str, int): return q_num_str
                    if isinstance(q_num_str, str):
                        numeric_part = "".join(filter(str.isdigit, q_num_str))
                        if numeric_part: return int(numeric_part)
                    return 0 # Fallback for non-numeric/complex identifiers

                for ta_page in page_result_data.get("teacher_answers_page", []):
                    page_q_num = ta_page.get("question_number_page", "N/A")
                    # Attempt numeric offset if previous pages had numeric questions, else use P#-<id>
                    # This logic tries to create a somewhat sensible global question number.
                    try:
                        # If question_number_offset is > 0, we assume we are in a numeric sequence.
                        # Otherwise, we stick to P{page_num}-{item_id} format.
                        q_val_page = get_page_q_num_val(page_q_num)
                        agg_q_num = (question_number_offset + q_val_page) if (question_number_offset > 0 or (max_q_num_on_this_page == 0 and num_pages == 1)) else f"P{current_page_number_display}-{page_q_num}"
                        if q_val_page == 0 and not (question_number_offset > 0 or (max_q_num_on_this_page == 0 and num_pages == 1)): # if get_page_q_num_val returned 0, stick to string ID
                             agg_q_num = f"P{current_page_number_display}-{page_q_num}"
                    except:
                         agg_q_num = f"P{current_page_number_display}-{page_q_num}"
                    
                    aggregated_results["teacher_answers"].append({
                        "question_number": agg_q_num,
                        "question_text": ta_page.get("question_text_page", "N/A"),
                        "answer": ta_page.get("answer_page", "")
                    })
                    # Track max question number *parsed as int* on this page for next offset
                    current_q_val = get_page_q_num_val(page_q_num)
                    if current_q_val > max_q_num_on_this_page:
                         max_q_num_on_this_page = current_q_val


                for sa_page in page_result_data.get("student_answers_page", []):
                    page_q_num = sa_page.get("question_number_page", "N/A")
                    try:
                        q_val_page = get_page_q_num_val(page_q_num)
                        agg_q_num = (question_number_offset + q_val_page) if (question_number_offset > 0 or (max_q_num_on_this_page == 0 and num_pages == 1)) else f"P{current_page_number_display}-{page_q_num}"
                        if q_val_page == 0 and not (question_number_offset > 0 or (max_q_num_on_this_page == 0 and num_pages == 1)):
                             agg_q_num = f"P{current_page_number_display}-{page_q_num}"
                    except:
                        agg_q_num = f"P{current_page_number_display}-{page_q_num}"
                    aggregated_results["student_answers"].append({
                        "question_number": agg_q_num,
                        "question_text": sa_page.get("question_text_page", "N/A"),
                        "answer": sa_page.get("answer_page", "")
                    })

                for eval_page in page_result_data.get("evaluations_page", []):
                    page_q_num = eval_page.get("question_number_page", "N/A")
                    try:
                        q_val_page = get_page_q_num_val(page_q_num)
                        agg_q_num = (question_number_offset + q_val_page) if (question_number_offset > 0 or (max_q_num_on_this_page == 0 and num_pages == 1)) else f"P{current_page_number_display}-{page_q_num}"
                        if q_val_page == 0 and not (question_number_offset > 0 or (max_q_num_on_this_page == 0 and num_pages == 1)):
                             agg_q_num = f"P{current_page_number_display}-{page_q_num}"
                    except:
                        agg_q_num = f"P{current_page_number_display}-{page_q_num}"
                    aggregated_results["evaluations"].append({
                        "question_number": agg_q_num,
                        "verdict": eval_page.get("verdict_page", "Not evaluated"),
                        "score": eval_page.get("score_page", 0),
                        "explanation": eval_page.get("explanation_page", "")
                    })
                
                # Update question_number_offset for the next page
                if max_q_num_on_this_page > 0 : # If we found any numeric question id on this page
                    question_number_offset += max_q_num_on_this_page
                else: # Fallback if no numeric question numbers, use count of evaluations as proxy for items on page
                    question_number_offset += len(page_result_data.get("evaluations_page", []))
                # Removed the 'else' for "Failed to parse JSON response for page" as it's covered by the try-except json.JSONDecodeError
        
        except Exception as e_page:
            st.error(f"Unhandled error processing page {current_page_number_display} with model {selected_model_for_page_name}: {str(e_page)}")
            import traceback
            st.error(traceback.format_exc())
            # Add error entry to page_stats if an unexpected error occurs before stats are added
            if not any(p['page_number'] == current_page_number_display for p in all_page_stats):
                 all_page_stats.append({
                    'page_number': current_page_number_display, 'model_used': selected_model_for_page_name, 
                    'status': f'Error - Unhandled: {str(e_page)[:100]}', 'input_tokens': 0, 'output_tokens': 0, 
                    'total_tokens': 0, 'cost_usd': 0, 'cost_inr': 0, 'processing_time': 0
                })
            else: # If stats entry exists, update status
                for p_stat in all_page_stats:
                    if p_stat['page_number'] == current_page_number_display:
                        p_stat['status'] = f'Error - Unhandled: {str(e_page)[:100]}'
                        break
            continue # Continue to the next page
            
    overall_end_time = time.time()
    
    # Consolidate overall statistics
    overall_stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'Per-Page Extraction & Evaluation',
        'input_tokens': total_input_tokens_overall,
        'output_tokens': total_output_tokens_overall,
        'total_tokens': total_input_tokens_overall + total_output_tokens_overall,
        # Approximating split for display; actual per-page costs are in 'page_details'
        'input_cost_usd': sum(p['cost_usd'] * (p['input_tokens'] / p['total_tokens']) for p in all_page_stats if p['total_tokens'] > 0),
        'output_cost_usd': sum(p['cost_usd'] * (p['output_tokens'] / p['total_tokens']) for p in all_page_stats if p['total_tokens'] > 0),
        'total_cost_usd': total_cost_usd_overall,
        'input_cost_inr': sum(p['cost_inr'] * (p['input_tokens'] / p['total_tokens']) for p in all_page_stats if p['total_tokens'] > 0),
        'output_cost_inr': sum(p['cost_inr'] * (p['output_tokens'] / p['total_tokens']) for p in all_page_stats if p['total_tokens'] > 0),
        'total_cost_inr': total_cost_inr_overall,
        'cached': False, # Overall caching status (document-level) is false
        'savings_usd': 0, # No document-level caching savings
        'savings_inr': 0,
        'processing_time': overall_end_time - overall_start_time,
        'num_pages_processed': num_pages,
        'page_details': all_page_stats 
    }
    st.session_state.token_history.append(overall_stats)

    if 'teacher_answers' in aggregated_results:
        st.session_state.teacher_answers = aggregated_results['teacher_answers'] # Store aggregated
        
    return aggregated_results, overall_stats

evaluation_agent_prompt_template = """
You are an expert AI educational evaluator. You have been provided with a JSON object detailing the results of an automated grading process performed by another AI.
Your task is to critically assess this automated grading.

Input:
The JSON object below contains:
- "teacher_answers": Extracted answers from the teacher's answer key.
- "student_answers": Extracted answers from the student's submission.
- "evaluations": The initial AI's verdict, score, and explanation for each student answer.

Your Analysis Task:
Carefully review the provided grading data. Based on this data, provide your expert analysis focusing on the following aspects:
1.  Overall Quality: Assess the general quality, perceived accuracy, and fairness of the automated grading.
2.  Consistency: Comment on the consistency of how scores and feedback were applied across different questions and answers.
3.  Potential Biases/Issues: Identify any potential biases (e.g., overly lenient/strict, specific subject matter bias if discernible) or common areas where the initial grading AI might have struggled, misinterpreted information, or provided unclear explanations.
4.  Prompt/Process Improvement (Suggestions): If applicable, suggest improvements to the initial grading AI's prompts or the grading process itself that could lead to better results in the future.
5.  Other Suggestions: Any other relevant observations or suggestions for improving the automated grading.
6.  Summary: Provide a concise summary of your key findings.

Output Format:
Return your analysis as a single, well-formed JSON object with the following exact keys. Ensure your response is only this JSON object and nothing else.
{
    "overall_assessment": "Your detailed assessment of overall grading quality, fairness, and accuracy.",
    "consistency_check": "Your comments on scoring and feedback consistency.",
    "bias_detection": "Your findings on potential biases or areas where the initial AI struggled.",
    "prompt_suggestions": "Your specific suggestions for improving prompts or the grading process for the initial AI.",
    "general_suggestions": "Any other general suggestions for enhancing the grading.",
    "summary_of_findings": "A concise bullet-point or paragraph summary of your most important findings."
}

Here is the grading data to analyze:
"""

def run_evaluation_agent(grading_results_json, evaluation_model_name):
    """
    Runs a GenAI model to perform a meta-evaluation on the initial grading results.

    Args:
        grading_results_json (dict): The JSON output from 
                                     `extract_and_evaluate_with_caching` 
                                     (the `aggregated_results`).
        evaluation_model_name (str): The name of the Gemini model to be used 
                                     for this meta-evaluation.

    Returns:
        tuple: A tuple containing:
            - feedback_text (str or None): The textual feedback from the evaluation agent.
                                           Could be a JSON string if the model adheres to the prompt.
            - stats (dict or None): Statistics for the evaluation agent's API call (tokens, cost).
                                    Returns (None, None) if a critical error occurs.
    """
    st.info(f"Evaluation Agent: Analyzing grading results with {evaluation_model_name}...")
    try:
        model = genai.GenerativeModel(evaluation_model_name)
        
        # Construct the full prompt, ensuring the JSON data is properly stringified.
        # Using json.dumps for robust conversion of the grading data to a JSON string.
        grading_data_string = json.dumps(grading_results_json, indent=2)
        full_prompt = f"{evaluation_agent_prompt_template}\n\n{grading_data_string}"
        
        start_time = time.time()
        response = model.generate_content(full_prompt)
        end_time = time.time()
        
        if not response or not response.text: # Check if response or response.text is empty
            st.error(f"Evaluation Agent: No response or empty response received from API using model {evaluation_model_name}.")
            return None, None # Return None for both feedback and stats on error
            
        feedback_text = response.text # The raw text from the model
        
        # Extract token counts
        # Ensure usage_metadata is available and has the expected fields
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
        else:
            st.warning("Evaluation Agent: Usage metadata not available. Token counts will be zero.")

        # Calculate costs (assuming standard, non-cached pricing for the evaluation agent)
        # Note: The PRICING_USD constant is for Flash. This might need adjustment if Pro models have different pricing.
        input_cost_usd, input_cost_inr = calculate_cost(
            input_tokens, evaluation_model_name, 'input', cached=False, prompt_length=input_tokens
        )
        output_cost_usd, output_cost_inr = calculate_cost(
            output_tokens, evaluation_model_name, 'output', prompt_length=input_tokens
        )
        
        total_cost_usd = input_cost_usd + output_cost_usd
        total_cost_inr = input_cost_inr + output_cost_inr
        
        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': f'Evaluation Agent Analysis ({evaluation_model_name})',
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'input_cost_usd': input_cost_usd,
            'output_cost_usd': output_cost_usd,
            'total_cost_usd': total_cost_usd,
            'input_cost_inr': input_cost_inr,
            'output_cost_inr': output_cost_inr,
            'total_cost_inr': total_cost_inr,
            'cached': False, 
            'savings_usd': 0,
            'savings_inr': 0,
            'processing_time': end_time - start_time
        }
        
        return feedback_text, stats
        
    except Exception as e:
        st.error(f"Error in Evaluation Agent: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def main():
    st.set_page_config(page_title="Answer Sheet Grader", page_icon="📝", layout="wide")
    
    st.title("📝 Intelligent Answer Sheet Grader")
    st.markdown("AI-powered grading with context caching for cost optimization")
    
    # API Key input
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    
    if not api_key:
        st.warning("Please enter your Gemini API key to continue")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Configure Gemini
    try:
        genai.configure(api_key=api_key)
        
        available_models_tuple = ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro")
        # The main dropdown now selects a default/fallback model.
        # The router might override this on a per-page basis.
        default_model_name = st.selectbox(
            "Select Default/Fallback Gemini Model", # Label changed
            available_models_tuple
        )
        
        # This model instance is primarily for configuration reference or as a fallback.
        main_model_instance = genai.GenerativeModel(default_model_name)
        # st.session_state.selected_model_name = default_model_name # This was for caching model name
        st.session_state.available_models = list(available_models_tuple) # Store as list for router
        
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return
    
    # Pricing info
    with st.expander("💰 Pricing Information (Illustrative Examples)"):
        st.warning(
            "**Disclaimer:** Prices for models other than Gemini 1.5 Flash are illustrative examples and may not be accurate. "
            "Always verify current pricing with the official Google Cloud documentation."
        )
        for model_disp_name, prices in PRICING_USD.items():
            st.markdown(f"#### {model_disp_name.replace('-', ' ').title()}")
            md_table = f"| Context Length | Input (per 1M tokens) | Output (per 1M tokens) |\n|---|---|---|\n"
            
            std_input_usd = prices.get('input', {}).get('standard', 'N/A')
            std_output_usd = prices.get('output', {}).get('standard', 'N/A')
            std_input_inr = f"₹{std_input_usd * USD_TO_INR:.2f}" if isinstance(std_input_usd, (int, float)) else "N/A"
            std_output_inr = f"₹{std_output_usd * USD_TO_INR:.2f}" if isinstance(std_output_usd, (int, float)) else "N/A"
            md_table += f"| Standard (≤128k) | {std_input_inr} (${std_input_usd}) | {std_output_inr} (${std_output_usd}) |\n"

            if 'long' in prices.get('input', {}) or 'long' in prices.get('output', {}):
                long_input_usd = prices.get('input', {}).get('long', 'N/A')
                long_output_usd = prices.get('output', {}).get('long', 'N/A')
                long_input_inr = f"₹{long_input_usd * USD_TO_INR:.2f}" if isinstance(long_input_usd, (int, float)) else "N/A"
                long_output_inr = f"₹{long_output_usd * USD_TO_INR:.2f}" if isinstance(long_output_usd, (int, float)) else "N/A"
                md_table += f"| Long (>128k) | {long_input_inr} (${long_input_usd}) | {long_output_inr} (${long_output_usd}) |\n"

            if prices.get('cached_input'):
                cached_std_usd = prices['cached_input'].get('standard', 'N/A')
                cached_std_inr = f"₹{cached_std_usd * USD_TO_INR:.2f}" if isinstance(cached_std_usd, (int, float)) else "N/A"
                md_table += f"| Cached Input (Std) | {cached_std_inr} (${cached_std_usd}) | - |\n"
                if 'long' in prices['cached_input']:
                    cached_long_usd = prices['cached_input'].get('long', 'N/A')
                    cached_long_inr = f"₹{cached_long_usd * USD_TO_INR:.2f}" if isinstance(cached_long_usd, (int, float)) else "N/A"
                    md_table += f"| Cached Input (Long) | {cached_long_inr} (${cached_long_usd}) | - |\n"
            
            cache_storage_hourly_usd = prices.get('cache_storage_hourly', 'N/A')
            cache_storage_hourly_inr = f"₹{cache_storage_hourly_usd * USD_TO_INR:.4f}" if isinstance(cache_storage_hourly_usd, (int, float)) else "N/A"
            if cache_storage_hourly_usd != 'N/A':
                 md_table += f"| Cache Storage | {cache_storage_hourly_inr} (${cache_storage_hourly_usd:.6f}) per hour (Example) | - |\n"

            st.markdown(md_table)
            st.markdown("---")

        st.markdown(f"**General Exchange Rate Used:** $1 = ₹{USD_TO_INR}")
        
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👩‍🏫 Teacher Answer Sheet")
        teacher_file = st.file_uploader(
            "Upload teacher's answer key",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            key="teacher"
        )
        
        if teacher_file:
            # Check if it's the same teacher file
            file_bytes = teacher_file.read()
            teacher_file.seek(0)  # Reset file pointer
            file_hash = calculate_file_hash(file_bytes)
            
            if st.session_state.teacher_file_hash != file_hash:
                st.session_state.teacher_file_hash = file_hash
                st.session_state.teacher_answers = None
                st.session_state.cached_content = None
                st.info("📝 New teacher answer sheet detected")
            elif st.session_state.cached_content:
                st.success("✅ Will use cached teacher answer sheet (75% cost savings!)")
    
    with col2:
        st.subheader("👨‍🎓 Student Answer Sheet")
        student_file = st.file_uploader(
            "Upload student's answer sheet",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            key="student"
        )
    
    # Grade button
    if st.button("🎯 Grade Answer Sheet", type="primary", disabled=not (teacher_file and student_file)):
        with st.spinner("Processing answer sheets with AI..."):
            # Process both files
            teacher_images = process_file(teacher_file)
            student_images = process_file(student_file)
            
            if teacher_images and student_images:
                # Check if we should use cache
                use_cache = (st.session_state.cached_content is not None and 
                           st.session_state.teacher_file_hash == file_hash)
                
                # Extract and evaluate using the new per-page logic
                # Pass the main_model_instance (as a placeholder/default config) 
                # and the list of available_models.
                result_data, stats = extract_and_evaluate_with_caching(
                    main_model_instance, # Pass the instance configured with default model
                    teacher_images, 
                    student_images,
                    st.session_state.available_models, # Pass the list of model names
                    use_cache=False # Document-level caching is effectively disabled here
                )
                
                if result_data and stats:
                    # Display processing stats
                    st.write("### 📊 Processing Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
                    with col2:
                        st.metric("Total Cost", f"₹{stats['total_cost_inr']:.4f}")
                    with col3:
                        st.metric("Processing Time", f"{stats['processing_time']:.2f}s")
                    with col4:
                        # Overall 'cached' status is False due to per-page processing.
                        # Individual page caching might be a future feature.
                        st.metric("Cache Status", "Per-Page Routing") 
                        if stats.get('savings_inr', 0) > 0 : # This should be 0 for now
                             st.metric("Savings", f"₹{stats['savings_inr']:.4f}")
                    
                    # Detailed token breakdown
                    with st.expander("📈 Detailed Token & Cost Breakdown (Aggregated)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Aggregated Token Counts:**")
                            st.write(f"- Input Tokens: {stats.get('input_tokens',0):,}")
                            st.write(f"- Output Tokens: {stats.get('output_tokens',0):,}")
                            st.write(f"- Total Tokens: {stats.get('total_tokens',0):,}")
                            st.write(f"- Pages Processed: {stats.get('num_pages_processed',0)}")
                        
                        with col2:
                            st.write("**Aggregated Cost Breakdown:**")
                            # Displaying the approximated split for overall input/output costs
                            st.write(f"- Input Cost (Approx): ₹{stats.get('input_cost_inr',0):.4f} (${stats.get('input_cost_usd',0):.6f})")
                            st.write(f"- Output Cost (Approx): ₹{stats.get('output_cost_inr',0):.4f} (${stats.get('output_cost_usd',0):.6f})")
                            st.write(f"- **Total Cost: ₹{stats.get('total_cost_inr',0):.4f} (${stats.get('total_cost_usd',0):.6f})**")
                        
                        if 'page_details' in stats and stats['page_details']:
                            st.write("---")
                            st.write("**Per-Page Processing Details (Router Agent Choices):**")
                            
                            per_page_md = "| Page | Model Used | Input Tokens | Output Tokens | Total Tokens | Cost (INR) | Time (s) |\n"
                            per_page_md += "|---|---|---|---|---|---|---|\n"
                            for page_stat in stats['page_details']:
                                per_page_md += (
                                    f"| {page_stat['page_number']} | {page_stat['model_used']} | "
                                    f"{page_stat.get('input_tokens',0):,} | {page_stat.get('output_tokens',0):,} | {page_stat.get('total_tokens',0):,} | "
                                    f"₹{page_stat.get('cost_inr',0):.4f} | {page_stat.get('processing_time',0):.2f} |\n"
                                )
                            st.markdown(per_page_md)
                    
                    # Prepare graded results
                    # Ensure current_grading_results is updated for the Evaluation Agent section
                    st.session_state.current_grading_results = result_data
                    teacher_dict = {ans['question_number']: ans for ans in result_data.get('teacher_answers', [])}
                    student_dict = {ans['question_number']: ans for ans in result_data.get('student_answers', [])}
                    eval_dict = {eval['question_number']: eval for eval in result_data.get('evaluations', [])}
                    
                    graded_results = []
                    for q_num in sorted(teacher_dict.keys()):
                        teacher_ans = teacher_dict.get(q_num, {})
                        student_ans = student_dict.get(q_num, {})
                        evaluation = eval_dict.get(q_num, {})
                        
                        graded_results.append({
                            'question_number': q_num,
                            'question_text': teacher_ans.get('question_text', student_ans.get('question_text', 'N/A')),
                            'teacher_answer': teacher_ans.get('answer', 'No answer key'),
                            'student_answer': student_ans.get('answer', 'No answer'),
                            'verdict': evaluation.get('verdict', 'Not evaluated'),
                            'score': evaluation.get('score', 0),
                            'explanation': evaluation.get('explanation', 'No explanation')
                        })
                    
                    # Calculate overall score
                    total_questions = len(graded_results)
                    total_score = sum(r['score'] for r in graded_results)
                    average_score = (total_score / total_questions * 100) if total_questions > 0 else 0
                    
                    # Display overall results
                    st.write("### 📋 Overall Results")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Questions", total_questions)
                    with col2:
                        st.metric("Average Score", f"{average_score:.1f}%")
                    with col3:
                        fully_correct = sum(1 for r in graded_results if r['verdict'] == 'Correct')
                        st.metric("Fully Correct", fully_correct)
                    with col4:
                        partial_correct = sum(1 for r in graded_results if r['verdict'] == 'Partially Correct')
                        st.metric("Partially Correct", partial_correct)
                    
                    # Display detailed results
                    st.write("### 📝 Detailed Results")
                    # Store result_data in session state to make it available for the evaluation agent button
                    st.session_state.current_grading_results = result_data 
                    for result in graded_results:
                        # Determine color based on verdict
                        if result['verdict'] == 'Correct':
                            color = "green"
                            emoji = "✅"
                        elif result['verdict'] == 'Partially Correct':
                            color = "orange"
                            emoji = "⚠️"
                        else:
                            color = "red"
                            emoji = "❌"
                        
                        with st.expander(f"{emoji} Question {result['question_number']} - {result['verdict']} (Score: {result['score']:.1f})"):
                            if result['question_text'] != 'N/A':
                                st.write(f"**Question:** {result['question_text']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Student Answer:**")
                                st.info(result['student_answer'])
                            with col2:
                                st.write("**Teacher Answer:**")
                                st.success(result['teacher_answer'])
                            
                            st.write(f"**AI Evaluation:**")
                            st.markdown(f"<p style='color: {color}'><b>{result['verdict']}</b> - {result['explanation']}</p>", 
                                      unsafe_allow_html=True)
                            
                            # Score visualization
                            st.progress(result['score'])

            # Evaluation Agent Section
            if 'current_grading_results' in st.session_state and st.session_state.current_grading_results:
                st.write("---")
                st.subheader("🕵️ Evaluation Agent: Assess Grading Quality")
                
                eval_agent_model_name = st.selectbox(
                    "Select Evaluation Agent Model",
                    st.session_state.available_models, # Use the same list of available models
                    index=st.session_state.available_models.index("gemini-1.5-pro") if "gemini-1.5-pro" in st.session_state.available_models else 0, # Default to 1.5 Pro
                    key="eval_agent_model_select"
                )

                if st.button("🔎 Run Evaluation Agent on Grading", key="run_eval_agent_button"):
                    with st.spinner(f"Evaluation Agent is scrutinizing the grading with {eval_agent_model_name}..."):
                        feedback, eval_stats = run_evaluation_agent(
                            st.session_state.current_grading_results,
                            eval_agent_model_name
                        )
                        
                        # Always update session state for feedback and stats
                        # This ensures that if an error occurs and feedback/stats are None,
                        # the UI will reflect that (e.g., hide feedback area or show error message if feedback is an error string)
                        st.session_state.last_evaluation_feedback = feedback
                        st.session_state.last_evaluation_stats = eval_stats
                        
                        if eval_stats: # Only append to history if stats were successfully generated
                            st.session_state.token_history.append(eval_stats)
                        # If feedback is None due to an error in run_evaluation_agent, 
                        # the error would have been displayed via st.error within that function.
                        # The UI below will then not display the feedback area.

                if st.session_state.get('last_evaluation_feedback'): # Use .get() for safety, displays if feedback is not None/empty
                    st.markdown("#### Evaluation Agent Feedback:")
                    # Try to parse feedback as JSON for pretty display, otherwise show as text
                    try:
                        feedback_json = json.loads(st.session_state.last_evaluation_feedback)
                        st.json(feedback_json)
                    except json.JSONDecodeError:
                        st.text_area("Feedback", st.session_state.last_evaluation_feedback, height=300)
                    
                    eval_stats_display = st.session_state.get('last_evaluation_stats')
                    if eval_stats_display:
                        st.caption(
                            f"Evaluation Agent ({eval_stats_display.get('type', 'N/A').split('(')[-1][:-1] if '(' in eval_stats_display.get('type', '') else eval_stats_display.get('type', 'N/A')}) - "
                            f"Tokens: {eval_stats_display.get('total_tokens', 0):,} | "
                            f"Cost: ₹{eval_stats_display.get('total_cost_inr', 0):.4f} | "
                            f"Time: {eval_stats_display.get('processing_time', 0):.2f}s"
                        )
                    else:
                        st.caption("Evaluation Agent statistics are not available (e.g. due to an error during evaluation).")
    
    # Token usage history
    if st.session_state.token_history:
        st.write("---")
        st.write("### 📈 Usage History & Analytics")
        
        # Summary stats
        total_cost_inr = sum(stat['total_cost_inr'] for stat in st.session_state.token_history)
        total_cost_usd = sum(stat['total_cost_usd'] for stat in st.session_state.token_history)
        total_tokens = sum(stat['total_tokens'] for stat in st.session_state.token_history)
        total_savings_inr = sum(stat['savings_inr'] for stat in st.session_state.token_history)
        cached_calls = sum(1 for stat in st.session_state.token_history if stat['cached'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost", f"₹{total_cost_inr:.4f}")
            st.caption(f"(${total_cost_usd:.6f})")
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col3:
            st.metric("Cache Savings", f"₹{total_savings_inr:.4f}")
        with col4:
            st.metric("API Calls", len(st.session_state.token_history))
            st.caption(f"{cached_calls} cached")
        
        # Detailed history
        with st.expander("📜 View Detailed History"):
            for i, stat in enumerate(reversed(st.session_state.token_history)):
                st.write(f"**{stat['timestamp']}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Tokens:**")
                    st.write(f"- Input: {stat['input_tokens']:,}")
                    st.write(f"- Output: {stat['output_tokens']:,}")
                    st.write(f"- Total: {stat['total_tokens']:,}")
                with col2:
                    st.write("**Costs (INR):**")
                    st.write(f"- Input: ₹{stat['input_cost_inr']:.4f}")
                    st.write(f"- Output: ₹{stat['output_cost_inr']:.4f}")
                    st.write(f"- Total: ₹{stat['total_cost_inr']:.4f}")
                    if stat['savings_inr'] > 0:
                        st.success(f"Saved: ₹{stat['savings_inr']:.4f}")
                with col3:
                    st.write("**Details:**")
                    st.write(f"- Cached: {'✅ Yes' if stat['cached'] else '❌ No'}")
                    st.write(f"- Time: {stat['processing_time']:.2f}s")
                    st.write(f"- Type: {stat['type']}")
                
                st.divider()
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.token_history = []
            st.rerun()
    
    # Tips section
    with st.expander("💡 Cost Optimization Tips"):
        st.markdown("""
        1. **Always reuse the same teacher answer sheet** - The app caches it for 1 hour
        2. **Batch multiple students** - Grade all students against the same teacher sheet
        3. **First student costs more** - Subsequent students save 75% on teacher sheet processing
        4. **Cache expires after 1 hour** - Plan your grading sessions accordingly
        
        **Example Savings:**
        - First student: Full cost
        - Students 2-10: 75% savings on ~50% of tokens
        - Net savings: ~35-40% on total costs
        """)

if __name__ == "__main__":
    main()