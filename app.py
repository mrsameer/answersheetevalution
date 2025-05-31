import streamlit as st
import google.generativeai as genai
from PIL import Image
import pdf2image
import io
import json
import time
from datetime import datetime, timedelta
import hashlib
# import os # Not strictly needed for this version
# import requests # Not used in mock Adobe
from typing import List, Dict, Tuple, Optional

# Initialize session state
if 'token_history' not in st.session_state: # Overall grading process history
    st.session_state.token_history = []
if 'teacher_answers' not in st.session_state: # Temp storage for UI, might be redundant
    st.session_state.teacher_answers = None
if 'teacher_file_hash' not in st.session_state:
    st.session_state.teacher_file_hash = None
if 'cached_content' not in st.session_state:
    st.session_state.cached_content = None
if 'router_decisions_ui' not in st.session_state: # For UI display of router choices
    st.session_state.router_decisions_ui = []
if 'last_grading_result' not in st.session_state: # Stores data for meta-evaluation
    st.session_state.last_grading_result = None
if 'performance_evaluations' not in st.session_state: # Log of meta-evaluation runs
    st.session_state.performance_evaluations = []

# Operational logs for individual agents
if 'extractor_agent_log' not in st.session_state:
    st.session_state.extractor_agent_log = []
if 'evaluation_agent_log' not in st.session_state:
    st.session_state.evaluation_agent_log = []
if 'router_agent_log' not in st.session_state: # Log for router's internal operations
    st.session_state.router_agent_log = []


# Constants
USD_TO_INR = 85.56

# Model configurations
GEMINI_MODELS = {
    'gemini-2.5-pro-preview': {
        'name': 'Gemini 2.5 Pro Preview (Best for Complex Reasoning)',
        'model_id': 'models/gemini-2.5-pro-preview-05-06',
        'input': {'standard': 1.25, 'long': 2.50},
        'output': {'standard': 10.00, 'long': 15.00},
        'cached_input': {'standard': 0.31, 'long': 0.625},
        'cache_storage': 4.50, 'supports_caching': True,
        'best_for': ['complex_reasoning', 'evaluation', 'detailed_analysis', 'meta_evaluation']
    },
    'gemini-2.5-flash-preview': {
        'name': 'Gemini 2.5 Flash Preview (Hybrid Reasoning)',
        'model_id': 'models/gemini-2.5-flash-preview-05-20',
        'input': {'standard': 0.15, 'long': 0.15},
        'output': {'standard': 0.60, 'long': 0.60},
        'cached_input': {'standard': 0.0375, 'long': 0.0375},
        'cache_storage': 1.00, 'supports_caching': True,
        'best_for': ['reasoning', 'general_purpose', 'meta_evaluation_alternative']
    },
    'gemini-2.0-flash': {
        'name': 'Gemini 2.0 Flash (Balanced Performance)',
        'model_id': 'models/gemini-2.0-flash',
        'input': {'standard': 0.10, 'long': 0.10},
        'output': {'standard': 0.40, 'long': 0.40},
        'cached_input': {'standard': 0.025, 'long': 0.025},
        'cache_storage': 1.00, 'supports_caching': True,
        'best_for': ['multimodal', 'general_purpose', 'agents', 'extraction']
    },
    'gemini-1.5-flash': {
        'name': 'Gemini 1.5 Flash (Fast & Efficient)',
        'model_id': 'models/gemini-1.5-flash-001',
        'input': {'standard': 0.075, 'long': 0.15},
        'output': {'standard': 0.30, 'long': 0.60},
        'cached_input': {'standard': 0.01875, 'long': 0.0375},
        'cache_storage': 1.00, 'supports_caching': True,
        'best_for': ['speed', 'simple_extraction', 'high_volume', 'extraction']
    },
    'gemini-1.5-flash-8b': {
        'name': 'Gemini 1.5 Flash-8B (Most Economical)',
        'model_id': 'models/gemini-1.5-flash-8b-001',
        'input': {'standard': 0.0375, 'long': 0.075},
        'output': {'standard': 0.15, 'long': 0.30},
        'cached_input': {'standard': 0.01, 'long': 0.02},
        'cache_storage': 0.25, 'supports_caching': True,
        'best_for': ['budget', 'simple_tasks', 'high_volume', 'extraction_very_simple']
    },
    'gemini-1.5-pro': {
        'name': 'Gemini 1.5 Pro (Highest Intelligence)',
        'model_id': 'models/gemini-1.5-pro-002',
        'input': {'standard': 1.25, 'long': 2.50},
        'output': {'standard': 5.00, 'long': 10.00},
        'cached_input': {'standard': 0.3125, 'long': 0.625},
        'cache_storage': 4.50, 'supports_caching': True,
        'best_for': ['highest_accuracy', 'complex_documents', 'evaluation_complex']
    }
}
# Define actual prompts used
ACTUAL_EXTRACTION_PROMPT = """
    Extract all questions and answers from this answer sheet.
    Return a JSON with structure:
    {
        "answers": [
            {"question_number": 1, "question_text": "...", "answer": "..."}
        ]
    }
    Be thorough and capture every detail. Ensure question_number is an integer.
    If a question text is not clearly identifiable, you can omit "question_text" or set it to "N/A".
    Capture the full answer text as accurately as possible.
    """

ACTUAL_EVALUATION_PROMPT = """
    Compare these student answers with teacher answers and evaluate each.
    Consider:
    - Conceptual understanding over exact matching.
    - Partial credit for partially correct answers.
    - Different phrasings that mean the same thing.
    
    Return JSON:
    {
        "evaluations": [
            {
                "question_number": 1,
                "verdict": "Correct/Incorrect/Partially Correct",
                "score": 0.0 to 1.0 (float),
                "explanation": "Detailed explanation for the verdict and score."
            }
        ]
    }
    Ensure question_number is an integer and score is a float.
    """

def analyze_page_with_adobe(image: Image, adobe_api_key: str) -> Dict: # Mock
    width, height = image.size
    complexity_score = 0.3 + (hash(image.tobytes()) % 5 / 10.0) # pseudo-random complexity
    analysis = {
        'complexity_score': complexity_score,
        'content_types': ['text', 'handwriting' if complexity_score > 0.5 else 'text'],
        'has_tables': complexity_score > 0.6, 'has_diagrams': False, 'has_math': False,
        'text_density': 'medium', 'quality_score': 0.8, 'recommended_model': None
    }
    time.sleep(0.05) # Simulate API call latency
    return analysis

def route_to_best_model(page_analysis: Dict, task_type: str = 'extraction') -> str: # Router Agent
    complexity = page_analysis.get('complexity_score', 0.5)
    content_types = page_analysis.get('content_types', [])
    if task_type == 'evaluation': return 'gemini-2.5-pro-preview' # Best for evaluation
    if complexity > 0.7 or 'math' in content_types or page_analysis.get('has_diagrams'): return 'gemini-2.0-flash'
    elif complexity > 0.4 or 'handwriting' in content_types or page_analysis.get('has_tables'): return 'gemini-1.5-flash'
    else: return 'gemini-1.5-flash-8b'

def evaluate_grading_performance(
    full_result_data: Dict, extraction_model_key_used: str, evaluation_model_key_used: str,
    performance_model_key: str, extraction_prompt_text: str, evaluation_prompt_text: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    perf_model_config = GEMINI_MODELS[performance_model_key]
    perf_model = genai.GenerativeModel(perf_model_config['model_id'])

    teacher_extracted_answers = full_result_data.get('teacher_answers', [])
    student_extracted_answers = full_result_data.get('student_answers', [])
    ai_evaluations = full_result_data.get('evaluations', [])

    teacher_sample = [item for item in teacher_extracted_answers if isinstance(item, dict)][:3]
    student_sample = [item for item in student_extracted_answers if isinstance(item, dict)][:3]
    eval_sample = []
    if student_sample and ai_evaluations:
        student_q_numbers_in_sample = {s.get('question_number') for s in student_sample if isinstance(s, dict) and 'question_number' in s}
        eval_sample = [e for e in ai_evaluations if isinstance(e, dict) and e.get('question_number') in student_q_numbers_in_sample][:3]
    elif ai_evaluations:
        eval_sample = [item for item in ai_evaluations if isinstance(item, dict)][:3]

    num_questions_teacher = len(teacher_extracted_answers)
    num_questions_student = len(student_extracted_answers)
    num_evaluations_done = len(ai_evaluations)
    summary_for_prompt = (f"Extraction: Teacher sheet yielded {num_questions_teacher} items, "
                          f"Student sheet yielded {num_questions_student} items. "
                          f"Evaluation: {num_evaluations_done} evaluations were performed.")

    performance_prompt = f"""
As a meticulous Performance Evaluation Agent, your task is to analyze the performance of two sub-agents: an Extractor Agent and an Evaluation Agent, based on the provided data. Your goal is to provide QUANTIFIED feedback and actionable model recommendations for each.

CONTEXT:
- Extraction Model Used: {GEMINI_MODELS[extraction_model_key_used]['name']} (Key: {extraction_model_key_used})
- Extraction Prompt Snippet: "{extraction_prompt_text[:150].strip()}..."
- Evaluation Model Used: {GEMINI_MODELS[evaluation_model_key_used]['name']} (Key: {evaluation_model_key_used})
- Evaluation Prompt Snippet: "{evaluation_prompt_text[:150].strip()}..."
- Overall Summary: {summary_for_prompt}

INPUT DATA SAMPLES:
- Teacher's Extracted Answers Sample (from Extractor Agent):
{json.dumps(teacher_sample, indent=2)}

- Student's Extracted Answers Sample (from Extractor Agent):
{json.dumps(student_sample, indent=2)}

- AI Evaluations Sample (from Evaluation Agent for the student answers against teacher answers):
{json.dumps(eval_sample, indent=2)}

TASKS (Provide your analysis in the JSON structure specified at the end):

I. EXTRACTOR AGENT PERFORMANCE ANALYSIS:
   Focus on the quality of `Teacher's Extracted Answers` and `Student's Extracted Answers`.
   1. Completeness Score (0.0-1.0): How well did it seem to capture all questions and answers from the samples? Consider if question numbering is sequential or if gaps are apparent.
   2. Accuracy Score (0.0-1.0): Based on the samples, how accurate are the extracted question texts (if present) and answer texts? Do they seem coherent and correctly transcribed?
   3. Formatting Adherence Score (0.0-1.0): How well did the extracted output (represented by the samples) adhere to the expected structure (e.g., question_number (int), question_text (str/null), answer (str))?
   4. Overall Extractor Effectiveness Score (0.0-1.0): Your holistic assessment based on the above.
   5. Strengths: List 2-3 key strengths observed (e.g., "Good at capturing question numbers").
   6. Weaknesses: List 2-3 key weaknesses or areas for improvement (e.g., "Answer text sometimes seems truncated").
   7. Model Recommendation:
      - suggested_model_key: (Provide a Gemini model key from the list: {list(GEMINI_MODELS.keys())}, or 'current_is_optimal')
      - reasoning: (Explain WHY you recommend this model or staying with the current one, linking to observed strengths/weaknesses and potential benefits like cost/accuracy improvements or specific capabilities like handling handwriting if that was an issue.).
      - potential_trade_offs: (e.g., "Upgrade: Higher cost, potentially slower. Downgrade: Lower cost, risk of more errors.")
   8. Specific Extractor Issues (Provide 1-2 examples if any are apparent from samples):
      [{{ "item_type": "teacher/student", "observed_in_sample": {{...example item...}}, "issue_description": "e.g., Missing answer text", "suggestion_for_improvement": "e.g., Review extraction prompt for robust answer capture."}}]

II. EVALUATION AGENT PERFORMANCE ANALYSIS:
    Focus on the `AI Evaluations Sample` compared to the `Teacher's Extracted Answers Sample` and `Student's Extracted Answers Sample`.
    1. Verdict Appropriateness Score (0.0-1.0): How appropriate do the 'Correct', 'Incorrect', 'Partially Correct' verdicts seem given the provided answer pairs?
    2. Score Consistency Score (0.0-1.0): How consistently do the scores (0.0-1.0, float) align with the verdicts and the degree of correctness?
    3. Explanation Quality Score (0.0-1.0): How insightful, helpful, and clear are the explanations? Do they go beyond simple "matches/doesn't match"?
    4. Nuance Handling Score (0.0-1.0): How well does it seem to handle partial correctness, synonyms, or conceptual similarity (if inferable from samples)?
    5. Overall Evaluator Effectiveness Score (0.0-1.0): Your holistic assessment based on the above.
    6. Strengths: List 2-3 key strengths (e.g., "Clear for clear-cut correct/incorrect cases").
    7. Weaknesses: List 2-3 key weaknesses (e.g., "Explanations for partial credit are vague").
    8. Model Recommendation:
       - suggested_model_key: (Provide a Gemini model key from the list: {list(GEMINI_MODELS.keys())}, or 'current_is_optimal')
       - reasoning: (Explain based on observed evaluation quality. e.g., "Current model provides basic evaluations. For more nuanced feedback, upgrading to X is recommended.").
       - potential_trade_offs: (e.g., "Upgrade: Higher cost, better explanations. Downgrade: Lower cost, simpler explanations.")
    9. Specific Evaluator Issues (Provide 1-2 examples if any are apparent from samples):
      [{{ "question_number_evaluated": "Any", "student_answer_snippet": "...", "teacher_answer_snippet":"...", "evaluation_observed": {{...example evaluation...}}, "issue_description": "e.g., Explanation unhelpful", "suggestion_for_improvement": "e.g., Prompt for specific feedback points like 'identify correct parts and incorrect parts'."}}]

JSON RESPONSE STRUCTURE REQUIRED:
Return a single JSON object:
{{
  "extractor_agent_performance": {{
    "completeness_score": 0.0, "accuracy_score": 0.0, "formatting_adherence_score": 0.0, "overall_effectiveness_score": 0.0,
    "strengths": ["string"], "weaknesses": ["string"],
    "model_recommendation": {{"suggested_model_key": "string", "reasoning": "string", "potential_trade_offs": "string"}},
    "specific_issues_examples": [{{ "item_type": "string", "observed_in_sample": {{}}, "issue_description": "string", "suggestion_for_improvement": "string"}}]
  }},
  "evaluation_agent_performance": {{
    "verdict_appropriateness_score": 0.0, "score_consistency_score": 0.0, "explanation_quality_score": 0.0, "nuance_handling_score": 0.0, "overall_effectiveness_score": 0.0,
    "strengths": ["string"], "weaknesses": ["string"],
    "model_recommendation": {{"suggested_model_key": "string", "reasoning": "string", "potential_trade_offs": "string"}},
    "specific_issues_examples": [{{ "question_number_evaluated": "Any", "student_answer_snippet": "string", "teacher_answer_snippet":"string", "evaluation_observed": {{}}, "issue_description": "string", "suggestion_for_improvement": "string"}}]
  }}
}}
"""
    try:
        start_time = time.time()
        response = perf_model.generate_content(performance_prompt)
        end_time = time.time()
        
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        response_text = response.text
        try:
            json_start_index = response_text.find('{')
            json_end_index = response_text.rfind('}')
            if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                json_str = response_text[json_start_index : json_end_index + 1]
                evaluation_data = json.loads(json_str)
            else: raise json.JSONDecodeError("No valid JSON object found", response_text, 0)
        except json.JSONDecodeError as json_err:
            st.error(f"Failed to parse performance evaluation JSON: {json_err}")
            st.text_area("Problematic LLM Response (Meta-Eval):", response_text, height=200)
            default_issue = [{"issue_description": "Failed to get structured data from LLM.", "suggestion_for_improvement": "Check LLM response and prompt."}]
            evaluation_data = {
                "extractor_agent_performance": {"overall_effectiveness_score": 0.0, "strengths": ["Parsing Error"], "weaknesses": ["LLM parsing error"], "model_recommendation": {"suggested_model_key":"N/A", "reasoning":"Parsing Error"}, "specific_issues_examples": default_issue},
                "evaluation_agent_performance": {"overall_effectiveness_score": 0.0, "strengths": ["Parsing Error"], "weaknesses": ["LLM parsing error"], "model_recommendation": {"suggested_model_key":"N/A", "reasoning":"Parsing Error"}, "specific_issues_examples": default_issue}
            }

        input_cost_usd, input_cost_inr = calculate_cost(input_tokens, 'input', performance_model_key, False, input_tokens)
        output_cost_usd, output_cost_inr = calculate_cost(output_tokens, 'output', performance_model_key)
        
        stats = {
            'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': input_tokens + output_tokens,
            'input_cost_usd': input_cost_usd, 'output_cost_usd': output_cost_usd,
            'total_cost_usd': input_cost_usd + output_cost_usd,
            'input_cost_inr': input_cost_inr, 'output_cost_inr': output_cost_inr,
            'total_cost_inr': input_cost_inr + output_cost_inr,
            'processing_time': end_time - start_time,
            'model_used': perf_model_config['name'], 'model_key': performance_model_key
        }
        st.session_state.performance_evaluations.append({
            'timestamp': datetime.now().isoformat(),
            'evaluation_output': evaluation_data, 'stats': stats
        })
        return evaluation_data, stats
    except Exception as e:
        st.error(f"Error in performance evaluation agent: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def calculate_cost(tokens: int, token_type: str, model_key: str, cached: bool = False, prompt_length: int = 0) -> Tuple[float, float]:
    model_config = GEMINI_MODELS.get(model_key)
    if not model_config: return 0.0, 0.0 # Should not happen if model_key is valid
    threshold = 200000 if 'gemini-2.5' in model_key else 128000
    is_long = prompt_length > threshold
    rate_type = 'long' if is_long else 'standard'
    if token_type == 'input':
        rate = model_config['cached_input'][rate_type] if cached and model_config['supports_caching'] else model_config['input'][rate_type]
    else: rate = model_config['output'][rate_type]
    cost_usd = (tokens / 1_000_000) * rate
    cost_inr = cost_usd * USD_TO_INR
    return cost_usd, cost_inr

def convert_pdf_to_images(pdf_bytes):
    try: return pdf2image.convert_from_bytes(pdf_bytes)
    except Exception as e: st.error(f"Error converting PDF: {str(e)}"); return None

def process_file(uploaded_file):
    if uploaded_file is None: return None
    file_bytes = uploaded_file.getvalue() # Use getvalue to read multiple times if needed
    if uploaded_file.type == "application/pdf": return convert_pdf_to_images(file_bytes)
    else: return [Image.open(io.BytesIO(file_bytes))]

def extract_and_evaluate_with_routing(
    teacher_images: List[Image], student_images: List[Image], 
    ui_extraction_model_key: str, ui_evaluation_model_key: str,
    adobe_api_key_param: Optional[str] = None, use_cache: bool = False
) -> Tuple[Optional[Dict], Optional[Dict]]: # Returns raw_result_data, total_stats_for_grading_process
    
    # Determine actual models to use (UI choice or router's choice if Adobe enabled)
    # For now, we'll use the UI selected models and log router's suggestion separately
    actual_extraction_model_key = ui_extraction_model_key
    
    extraction_model_config = GEMINI_MODELS[actual_extraction_model_key]
    evaluation_model_config = GEMINI_MODELS[ui_evaluation_model_key]
    extraction_model = genai.GenerativeModel(extraction_model_config['model_id'])
    evaluation_model = genai.GenerativeModel(evaluation_model_config['model_id'])
    
    total_stats_for_grading_process = {
        'input_tokens': 0, 'output_tokens': 0, 'total_cost_usd': 0, 
        'total_cost_inr': 0, 'router_decisions_log': [] # For router agent log
    }
    
    # --- Router Agent Logging (Mock Adobe) ---
    if adobe_api_key_param and teacher_images:
        st.info("🔍 (Mock) Analyzing document complexity with Adobe Document Intelligence...")
        for i, img in enumerate(teacher_images):
            page_id = f'Teacher Page {i+1}'
            analysis_start_time = time.time()
            analysis = analyze_page_with_adobe(img, adobe_api_key_param) # Mock
            analysis_duration = time.time() - analysis_start_time
            routing_start_time = time.time()
            recommended_model_key_by_router = route_to_best_model(analysis, 'extraction')
            routing_duration = time.time() - routing_start_time
            
            router_decision_for_ui = { # For immediate UI display
                'page': page_id, 'complexity': analysis['complexity_score'],
                'recommended_model_name': GEMINI_MODELS[recommended_model_key_by_router]['name']
            }
            st.session_state.router_decisions_ui.append(router_decision_for_ui)
            
            st.session_state.router_agent_log.append({ # For router operational log
                'timestamp': datetime.now().isoformat(), 'page_id': page_id, 'task_type': 'extraction',
                'input_complexity_score': analysis.get('complexity_score'),
                'recommended_model_key': recommended_model_key_by_router,
                'recommended_model_name': GEMINI_MODELS[recommended_model_key_by_router]['name'],
                'analysis_time_seconds': analysis_duration,
                'routing_logic_time_seconds': routing_duration,
            })
    
    raw_result_data = {'teacher_answers': [], 'student_answers': [], 'evaluations': []}
    was_cached_teacher = False

    try: # --- Teacher Extraction ---
        st.info(f"📖 Extracting teacher answers using {extraction_model_config['name']}...")
        ext_start_time = time.time()
        teacher_content = [ACTUAL_EXTRACTION_PROMPT, "TEACHER ANSWER SHEET:"] + teacher_images
        if use_cache and st.session_state.cached_content and extraction_model_config['supports_caching']:
            try:
                cached_model = genai.GenerativeModel.from_cached_content(st.session_state.cached_content)
                teacher_response = cached_model.generate_content([])
                was_cached_teacher = True; st.success("Teacher content loaded from cache!")
            except Exception as cache_err:
                st.warning(f"Cache load failed: {cache_err}. Generating fresh content."); teacher_response = extraction_model.generate_content(teacher_content)
        else:
            teacher_response = extraction_model.generate_content(teacher_content)
            if extraction_model_config['supports_caching'] and not was_cached_teacher:
                try:
                    cache = genai.caching.CachedContent.create(model=extraction_model_config['model_id'], contents=teacher_content, ttl=timedelta(hours=1))
                    st.session_state.cached_content = cache; st.success("Teacher content cached.")
                except Exception as cache_create_err: st.warning(f"Failed to create cache: {cache_create_err}")
        
        ext_duration = time.time() - ext_start_time
        teacher_text = teacher_response.text
        json_valid = False
        try:
            parsed_json = json.loads(teacher_text[teacher_text.find('{'):teacher_text.rfind('}')+1])
            raw_result_data['teacher_answers'] = parsed_json.get('answers', [])
            json_valid = True
        except json.JSONDecodeError: st.error("Failed to parse teacher extraction JSON."); raw_result_data['teacher_answers'] = []
        
        tokens_in = teacher_response.usage_metadata.prompt_token_count
        tokens_out = teacher_response.usage_metadata.candidates_token_count
        cost_usd_in, cost_inr_in = calculate_cost(tokens_in, 'input', actual_extraction_model_key, was_cached_teacher, tokens_in)
        cost_usd_out, cost_inr_out = calculate_cost(tokens_out, 'output', actual_extraction_model_key)
        st.session_state.extractor_agent_log.append({
            'timestamp': datetime.now().isoformat(), 'document_type': 'teacher', 'model_key': actual_extraction_model_key,
            'model_name': extraction_model_config['name'], 'input_tokens': tokens_in, 'output_tokens': tokens_out,
            'cost_usd': cost_usd_in + cost_usd_out, 'cost_inr': cost_inr_in + cost_inr_out,
            'processing_time_seconds': ext_duration, 'json_output_valid': json_valid, 'cached_hit': was_cached_teacher
        })
        total_stats_for_grading_process['input_tokens'] += tokens_in; total_stats_for_grading_process['output_tokens'] += tokens_out
        total_stats_for_grading_process['total_cost_usd'] += cost_usd_in + cost_usd_out
        total_stats_for_grading_process['total_cost_inr'] += cost_inr_in + cost_inr_out
        total_stats_for_grading_process['cached_teacher_extraction'] = was_cached_teacher


        # --- Student Extraction ---
        st.info(f"📝 Extracting student answers using {extraction_model_config['name']}...")
        ext_start_time = time.time()
        student_content = [ACTUAL_EXTRACTION_PROMPT, "STUDENT ANSWER SHEET:"] + student_images
        student_response = extraction_model.generate_content(student_content)
        ext_duration = time.time() - ext_start_time
        student_text = student_response.text
        json_valid = False
        try:
            parsed_json = json.loads(student_text[student_text.find('{'):student_text.rfind('}')+1])
            raw_result_data['student_answers'] = parsed_json.get('answers', [])
            json_valid = True
        except json.JSONDecodeError: st.error("Failed to parse student extraction JSON."); raw_result_data['student_answers'] = []

        tokens_in = student_response.usage_metadata.prompt_token_count
        tokens_out = student_response.usage_metadata.candidates_token_count
        cost_usd_in, cost_inr_in = calculate_cost(tokens_in, 'input', actual_extraction_model_key, False, tokens_in)
        cost_usd_out, cost_inr_out = calculate_cost(tokens_out, 'output', actual_extraction_model_key)
        st.session_state.extractor_agent_log.append({
            'timestamp': datetime.now().isoformat(), 'document_type': 'student', 'model_key': actual_extraction_model_key,
            'model_name': extraction_model_config['name'], 'input_tokens': tokens_in, 'output_tokens': tokens_out,
            'cost_usd': cost_usd_in + cost_usd_out, 'cost_inr': cost_inr_in + cost_inr_out,
            'processing_time_seconds': ext_duration, 'json_output_valid': json_valid, 'cached_hit': False
        })
        total_stats_for_grading_process['input_tokens'] += tokens_in; total_stats_for_grading_process['output_tokens'] += tokens_out
        total_stats_for_grading_process['total_cost_usd'] += cost_usd_in + cost_usd_out
        total_stats_for_grading_process['total_cost_inr'] += cost_inr_in + cost_inr_out

        # --- Evaluation ---
        st.info(f"🎯 Evaluating answers using {evaluation_model_config['name']}...")
        eval_start_time = time.time()
        eval_content = [
            ACTUAL_EVALUATION_PROMPT,
            f"Teacher Answers JSON: {json.dumps(raw_result_data['teacher_answers'])}",
            f"Student Answers JSON: {json.dumps(raw_result_data['student_answers'])}"
        ]
        eval_response = evaluation_model.generate_content(eval_content)
        eval_duration = time.time() - eval_start_time
        eval_text = eval_response.text
        json_valid = False
        try:
            parsed_json = json.loads(eval_text[eval_text.find('{'):eval_text.rfind('}')+1])
            raw_result_data['evaluations'] = parsed_json.get('evaluations', [])
            json_valid = True
        except json.JSONDecodeError: st.error("Failed to parse evaluation JSON."); raw_result_data['evaluations'] = []
        
        tokens_in = eval_response.usage_metadata.prompt_token_count
        tokens_out = eval_response.usage_metadata.candidates_token_count
        cost_usd_in, cost_inr_in = calculate_cost(tokens_in, 'input', ui_evaluation_model_key, False, tokens_in)
        cost_usd_out, cost_inr_out = calculate_cost(tokens_out, 'output', ui_evaluation_model_key)
        num_q_input = len(raw_result_data.get('teacher_answers',[])) # Or student
        num_eval_output = len(raw_result_data.get('evaluations',[]))
        st.session_state.evaluation_agent_log.append({
            'timestamp': datetime.now().isoformat(), 'model_key': ui_evaluation_model_key,
            'model_name': evaluation_model_config['name'], 'input_tokens': tokens_in, 'output_tokens': tokens_out,
            'cost_usd': cost_usd_in + cost_usd_out, 'cost_inr': cost_inr_in + cost_inr_out,
            'processing_time_seconds': eval_duration, 'json_output_valid': json_valid,
            'num_questions_input': num_q_input, 'num_evaluations_output': num_eval_output
        })
        total_stats_for_grading_process['input_tokens'] += tokens_in; total_stats_for_grading_process['output_tokens'] += tokens_out
        total_stats_for_grading_process['total_cost_usd'] += cost_usd_in + cost_usd_out
        total_stats_for_grading_process['total_cost_inr'] += cost_inr_in + cost_inr_out
        
        total_stats_for_grading_process['total_tokens'] = total_stats_for_grading_process['input_tokens'] + total_stats_for_grading_process['output_tokens']
        
        # Log this entire grading process to token_history
        st.session_state.token_history.append({
            'timestamp': datetime.now().isoformat(), 'type': 'Complete Grading Process',
            'extraction_model_used': extraction_model_config['name'],
            'evaluation_model_used': evaluation_model_config['name'],
            **total_stats_for_grading_process
        })
        return raw_result_data, total_stats_for_grading_process
        
    except Exception as e:
        st.error(f"Error in main processing pipeline: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def main():
    st.set_page_config(page_title="AI Grader & Analyzer", page_icon="🎓", layout="wide")
    st.title("🎓 AI-Powered Answer Grader & Performance Analyzer")
    
    with st.sidebar:
        st.header("🔧 Configuration")
        gemini_api_key = st.text_input("Google Gemini API Key:", type="password", key="gemini_key")
        adobe_api_key_ui = st.text_input("Adobe API Key (Optional - Mock):", type="password", key="adobe_key_ui")
        if adobe_api_key_ui: st.success("✅ Adobe routing (mock) enabled")
        else: st.info("ℹ️ Adobe routing (mock) disabled")
        
        st.subheader("🤖 Agent Model Selection")
        extraction_model_choice = st.selectbox("Extractor Model:", options=list(GEMINI_MODELS.keys()), format_func=lambda x: GEMINI_MODELS[x]['name'], index=3, key="ext_model_choice")
        evaluation_model_choice = st.selectbox("Evaluator Model:", options=list(GEMINI_MODELS.keys()), format_func=lambda x: GEMINI_MODELS[x]['name'], index=0, key="eval_model_choice")
        st.subheader("🔬 Meta-Evaluation Agent")
        meta_eval_model_choice = st.selectbox("Meta-Evaluator Model:", options=list(GEMINI_MODELS.keys()), format_func=lambda x: GEMINI_MODELS[x]['name'], index=0, key="meta_eval_model_choice", help="Model to analyze other agents' performance.")

    if not gemini_api_key:
        st.warning("Please enter your Gemini API key."); return
    try: genai.configure(api_key=gemini_api_key)
    except Exception as e: st.error(f"Error configuring Gemini: {str(e)}"); return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👩‍🏫 Teacher Answer Sheet")
        teacher_file = st.file_uploader("Upload teacher's key (PDF/Image)", type=['png','jpg','jpeg','pdf'], key="teacher_file")
        if teacher_file:
            current_teacher_hash = hashlib.sha256(teacher_file.getvalue()).hexdigest()
            if st.session_state.teacher_file_hash != current_teacher_hash:
                st.session_state.teacher_file_hash = current_teacher_hash
                st.session_state.cached_content = None # Invalidate cache
                st.info("📝 New teacher sheet. Cache invalidated.")
            elif st.session_state.cached_content: st.success("✅ Teacher sheet matches. Cache can be used.")
    with col2:
        st.subheader("👨‍🎓 Student Answer Sheet")
        student_file = st.file_uploader("Upload student's sheet (PDF/Image)", type=['png','jpg','jpeg','pdf'], key="student_file")

    if st.button("🎯 Grade Answer Sheet", type="primary", disabled=not (teacher_file and student_file)):
        st.session_state.router_decisions_ui = [] # Clear previous UI router decisions
        with st.spinner("Grading in progress..."):
            teacher_images = process_file(teacher_file)
            student_images = process_file(student_file)
            if teacher_images and student_images:
                use_cache_flag = (st.session_state.cached_content is not None and 
                                  st.session_state.teacher_file_hash == hashlib.sha256(teacher_file.getvalue()).hexdigest())
                
                raw_results, grading_stats = extract_and_evaluate_with_routing(
                    teacher_images, student_images, extraction_model_choice,
                    evaluation_model_choice, adobe_api_key_ui, use_cache_flag
                )
                if raw_results and grading_stats:
                    st.success("Grading complete!")
                    # Store for meta-evaluation
                    st.session_state.last_grading_result = {
                        'raw_result_data': raw_results,
                        'extraction_model': extraction_model_choice,
                        'evaluation_model': evaluation_model_choice,
                        'extraction_prompt': ACTUAL_EXTRACTION_PROMPT,
                        'evaluation_prompt': ACTUAL_EVALUATION_PROMPT
                    }
                    # Display router decisions for UI (if any)
                    if st.session_state.router_decisions_ui:
                        with st.expander("🤖 Router Agent Decisions (Mock Adobe)", expanded=False):
                            for dec in st.session_state.router_decisions_ui:
                                st.write(f"- {dec['page']}: Complexity {dec['complexity']:.2f} → Recommends: {dec['recommended_model_name']}")
                    
                    # Display grading process stats
                    st.subheader("📊 Grading Process Statistics")
                    cols = st.columns(4)
                    cols[0].metric("Total Tokens (Grading)", f"{grading_stats.get('total_tokens',0):,}")
                    cols[1].metric("Total Cost (Grading)", f"₹{grading_stats.get('total_cost_inr',0):.3f}")
                    cols[2].metric("Teacher Cache Hit", "✅ Yes" if grading_stats.get('cached_teacher_extraction') else "❌ No")

                    # Display detailed graded results (as before)
                    teacher_ans_list = raw_results.get('teacher_answers', [])
                    student_ans_list = raw_results.get('student_answers', [])
                    evaluations_list = raw_results.get('evaluations', [])
                    
                    # Robust parsing of question numbers for dictionaries
                    teacher_dict = {int(ans['question_number']): ans for ans in teacher_ans_list if isinstance(ans, dict) and str(ans.get('question_number','')).isdigit()}
                    student_dict = {int(ans['question_number']): ans for ans in student_ans_list if isinstance(ans, dict) and str(ans.get('question_number','')).isdigit()}
                    eval_dict = {int(ev['question_number']): ev for ev in evaluations_list if isinstance(ev, dict) and str(ev.get('question_number','')).isdigit()}
                    
                    all_q_nums = sorted(list(set(teacher_dict.keys()) | set(student_dict.keys()) | set(eval_dict.keys())))
                    
                    display_results = []
                    for q_num in all_q_nums:
                        t_ans = teacher_dict.get(q_num, {})
                        s_ans = student_dict.get(q_num, {})
                        ev = eval_dict.get(q_num, {})
                        display_results.append({
                            'q_num': q_num,
                            'q_text': t_ans.get('question_text', s_ans.get('question_text', 'N/A')),
                            'teacher': t_ans.get('answer', '-'), 'student': s_ans.get('answer', '-'),
                            'verdict': ev.get('verdict', 'N/E'), 'score': float(ev.get('score', 0.0)),
                            'exp': ev.get('explanation', 'N/A')
                        })
                    
                    st.subheader("📋 Detailed Grading Results")
                    avg_score = sum(r['score'] for r in display_results) / len(display_results) * 100 if display_results else 0
                    st.metric("Average Score", f"{avg_score:.1f}%")

                    for res in display_results:
                        emoji = "✅" if res['verdict'] == 'Correct' else ("⚠️" if res['verdict'] == 'Partially Correct' else "❌")
                        with st.expander(f"{emoji} Q{res['q_num']}: {res['verdict']} (Score: {res['score']:.1f})"):
                            st.markdown(f"**Question:** {res['q_text']}")
                            r_cols = st.columns(2)
                            r_cols[0].info(f"**Student:** {res['student']}")
                            r_cols[1].success(f"**Teacher:** {res['teacher']}")
                            st.write(f"**Explanation:** {res['exp']}")
                            st.progress(res['score'])
                else: st.error("Grading process failed to return results.")
            else: st.error("Could not process uploaded files.")

    if st.session_state.last_grading_result and st.session_state.last_grading_result.get('raw_result_data'):
        st.divider()
        st.subheader("🔬 Meta-Evaluation: Agent Performance Analysis")
        if st.button("🚀 Analyze Agent Performance Now", type="primary", help="Run Meta-Evaluator LLM"):
            with st.spinner("Meta-Evaluation Agent is working..."):
                meta_input = st.session_state.last_grading_result
                feedback, meta_stats = evaluate_grading_performance(
                    meta_input['raw_result_data'], meta_input['extraction_model'], meta_input['evaluation_model'],
                    meta_eval_model_choice, meta_input['extraction_prompt'], meta_input['evaluation_prompt']
                )
                if feedback and meta_stats:
                    st.success("✅ Meta-Evaluation complete!")
                    ext_perf = feedback.get('extractor_agent_performance', {})
                    eval_perf = feedback.get('evaluation_agent_performance', {})

                    st.write("#### 🔦 Extractor Agent Analysis")
                    if ext_perf:
                        cols = st.columns(4)
                        cols[0].metric("Overall", f"{ext_perf.get('overall_effectiveness_score',0)*100:.0f}%")
                        cols[1].metric("Completeness", f"{ext_perf.get('completeness_score',0)*100:.0f}%")
                        cols[2].metric("Accuracy", f"{ext_perf.get('accuracy_score',0)*100:.0f}%")
                        cols[3].metric("Formatting", f"{ext_perf.get('formatting_adherence_score',0)*100:.0f}%")
                        st.markdown(f"**Strengths:** {', '.join(ext_perf.get('strengths',[])) if ext_perf.get('strengths') else 'N/A'}")
                        st.markdown(f"**Weaknesses:** {', '.join(ext_perf.get('weaknesses',[])) if ext_perf.get('weaknesses') else 'N/A'}")
                        rec = ext_perf.get('model_recommendation',{})
                        st.markdown(f"**Recommendation:** {GEMINI_MODELS.get(rec.get('suggested_model_key'),{}).get('name', rec.get('suggested_model_key','N/A'))}")
                        st.caption(f"Reason: {rec.get('reasoning','N/A')} | Trade-offs: {rec.get('potential_trade_offs','N/A')}")
                        if ext_perf.get('specific_issues_examples'):
                            with st.expander("Specific Extractor Issues Noted"):
                                for issue in ext_perf['specific_issues_examples']: st.write(issue)
                    
                    st.write("#### ⚖️ Evaluation Agent Analysis")
                    if eval_perf:
                        cols = st.columns(4)
                        cols[0].metric("Overall", f"{eval_perf.get('overall_effectiveness_score',0)*100:.0f}%")
                        cols[1].metric("Verdict Ok?", f"{eval_perf.get('verdict_appropriateness_score',0)*100:.0f}%")
                        cols[2].metric("Explanation Quality", f"{eval_perf.get('explanation_quality_score',0)*100:.0f}%")
                        cols[3].metric("Nuance Handling", f"{eval_perf.get('nuance_handling_score',0)*100:.0f}%")
                        st.markdown(f"**Strengths:** {', '.join(eval_perf.get('strengths',[])) if eval_perf.get('strengths') else 'N/A'}")
                        st.markdown(f"**Weaknesses:** {', '.join(eval_perf.get('weaknesses',[])) if eval_perf.get('weaknesses') else 'N/A'}")
                        rec = eval_perf.get('model_recommendation',{})
                        st.markdown(f"**Recommendation:** {GEMINI_MODELS.get(rec.get('suggested_model_key'),{}).get('name', rec.get('suggested_model_key','N/A'))}")
                        st.caption(f"Reason: {rec.get('reasoning','N/A')} | Trade-offs: {rec.get('potential_trade_offs','N/A')}")
                        if eval_perf.get('specific_issues_examples'):
                            with st.expander("Specific Evaluator Issues Noted"):
                                for issue in eval_perf['specific_issues_examples']: st.write(issue)

                    with st.expander("📊 Meta-Evaluation Agent Cost (this analysis)"):
                        st.write(f"Model: {meta_stats['model_used']}, Time: {meta_stats['processing_time']:.2f}s, Tokens: {meta_stats['total_tokens']:,}, Cost: ₹{meta_stats['total_cost_inr']:.3f}")
                else: st.error("Meta-Evaluation failed.")

    # --- History & Logs Section ---
    st.divider()
    st.header("📜 History & Operational Logs")
    
    with st.expander("📈 Overall Grading Process History"):
        if st.session_state.token_history:
            for run in reversed(st.session_state.token_history):
                st.markdown(f"**Run ({run['timestamp']}):** Ext: {run['extraction_model_used']}, Eval: {run['evaluation_model_used']}")
                st.markdown(f"Cost: ₹{run['total_cost_inr']:.3f}, Tokens: {run['total_tokens']:,}, Cached: {run.get('cached_teacher_extraction','N/A')}")
                st.divider()
        else: st.caption("No grading runs yet.")

    def display_op_log(log_name, log_data, time_col='processing_time_seconds', cost_col='cost_inr', success_col=None, success_label="Valid JSON"):
        with st.expander(f"🛠️ {log_name} Operational Log ({len(log_data)} entries)"):
            if log_data:
                st.dataframe(log_data[-5:]) # Show last 5 entries
                # Add summary stats if desired
            else: st.caption("No entries yet.")
    
    display_op_log("Extractor Agent", st.session_state.extractor_agent_log, success_col='json_output_valid')
    display_op_log("Evaluation Agent", st.session_state.evaluation_agent_log, success_col='json_output_valid')
    display_op_log("Router Agent (Mock Adobe)", st.session_state.router_agent_log, time_col='routing_logic_time_seconds', cost_col=None) # No direct cost for mock

    with st.expander("🔬 Meta-Evaluation Run History"):
        if st.session_state.performance_evaluations:
            for run in reversed(st.session_state.performance_evaluations):
                st.markdown(f"**Analysis ({run['timestamp']}):** Meta-Model: {run['stats']['model_used']}")
                st.markdown(f"Cost: ₹{run['stats']['total_cost_inr']:.3f}, Tokens: {run['stats']['total_tokens']:,}, Time: {run['stats']['processing_time']:.1f}s")
                # Optionally show summary scores from run['evaluation_output']
                st.divider()
        else: st.caption("No meta-evaluations run yet.")

    if st.button("🗑️ Clear All Logs & History"):
        keys_to_clear = ['token_history', 'teacher_file_hash', 'cached_content', 
                         'router_decisions_ui', 'last_grading_result', 'performance_evaluations', 
                         'extractor_agent_log', 'evaluation_agent_log', 'router_agent_log']
        for key in keys_to_clear:
            if key in st.session_state:
                if isinstance(st.session_state[key], list): st.session_state[key] = []
                else: st.session_state[key] = None
        st.rerun()

if __name__ == "__main__":
    main()