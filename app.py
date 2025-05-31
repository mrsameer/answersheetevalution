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
import requests
from typing import List, Dict, Tuple, Optional

# Initialize session state
if 'token_history' not in st.session_state:
    st.session_state.token_history = []
if 'teacher_answers' not in st.session_state:
    st.session_state.teacher_answers = None
if 'teacher_file_hash' not in st.session_state:
    st.session_state.teacher_file_hash = None
if 'cached_content' not in st.session_state:
    st.session_state.cached_content = None
if 'router_decisions' not in st.session_state:
    st.session_state.router_decisions = []
if 'last_grading_result' not in st.session_state:
    st.session_state.last_grading_result = None
if 'performance_evaluations' not in st.session_state:
    st.session_state.performance_evaluations = []

# Constants
USD_TO_INR = 85.56  # Updated exchange rate from document

# Model configurations
GEMINI_MODELS = {
    'gemini-2.5-pro-preview': {
        'name': 'Gemini 2.5 Pro Preview (Best for Complex Reasoning)',
        'model_id': 'models/gemini-2.5-pro-preview',
        'input': {'standard': 1.25, 'long': 2.50},  # ≤200k / >200k
        'output': {'standard': 10.00, 'long': 15.00},
        'cached_input': {'standard': 0.31, 'long': 0.625},
        'cache_storage': 4.50,
        'supports_caching': True,
        'best_for': ['complex_reasoning', 'evaluation', 'detailed_analysis']
    },
    'gemini-2.5-flash-preview': {
        'name': 'Gemini 2.5 Flash Preview (Hybrid Reasoning)',
        'model_id': 'models/gemini-2.5-flash-preview',
        'input': {'standard': 0.15, 'long': 0.15},
        'output': {'standard': 0.60, 'long': 0.60},  # Non-thinking mode
        'cached_input': {'standard': 0.0375, 'long': 0.0375},
        'cache_storage': 1.00,
        'supports_caching': True,
        'best_for': ['reasoning', 'general_purpose']
    },
    'gemini-2.0-flash': {
        'name': 'Gemini 2.0 Flash (Balanced Performance)',
        'model_id': 'models/gemini-2.0-flash',
        'input': {'standard': 0.10, 'long': 0.10},
        'output': {'standard': 0.40, 'long': 0.40},
        'cached_input': {'standard': 0.025, 'long': 0.025},
        'cache_storage': 1.00,
        'supports_caching': True,
        'best_for': ['multimodal', 'general_purpose', 'agents']
    },
    'gemini-1.5-flash': {
        'name': 'Gemini 1.5 Flash (Fast & Efficient)',
        'model_id': 'models/gemini-1.5-flash-001',
        'input': {'standard': 0.075, 'long': 0.15},  # ≤128k / >128k
        'output': {'standard': 0.30, 'long': 0.60},
        'cached_input': {'standard': 0.01875, 'long': 0.0375},
        'cache_storage': 1.00,
        'supports_caching': True,
        'best_for': ['speed', 'simple_extraction', 'high_volume']
    },
    'gemini-1.5-flash-8b': {
        'name': 'Gemini 1.5 Flash-8B (Most Economical)',
        'model_id': 'models/gemini-1.5-flash-8b-001',
        'input': {'standard': 0.0375, 'long': 0.075},  # ≤128k / >128k
        'output': {'standard': 0.15, 'long': 0.30},
        'cached_input': {'standard': 0.01, 'long': 0.02},
        'cache_storage': 0.25,
        'supports_caching': True,
        'best_for': ['budget', 'simple_tasks', 'high_volume']
    },
    'gemini-1.5-pro': {
        'name': 'Gemini 1.5 Pro (Highest Intelligence)',
        'model_id': 'models/gemini-1.5-pro-002',
        'input': {'standard': 1.25, 'long': 2.50},  # ≤128k / >128k
        'output': {'standard': 5.00, 'long': 10.00},
        'cached_input': {'standard': 0.3125, 'long': 0.625},
        'cache_storage': 4.50,
        'supports_caching': True,
        'best_for': ['highest_accuracy', 'complex_documents']
    }
}

# Adobe Document Intelligence mock configuration (replace with actual API)
ADOBE_DOC_INTELLIGENCE_ENDPOINT = "https://api.adobe.io/document-services/..."
ADOBE_API_KEY = None  # Will be provided by user

def analyze_page_with_adobe(image: Image, adobe_api_key: str) -> Dict:
    """
    Analyze page content using Adobe Document Intelligence to determine complexity.
    This is a mock implementation - replace with actual Adobe API calls.
    """
    # Mock implementation - replace with actual Adobe Document Intelligence API
    # In production, you would:
    # 1. Convert image to bytes
    # 2. Send to Adobe Document Intelligence API
    # 3. Get analysis results
    
    # For now, we'll simulate analysis based on image characteristics
    width, height = image.size
    
    # Simulate complexity analysis
    complexity_score = 0.5  # Default medium complexity
    
    # Mock analysis results
    analysis = {
        'complexity_score': complexity_score,
        'content_types': ['text', 'handwriting'],
        'has_tables': False,
        'has_diagrams': False,
        'has_math': False,
        'text_density': 'medium',
        'quality_score': 0.8,
        'recommended_model': None  # Will be determined by router
    }
    
    # In production, this would be actual API call:
    # response = requests.post(
    #     ADOBE_DOC_INTELLIGENCE_ENDPOINT,
    #     headers={'X-API-Key': adobe_api_key},
    #     files={'document': image_bytes}
    # )
    # analysis = response.json()
    
    return analysis

def route_to_best_model(page_analysis: Dict, task_type: str = 'extraction') -> str:
    """
    Router agent that selects the best Gemini model based on page analysis.
    """
    complexity = page_analysis.get('complexity_score', 0.5)
    content_types = page_analysis.get('content_types', [])
    
    # Router logic based on complexity and task
    if task_type == 'evaluation':
        # Always use best model for evaluation
        return 'gemini-2.5-pro-preview'
    
    # For extraction tasks
    if complexity > 0.8 or 'math' in content_types or page_analysis.get('has_diagrams'):
        # Complex content needs better model
        return 'gemini-2.0-flash'
    elif complexity > 0.5 or 'handwriting' in content_types:
        # Medium complexity
        return 'gemini-1.5-flash'
    else:
        # Simple content - use most economical
        return 'gemini-1.5-flash-8b'

def evaluate_grading_performance(
    grading_results: List[Dict],
    extraction_model: str,
    evaluation_model: str,
    performance_model_key: str,
    extraction_prompt: str,
    evaluation_prompt: str
) -> Tuple[Dict, Dict]:
    """
    Performance Evaluation Agent that assesses the quality of grading and provides recommendations.
    """
    
    # Initialize performance evaluation model
    perf_model = genai.GenerativeModel(GEMINI_MODELS[performance_model_key]['model_id'])
    
    # Prepare grading summary
    correct_count = sum(1 for r in grading_results if r['verdict'] == 'Correct')
    incorrect_count = sum(1 for r in grading_results if r['verdict'] == 'Incorrect')
    partial_count = sum(1 for r in grading_results if r['verdict'] == 'Partially Correct')
    avg_score = sum(r['score'] for r in grading_results) / len(grading_results) if grading_results else 0
    
    # Sample of evaluations for analysis
    sample_evaluations = grading_results[:5] if len(grading_results) > 5 else grading_results
    
    performance_prompt = f"""
    As a Performance Evaluation Agent, analyze the quality of this AI grading system and provide recommendations.
    
    GRADING SYSTEM CONFIGURATION:
    - Extraction Model: {extraction_model}
    - Evaluation Model: {evaluation_model}
    - Extraction Prompt: "{extraction_prompt[:200]}..."
    - Evaluation Prompt: "{evaluation_prompt[:200]}..."
    
    GRADING RESULTS SUMMARY:
    - Total Questions: {len(grading_results)}
    - Correct: {correct_count} ({correct_count/len(grading_results)*100:.1f}%)
    - Incorrect: {incorrect_count} ({incorrect_count/len(grading_results)*100:.1f}%)
    - Partially Correct: {partial_count} ({partial_count/len(grading_results)*100:.1f}%)
    - Average Score: {avg_score*100:.1f}%
    
    SAMPLE EVALUATIONS:
    {json.dumps(sample_evaluations, indent=2)}
    
    Please analyze:
    1. Grading Consistency: Are the evaluations consistent and fair?
    2. Explanation Quality: Are the explanations clear and helpful?
    3. Scoring Accuracy: Does the scoring align with the explanations?
    4. Model Performance: Are the current models appropriate for this task?
    5. Prompt Effectiveness: How can the prompts be improved?
    
    Return a JSON response with this structure:
    {{
        "overall_quality_score": 0.0 to 1.0,
        "consistency_score": 0.0 to 1.0,
        "explanation_quality_score": 0.0 to 1.0,
        "scoring_accuracy_score": 0.0 to 1.0,
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "prompt_recommendations": {{
            "extraction_prompt_improvements": ["improvement1", "improvement2"],
            "evaluation_prompt_improvements": ["improvement1", "improvement2"]
        }},
        "model_recommendations": {{
            "extraction_model_suggestion": "model_name or 'current is optimal'",
            "evaluation_model_suggestion": "model_name or 'current is optimal'",
            "reasoning": "explanation"
        }},
        "specific_issues": [
            {{
                "question_number": 1,
                "issue": "description",
                "suggestion": "how to fix"
            }}
        ],
        "general_recommendations": ["recommendation1", "recommendation2"]
    }}
    """
    
    try:
        # Generate performance evaluation
        start_time = time.time()
        response = perf_model.generate_content(performance_prompt)
        end_time = time.time()
        
        # Extract token counts
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count
        
        # Parse response
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != 0:
            json_str = response_text[json_start:json_end]
            evaluation_data = json.loads(json_str)
        else:
            st.error("Failed to parse performance evaluation")
            return None, None
        
        # Calculate costs
        input_cost_usd, input_cost_inr = calculate_cost(
            input_tokens, 'input', performance_model_key, False, input_tokens
        )
        output_cost_usd, output_cost_inr = calculate_cost(
            output_tokens, 'output', performance_model_key
        )
        
        stats = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_cost_usd': input_cost_usd,
            'output_cost_usd': output_cost_usd,
            'total_cost_usd': input_cost_usd + output_cost_usd,
            'input_cost_inr': input_cost_inr,
            'output_cost_inr': output_cost_inr,
            'total_cost_inr': input_cost_inr + output_cost_inr,
            'processing_time': end_time - start_time,
            'model_used': GEMINI_MODELS[performance_model_key]['name']
        }
        
        # Store in session state
        st.session_state.performance_evaluations.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation': evaluation_data,
            'stats': stats
        })
        
        return evaluation_data, stats
        
    except Exception as e:
        st.error(f"Error in performance evaluation: {str(e)}")
        return None, None

def calculate_cost(tokens: int, token_type: str, model_key: str, cached: bool = False, prompt_length: int = 0) -> Tuple[float, float]:
    """Calculate cost based on token count, type, and model"""
    model_config = GEMINI_MODELS[model_key]
    
    # Determine if it's a long context
    threshold = 200000 if 'gemini-2.5' in model_key else 128000
    is_long = prompt_length > threshold
    
    if token_type == 'input':
        if cached:
            rate = model_config['cached_input']['long' if is_long else 'standard']
        else:
            rate = model_config['input']['long' if is_long else 'standard']
    else:  # output
        rate = model_config['output']['long' if is_long else 'standard']
    
    # Convert to cost (rates are per 1M tokens)
    cost_usd = (tokens / 1_000_000) * rate
    cost_inr = cost_usd * USD_TO_INR
    
    return cost_usd, cost_inr

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

def extract_and_evaluate_with_routing(
    teacher_images: List[Image], 
    student_images: List[Image], 
    extraction_model_key: str,
    evaluation_model_key: str,
    adobe_api_key: Optional[str] = None,
    use_cache: bool = False
) -> Tuple[Dict, Dict]:
    """
    Extract answers from both sheets and evaluate using router agent for optimal model selection.
    """
    
    # Initialize models
    extraction_model = genai.GenerativeModel(GEMINI_MODELS[extraction_model_key]['model_id'])
    evaluation_model = genai.GenerativeModel(GEMINI_MODELS[evaluation_model_key]['model_id'])
    
    # Base prompts
    extraction_prompt = """
    Extract all questions and answers from this answer sheet.
    Return a JSON with structure:
    {
        "answers": [
            {"question_number": 1, "question_text": "...", "answer": "..."}
        ]
    }
    Be thorough and capture every detail.
    """
    
    evaluation_prompt = """
    Compare these student answers with teacher answers and evaluate each.
    Consider:
    - Conceptual understanding over exact matching
    - Partial credit for partially correct answers
    - Different phrasings that mean the same thing
    
    Return JSON:
    {
        "evaluations": [
            {
                "question_number": 1,
                "verdict": "Correct/Incorrect/Partially Correct",
                "score": 0.0 to 1.0,
                "explanation": "..."
            }
        ]
    }
    """
    
    total_stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_cost_usd': 0,
        'total_cost_inr': 0,
        'models_used': [],
        'router_decisions': []
    }
    
    try:
        # Step 1: Analyze pages with Adobe (if API key provided)
        if adobe_api_key:
            st.info("🔍 Analyzing document complexity with Adobe Document Intelligence...")
            
            for i, img in enumerate(teacher_images):
                analysis = analyze_page_with_adobe(img, adobe_api_key)
                recommended_model = route_to_best_model(analysis, 'extraction')
                
                total_stats['router_decisions'].append({
                    'page': f'Teacher Page {i+1}',
                    'complexity': analysis['complexity_score'],
                    'recommended_model': recommended_model
                })
        
        # Step 2: Extract teacher answers
        st.info(f"📖 Extracting teacher answers using {GEMINI_MODELS[extraction_model_key]['name']}...")
        
        teacher_content = [extraction_prompt, "TEACHER ANSWER SHEET:"] + teacher_images
        
        # Check for caching
        if use_cache and st.session_state.cached_content and GEMINI_MODELS[extraction_model_key]['supports_caching']:
            try:
                cached_model = genai.GenerativeModel.from_cached_content(st.session_state.cached_content)
                teacher_response = cached_model.generate_content([])
                was_cached = True
            except:
                teacher_response = extraction_model.generate_content(teacher_content)
                was_cached = False
        else:
            teacher_response = extraction_model.generate_content(teacher_content)
            was_cached = False
            
            # Create cache for future use
            if GEMINI_MODELS[extraction_model_key]['supports_caching']:
                try:
                    cache = genai.caching.CachedContent.create(
                        model=GEMINI_MODELS[extraction_model_key]['model_id'],
                        contents=teacher_content,
                        ttl=timedelta(hours=1)
                    )
                    st.session_state.cached_content = cache
                except:
                    pass
        
        # Parse teacher answers
        teacher_text = teacher_response.text
        teacher_data = json.loads(teacher_text[teacher_text.find('{'):teacher_text.rfind('}')+1])
        
        # Update stats
        t_input = teacher_response.usage_metadata.prompt_token_count
        t_output = teacher_response.usage_metadata.candidates_token_count
        t_cost_usd, t_cost_inr = calculate_cost(t_input, 'input', extraction_model_key, was_cached, t_input)
        t_out_cost_usd, t_out_cost_inr = calculate_cost(t_output, 'output', extraction_model_key)
        
        total_stats['input_tokens'] += t_input
        total_stats['output_tokens'] += t_output
        total_stats['total_cost_usd'] += t_cost_usd + t_out_cost_usd
        total_stats['total_cost_inr'] += t_cost_inr + t_out_cost_inr
        
        # Step 3: Extract student answers
        st.info(f"📝 Extracting student answers using {GEMINI_MODELS[extraction_model_key]['name']}...")
        
        student_content = [extraction_prompt, "STUDENT ANSWER SHEET:"] + student_images
        student_response = extraction_model.generate_content(student_content)
        
        # Parse student answers
        student_text = student_response.text
        student_data = json.loads(student_text[student_text.find('{'):student_text.rfind('}')+1])
        
        # Update stats
        s_input = student_response.usage_metadata.prompt_token_count
        s_output = student_response.usage_metadata.candidates_token_count
        s_cost_usd, s_cost_inr = calculate_cost(s_input, 'input', extraction_model_key, False, s_input)
        s_out_cost_usd, s_out_cost_inr = calculate_cost(s_output, 'output', extraction_model_key)
        
        total_stats['input_tokens'] += s_input
        total_stats['output_tokens'] += s_output
        total_stats['total_cost_usd'] += s_cost_usd + s_out_cost_usd
        total_stats['total_cost_inr'] += s_cost_inr + s_out_cost_inr
        
        # Step 4: Evaluate using best model
        st.info(f"🎯 Evaluating answers using {GEMINI_MODELS[evaluation_model_key]['name']}...")
        
        eval_content = [
            evaluation_prompt,
            f"Teacher Answers: {json.dumps(teacher_data)}",
            f"Student Answers: {json.dumps(student_data)}"
        ]
        
        eval_response = evaluation_model.generate_content(eval_content)
        
        # Parse evaluations
        eval_text = eval_response.text
        eval_data = json.loads(eval_text[eval_text.find('{'):eval_text.rfind('}')+1])
        
        # Update stats
        e_input = eval_response.usage_metadata.prompt_token_count
        e_output = eval_response.usage_metadata.candidates_token_count
        e_cost_usd, e_cost_inr = calculate_cost(e_input, 'input', evaluation_model_key, False, e_input)
        e_out_cost_usd, e_out_cost_inr = calculate_cost(e_output, 'output', evaluation_model_key)
        
        total_stats['input_tokens'] += e_input
        total_stats['output_tokens'] += e_output
        total_stats['total_cost_usd'] += e_cost_usd + e_out_cost_usd
        total_stats['total_cost_inr'] += e_cost_inr + e_out_cost_inr
        total_stats['total_tokens'] = total_stats['input_tokens'] + total_stats['output_tokens']
        
        # Combine results
        result_data = {
            'teacher_answers': teacher_data['answers'],
            'student_answers': student_data['answers'],
            'evaluations': eval_data['evaluations']
        }
        
        # Store in session state
        st.session_state.teacher_answers = teacher_data['answers']
        st.session_state.router_decisions = total_stats['router_decisions']
        
        # Store last grading result for performance evaluation
        st.session_state.last_grading_result = {
            'graded_results': result_data,
            'extraction_model': extraction_model_key,
            'evaluation_model': evaluation_model_key,
            'extraction_prompt': extraction_prompt,
            'evaluation_prompt': evaluation_prompt
        }
        
        # Add to history
        st.session_state.token_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Complete Grading Process',
            'extraction_model': GEMINI_MODELS[extraction_model_key]['name'],
            'evaluation_model': GEMINI_MODELS[evaluation_model_key]['name'],
            **total_stats,
            'cached': was_cached
        })
        
        return result_data, total_stats
        
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def main():
    st.set_page_config(page_title="AI-Powered Answer Sheet Grader", page_icon="🎓", layout="wide")
    
    st.title("🎓 AI-Powered Answer Sheet Grader")
    st.markdown("Advanced grading system with model routing and intelligent evaluation")
    
    # Sidebar for API keys and model selection
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Keys
        st.subheader("API Keys")
        gemini_api_key = st.text_input("Google Gemini API Key:", type="password", key="gemini_key")
        adobe_api_key = st.text_input("Adobe Document Intelligence API Key (Optional):", type="password", key="adobe_key")
        
        if adobe_api_key:
            st.success("✅ Adobe routing enabled")
        else:
            st.info("ℹ️ Adobe routing disabled - using default models")
        
        # Model Selection
        st.subheader("Model Selection")
        
        extraction_model = st.selectbox(
            "Extraction Model:",
            options=list(GEMINI_MODELS.keys()),
            format_func=lambda x: GEMINI_MODELS[x]['name'],
            index=3  # Default to 1.5 Flash
        )
        
        evaluation_model = st.selectbox(
            "Evaluation Model:",
            options=list(GEMINI_MODELS.keys()),
            format_func=lambda x: GEMINI_MODELS[x]['name'],
            index=0  # Default to 2.5 Pro Preview
        )
        
        st.divider()
        
        # Performance Evaluation Model
        st.subheader("🔍 Performance Evaluation")
        
        performance_model = st.selectbox(
            "Performance Evaluation Model:",
            options=list(GEMINI_MODELS.keys()),
            format_func=lambda x: GEMINI_MODELS[x]['name'],
            index=0,  # Default to 2.5 Pro Preview
            help="This model analyzes the grading quality and provides recommendations"
        )
        
        # Display selected model costs
        st.subheader("Selected Model Pricing (₹)")
        
        ext_model = GEMINI_MODELS[extraction_model]
        eval_model = GEMINI_MODELS[evaluation_model]
        perf_model = GEMINI_MODELS[performance_model]
        
        st.write("**Extraction Model:**")
        st.write(f"- Input: ₹{ext_model['input']['standard'] * USD_TO_INR:.2f}/1M tokens")
        st.write(f"- Output: ₹{ext_model['output']['standard'] * USD_TO_INR:.2f}/1M tokens")
        
        st.write("**Evaluation Model:**")
        st.write(f"- Input: ₹{eval_model['input']['standard'] * USD_TO_INR:.2f}/1M tokens")
        st.write(f"- Output: ₹{eval_model['output']['standard'] * USD_TO_INR:.2f}/1M tokens")
        
        st.write("**Performance Model:**")
        st.write(f"- Input: ₹{perf_model['input']['standard'] * USD_TO_INR:.2f}/1M tokens")
        st.write(f"- Output: ₹{perf_model['output']['standard'] * USD_TO_INR:.2f}/1M tokens")
    
    if not gemini_api_key:
        st.warning("Please enter your Gemini API key in the sidebar")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Configure Gemini
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👩‍🏫 Teacher Answer Sheet")
        teacher_file = st.file_uploader(
            "Upload teacher's answer key",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            key="teacher"
        )
        
        if teacher_file:
            file_bytes = teacher_file.read()
            teacher_file.seek(0)
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            
            if st.session_state.teacher_file_hash != file_hash:
                st.session_state.teacher_file_hash = file_hash
                st.session_state.teacher_answers = None
                st.session_state.cached_content = None
                st.info("📝 New teacher answer sheet detected")
            elif st.session_state.cached_content:
                st.success("✅ Will use cached teacher content (75% savings!)")
    
    with col2:
        st.subheader("👨‍🎓 Student Answer Sheet")
        student_file = st.file_uploader(
            "Upload student's answer sheet",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            key="student"
        )
    
    # Grade button
    if st.button("🎯 Grade Answer Sheet", type="primary", disabled=not (teacher_file and student_file)):
        with st.spinner("Processing answer sheets..."):
            # Process files
            teacher_images = process_file(teacher_file)
            student_images = process_file(student_file)
            
            if teacher_images and student_images:
                # Check cache status
                use_cache = (st.session_state.cached_content is not None and 
                           st.session_state.teacher_file_hash == file_hash)
                
                # Process with routing
                result_data, stats = extract_and_evaluate_with_routing(
                    teacher_images,
                    student_images,
                    extraction_model,
                    evaluation_model,
                    adobe_api_key,
                    use_cache
                )
                
                if result_data and stats:
                    # Display router decisions if Adobe was used
                    if adobe_api_key and stats['router_decisions']:
                        with st.expander("🤖 Router Agent Decisions"):
                            for decision in stats['router_decisions']:
                                st.write(f"**{decision['page']}**")
                                st.write(f"- Complexity: {decision['complexity']:.2f}")
                                st.write(f"- Recommended: {decision['recommended_model']}")
                    
                    # Display processing stats
                    st.write("### 📊 Processing Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
                    with col2:
                        st.metric("Total Cost", f"₹{stats['total_cost_inr']:.2f}")
                    with col3:
                        st.metric("Models Used", len(set([extraction_model, evaluation_model])))
                    with col4:
                        st.metric("Cached", "Yes ✅" if stats.get('cached') else "No ❌")
                    
                    # Detailed breakdown
                    with st.expander("💰 Detailed Cost Breakdown"):
                        st.write(f"**Total Input Tokens:** {stats['input_tokens']:,}")
                        st.write(f"**Total Output Tokens:** {stats['output_tokens']:,}")
                        st.write(f"**Total Cost (INR):** ₹{stats['total_cost_inr']:.2f}")
                        st.write(f"**Total Cost (USD):** ${stats['total_cost_usd']:.4f}")
                        st.write(f"**Exchange Rate:** $1 = ₹{USD_TO_INR}")
                    
                    # Prepare graded results
                    teacher_dict = {ans['question_number']: ans for ans in result_data['teacher_answers']}
                    student_dict = {ans['question_number']: ans for ans in result_data['student_answers']}
                    eval_dict = {eval['question_number']: eval for eval in result_data['evaluations']}
                    
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
                    
                    # Store graded results for performance evaluation
                    st.session_state.last_grading_result = {
                        'graded_results': graded_results,
                        'extraction_model': extraction_model,
                        'evaluation_model': evaluation_model,
                        'extraction_prompt': "Extract all questions and answers from answer sheets",
                        'evaluation_prompt': "Compare and evaluate student answers against teacher answers"
                    }
    
    # Performance Evaluation Section
    if st.session_state.last_grading_result:
        st.write("---")
        st.write("### 🔍 Performance Evaluation")
        st.markdown("Analyze the quality of the grading system and get recommendations for improvement")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("The Performance Evaluation Agent will analyze your grading results and provide recommendations for improving accuracy and consistency.")
        with col2:
            if st.button("🎯 Evaluate Performance", type="secondary"):
                with st.spinner("Analyzing grading performance..."):
                    last_result = st.session_state.last_grading_result
                    
                    # Run performance evaluation
                    perf_eval, perf_stats = evaluate_grading_performance(
                        last_result['graded_results'],
                        last_result['extraction_model'],
                        last_result['evaluation_model'],
                        performance_model,
                        last_result['extraction_prompt'],
                        last_result['evaluation_prompt']
                    )
                    
                    if perf_eval and perf_stats:
                        # Display performance evaluation results
                        st.success("✅ Performance evaluation complete!")
                        
                        # Overall scores
                        st.write("#### 📊 Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Overall Quality", f"{perf_eval['overall_quality_score']*100:.1f}%")
                        with col2:
                            st.metric("Consistency", f"{perf_eval['consistency_score']*100:.1f}%")
                        with col3:
                            st.metric("Explanation Quality", f"{perf_eval['explanation_quality_score']*100:.1f}%")
                        with col4:
                            st.metric("Scoring Accuracy", f"{perf_eval['scoring_accuracy_score']*100:.1f}%")
                        
                        # Strengths and Weaknesses
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### ✅ Strengths")
                            for strength in perf_eval.get('strengths', []):
                                st.write(f"• {strength}")
                        
                        with col2:
                            st.write("#### ⚠️ Areas for Improvement")
                            for weakness in perf_eval.get('weaknesses', []):
                                st.write(f"• {weakness}")
                        
                        # Prompt Recommendations
                        with st.expander("📝 Prompt Improvement Recommendations"):
                            st.write("**Extraction Prompt Improvements:**")
                            for improvement in perf_eval.get('prompt_recommendations', {}).get('extraction_prompt_improvements', []):
                                st.write(f"• {improvement}")
                            
                            st.write("\n**Evaluation Prompt Improvements:**")
                            for improvement in perf_eval.get('prompt_recommendations', {}).get('evaluation_prompt_improvements', []):
                                st.write(f"• {improvement}")
                        
                        # Model Recommendations
                        with st.expander("🤖 Model Selection Recommendations"):
                            model_recs = perf_eval.get('model_recommendations', {})
                            
                            st.write(f"**Extraction Model:**")
                            st.write(f"Current: {GEMINI_MODELS[last_result['extraction_model']]['name']}")
                            st.write(f"Recommendation: {model_recs.get('extraction_model_suggestion', 'Current is optimal')}")
                            
                            st.write(f"\n**Evaluation Model:**")
                            st.write(f"Current: {GEMINI_MODELS[last_result['evaluation_model']]['name']}")
                            st.write(f"Recommendation: {model_recs.get('evaluation_model_suggestion', 'Current is optimal')}")
                            
                            if 'reasoning' in model_recs:
                                st.write(f"\n**Reasoning:** {model_recs['reasoning']}")
                        
                        # Specific Issues
                        if perf_eval.get('specific_issues'):
                            with st.expander("🔍 Specific Issues Found"):
                                for issue in perf_eval['specific_issues']:
                                    st.write(f"**Question {issue['question_number']}:**")
                                    st.write(f"- Issue: {issue['issue']}")
                                    st.write(f"- Suggestion: {issue['suggestion']}")
                                    st.divider()
                        
                        # General Recommendations
                        with st.expander("💡 General Recommendations"):
                            for rec in perf_eval.get('general_recommendations', []):
                                st.write(f"• {rec}")
                        
                        # Performance Stats
                        with st.expander("📊 Performance Evaluation Stats"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Model Used:** {perf_stats['model_used']}")
                                st.write(f"**Processing Time:** {perf_stats['processing_time']:.2f}s")
                            with col2:
                                st.write(f"**Tokens:** {perf_stats['total_tokens']:,}")
                                st.write(f"**Cost:** ₹{perf_stats['total_cost_inr']:.2f} (${perf_stats['total_cost_usd']:.4f})")
    
    # Performance Evaluation History
    if st.session_state.performance_evaluations:
        with st.expander("📜 Performance Evaluation History"):
            for eval_record in reversed(st.session_state.performance_evaluations):
                st.write(f"**{eval_record['timestamp']}**")
                eval_data = eval_record['evaluation']
                stats = eval_record['stats']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Overall Quality: {eval_data['overall_quality_score']*100:.1f}%")
                with col2:
                    st.write(f"Model: {stats['model_used']}")
                with col3:
                    st.write(f"Cost: ₹{stats['total_cost_inr']:.2f}")
                
                st.divider()
    
    # Token usage history
    if st.session_state.token_history:
        st.write("---")
        st.write("### 📈 Usage History & Analytics")
        
        # Summary stats
        total_cost_inr = sum(stat['total_cost_inr'] for stat in st.session_state.token_history)
        total_cost_usd = sum(stat['total_cost_usd'] for stat in st.session_state.token_history)
        total_tokens = sum(stat['total_tokens'] for stat in st.session_state.token_history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost", f"₹{total_cost_inr:.2f}")
            st.caption(f"(${total_cost_usd:.4f})")
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col3:
            st.metric("API Calls", len(st.session_state.token_history))
        with col4:
            avg_cost = total_cost_inr / len(st.session_state.token_history) if st.session_state.token_history else 0
            st.metric("Avg Cost/Call", f"₹{avg_cost:.2f}")
        
        # Model usage stats
        with st.expander("📊 Model Usage Statistics"):
            model_usage = {}
            for stat in st.session_state.token_history:
                ext_model = stat.get('extraction_model', 'Unknown')
                eval_model = stat.get('evaluation_model', 'Unknown')
                
                model_usage[ext_model] = model_usage.get(ext_model, 0) + 1
                model_usage[eval_model] = model_usage.get(eval_model, 0) + 1
            
            for model, count in model_usage.items():
                st.write(f"**{model}:** {count} uses")
        
        # Detailed history
        with st.expander("📜 View Detailed History"):
            for i, stat in enumerate(reversed(st.session_state.token_history)):
                st.write(f"**{stat['timestamp']}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Models:**")
                    st.write(f"- Extraction: {stat.get('extraction_model', 'N/A')}")
                    st.write(f"- Evaluation: {stat.get('evaluation_model', 'N/A')}")
                with col2:
                    st.write("**Costs:**")
                    st.write(f"- Total: ₹{stat['total_cost_inr']:.2f} (${stat['total_cost_usd']:.4f})")
                    st.write(f"- Tokens: {stat['total_tokens']:,}")
                
                st.divider()
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.token_history = []
            st.session_state.router_decisions = []
            st.session_state.performance_evaluations = []
            st.session_state.last_grading_result = None
            st.rerun()
    
    # Tips section
    with st.expander("💡 Optimization Tips"):
        st.markdown("""
        ### Model Selection Guide:
        
        **For Extraction:**
        - **Gemini 1.5 Flash-8B**: Best for simple, clear answer sheets (₹3.21/1M tokens)
        - **Gemini 1.5 Flash**: Good balance of speed and accuracy (₹6.42/1M tokens)
        - **Gemini 2.0 Flash**: Best for complex layouts (₹8.56/1M tokens)
        
        **For Evaluation:**
        - **Gemini 2.5 Pro Preview**: Most accurate evaluation (₹106.95/1M tokens input)
        - **Gemini 2.0 Flash**: Good balance (₹34.22/1M tokens output)
        
        **For Performance Analysis:**
        - **Gemini 2.5 Pro Preview**: Best for comprehensive analysis
        - **Gemini 1.5 Pro**: Alternative for detailed insights
        
        ### Performance Evaluation Benefits:
        1. **Quality Assurance**: Verify grading consistency and accuracy
        2. **Prompt Optimization**: Get specific suggestions for better prompts
        3. **Model Selection**: Discover if you're using optimal models
        4. **Cost vs Quality**: Balance accuracy with operational costs
        
        ### Cost Optimization:
        1. **Use Adobe routing** to automatically select optimal models per page
        2. **Batch grade** multiple students with same teacher sheet
        3. **Use economical models** for simple answer sheets
        4. **Enable caching** for repeated teacher sheets
        5. **Run performance evaluation** periodically to optimize your setup
        """)
        
    # How to Use Performance Evaluation
    with st.expander("🔍 How to Use Performance Evaluation"):
        st.markdown("""
        ### Performance Evaluation Agent Guide:
        
        1. **Grade some answer sheets** first to generate results
        2. **Click "Evaluate Performance"** to analyze grading quality
        3. **Review the metrics**:
           - Overall Quality Score
           - Consistency Score
           - Explanation Quality
           - Scoring Accuracy
        
        4. **Apply recommendations**:
           - Update prompts based on suggestions
           - Switch models if recommended
           - Address specific issues identified
        
        5. **Iterate and improve**:
           - Re-grade with new settings
           - Run performance evaluation again
           - Track improvements over time
        
        ### When to Use:
        - After grading your first batch
        - When results seem inconsistent
        - Before processing large batches
        - When testing new model combinations
        """)

if __name__ == "__main__":
    main()