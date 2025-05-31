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

# Constants
USD_TO_INR = 83.0  # Exchange rate

# Pricing constants for Gemini 1.5 Flash (per 1M tokens in USD)
PRICING_USD = {
    'input': {
        'standard': 0.075,  # ≤128k tokens
        'long': 0.15,       # >128k tokens
    },
    'output': {
        'standard': 0.30,   # ≤128k tokens
        'long': 0.60,       # >128k tokens
    },
    'cached_input': {
        'standard': 0.01875,  # ≤128k tokens (75% discount)
        'long': 0.0375,       # >128k tokens (75% discount)
    },
    'cache_storage': 1.00  # per hour
}

def calculate_file_hash(file_bytes):
    """Calculate SHA256 hash of file bytes"""
    return hashlib.sha256(file_bytes).hexdigest()

def calculate_cost(tokens, token_type='input', cached=False, prompt_length=0):
    """Calculate cost based on token count and type"""
    # Determine if it's a long context (>128k tokens)
    is_long = prompt_length > 128000
    
    if token_type == 'input':
        if cached:
            rate = PRICING_USD['cached_input']['long' if is_long else 'standard']
        else:
            rate = PRICING_USD['input']['long' if is_long else 'standard']
    else:  # output
        rate = PRICING_USD['output']['long' if is_long else 'standard']
    
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

def extract_and_evaluate_with_caching(model, teacher_images, student_images, use_cache=False):
    """Extract answers from both sheets and evaluate in a single call"""
    
    # Base prompt for extraction and evaluation
    evaluation_prompt = """
    You will analyze two answer sheets: a teacher's answer key and a student's submission.
    
    First, extract all answers from both sheets, then evaluate the student's answers.
    
    For evaluation, consider:
    - Answers don't need to be word-for-word identical
    - Mathematical expressions can be written differently but mean the same
    - Consider partial credit for partially correct answers
    - Check for conceptual understanding, not just exact matching
    
    Return a JSON response with this exact structure:
    {
        "teacher_answers": [
            {
                "question_number": 1,
                "question_text": "Question if visible",
                "answer": "Teacher's answer"
            }
        ],
        "student_answers": [
            {
                "question_number": 1,
                "question_text": "Question if visible", 
                "answer": "Student's answer"
            }
        ],
        "evaluations": [
            {
                "question_number": 1,
                "verdict": "Correct" or "Incorrect" or "Partially Correct",
                "score": 0.0 to 1.0,
                "explanation": "Why this score was given"
            }
        ]
    }
    """
    
    try:
        start_time = time.time()
        response = None
        was_cached = False
        
        # If we should use cache and have a cached content object
        if use_cache and st.session_state.cached_content:
            try:
                # Try to use the cached content
                cached_model = genai.GenerativeModel.from_cached_content(
                    cached_content=st.session_state.cached_content
                )
                
                # For cached content, we only need to provide the student images
                # since teacher images are already in the cache
                content = ["Now analyze this student answer sheet:"] + student_images
                
                response = cached_model.generate_content(content)
                was_cached = True
                st.success("✅ Successfully used cached teacher answer sheet!")
                
            except Exception as e:
                st.warning(f"Cache miss or error: {str(e)}. Creating new cache...")
                was_cached = False
        
        # If not using cache or cache failed, make a regular call
        if not was_cached:
            # Prepare content with both teacher and student images
            content = [
                evaluation_prompt,
                "TEACHER ANSWER SHEET:"
            ] + teacher_images + [
                "STUDENT ANSWER SHEET:"
            ] + student_images
            
            # Check if we need to create a cache for teacher content
            if not use_cache:  # First time processing this teacher sheet
                try:
                    # Create cached content for teacher sheet
                    teacher_content = [
                        evaluation_prompt,
                        "TEACHER ANSWER SHEET:"
                    ] + teacher_images + [
                        "Analyze the teacher answer sheet above. When I provide a student sheet, compare and evaluate."
                    ]
                    
                    # Create cache with 1 hour TTL
                    cache = genai.caching.CachedContent.create(
                        model='models/gemini-1.5-flash-001',
                        contents=teacher_content,
                        ttl=timedelta(hours=1),
                        display_name=f"teacher_sheet_{st.session_state.teacher_file_hash[:8]}"
                    )
                    st.session_state.cached_content = cache
                    st.info("📦 Created cache for teacher answer sheet")
                    
                except Exception as e:
                    st.warning(f"Could not create cache: {str(e)}")
            
            # Make the regular API call
            response = model.generate_content(content)
        
        end_time = time.time()
        
        if not response:
            st.error("No response received from API")
            return None, None
        
        # Extract token counts
        token_metadata = response.usage_metadata
        input_tokens = token_metadata.prompt_token_count
        output_tokens = token_metadata.candidates_token_count
        total_tokens = token_metadata.total_token_count
        
        # Parse response
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != 0:
            json_str = response_text[json_start:json_end]
            result_data = json.loads(json_str)
        else:
            st.error("Failed to parse JSON response")
            return None, None
        
        # Calculate costs
        input_cost_usd, input_cost_inr = calculate_cost(
            input_tokens, 'input', cached=was_cached, prompt_length=input_tokens
        )
        output_cost_usd, output_cost_inr = calculate_cost(
            output_tokens, 'output', prompt_length=input_tokens
        )
        
        total_cost_usd = input_cost_usd + output_cost_usd
        total_cost_inr = input_cost_inr + output_cost_inr
        
        # Calculate savings if cached
        if was_cached:
            regular_input_cost_usd, regular_input_cost_inr = calculate_cost(
                input_tokens, 'input', cached=False, prompt_length=input_tokens
            )
            savings_usd = regular_input_cost_usd - input_cost_usd
            savings_inr = regular_input_cost_inr - input_cost_inr
        else:
            savings_usd = 0
            savings_inr = 0
        
        # Store token stats
        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Combined Extraction & Evaluation',
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_cost_usd': input_cost_usd,
            'output_cost_usd': output_cost_usd,
            'total_cost_usd': total_cost_usd,
            'input_cost_inr': input_cost_inr,
            'output_cost_inr': output_cost_inr,
            'total_cost_inr': total_cost_inr,
            'cached': was_cached,
            'savings_usd': savings_usd,
            'savings_inr': savings_inr,
            'processing_time': end_time - start_time
        }
        
        st.session_state.token_history.append(stats)
        
        # Store teacher answers for future use
        if 'teacher_answers' in result_data:
            st.session_state.teacher_answers = result_data['teacher_answers']
        
        return result_data, stats
        
    except Exception as e:
        st.error(f"Error in extraction and evaluation: {str(e)}")
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
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return
    
    # Pricing info
    with st.expander("💰 Pricing Information"):
        st.markdown(f"""
        **Gemini 1.5 Flash Pricing (per 1M tokens):**
        - Input: ₹{0.075 * USD_TO_INR:.2f} (≤128k) / ₹{0.15 * USD_TO_INR:.2f} (>128k)
        - Output: ₹{0.30 * USD_TO_INR:.2f} (≤128k) / ₹{0.60 * USD_TO_INR:.2f} (>128k)
        - Cached Input: ₹{0.01875 * USD_TO_INR:.2f} (≤128k) / ₹{0.0375 * USD_TO_INR:.2f} (>128k) **[75% discount!]**
        - Cache Storage: ₹{1.00 * USD_TO_INR:.2f} per hour
        
        **Exchange Rate:** $1 = ₹{USD_TO_INR}
        """)
    
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
                
                # Extract and evaluate in a single call
                result_data, stats = extract_and_evaluate_with_caching(
                    model, teacher_images, student_images, use_cache=use_cache
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
                        if stats['cached']:
                            st.metric("Savings", f"₹{stats['savings_inr']:.4f}")
                        else:
                            st.metric("Cache Status", "Not Cached")
                    
                    # Detailed token breakdown
                    with st.expander("📈 Detailed Token & Cost Breakdown"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Token Counts:**")
                            st.write(f"- Input Tokens: {stats['input_tokens']:,}")
                            st.write(f"- Output Tokens: {stats['output_tokens']:,}")
                            st.write(f"- Total Tokens: {stats['total_tokens']:,}")
                            
                            if stats['cached']:
                                st.success("✅ Used cached content for teacher sheet!")
                                st.write(f"- Regular Input Cost: ₹{stats['input_cost_inr'] + stats['savings_inr']:.4f}")
                                st.write(f"- Cached Input Cost: ₹{stats['input_cost_inr']:.4f}")
                                st.write(f"- **Savings: ₹{stats['savings_inr']:.4f} (75%)**")
                        
                        with col2:
                            st.write("**Cost Breakdown:**")
                            st.write(f"- Input Cost: ₹{stats['input_cost_inr']:.4f} (${stats['input_cost_usd']:.6f})")
                            st.write(f"- Output Cost: ₹{stats['output_cost_inr']:.4f} (${stats['output_cost_usd']:.6f})")
                            st.write(f"- **Total Cost: ₹{stats['total_cost_inr']:.4f} (${stats['total_cost_usd']:.6f})**")
                    
                    # Prepare graded results
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