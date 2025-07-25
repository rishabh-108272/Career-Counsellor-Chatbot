from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import json
import os
from django.conf import settings
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch


QA_DATASET_PATH = os.path.join(settings.BASE_DIR,'career_consulting', 'career', 'qa_data')
QA_MODEL_PATH = os.path.join(settings.BASE_DIR,'career_consulting', 'career', 'qa_model')
FEEDBACK_LOG_PATH = os.path.join(settings.BASE_DIR, 'feedback_log_qa.jsonl')
qa_pipeline = None
embedding_model = None
faiss_index = None
all_context_metadata = []
all_contexts_for_faiss = []
gemini_model = None

def _load_qa_rag_resources():
    """
    Loads all necessary data and models for the RAG QA system.
    This is a one-time process when the Django server starts.
    """
    global qa_pipeline, embedding_model, faiss_index, all_context_metadata, all_contexts_for_faiss, gemini_model

    # Prevent re-loading if already loaded
    if qa_pipeline and faiss_index:
        return

    print("--- Loading QA RAG resources (this will run once on server start) ---")

    # 1. --- Configure Gemini API ---
    try:
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            print("--- QA RAG WARNING: GEMINI_API_KEY not found in settings. Gemini features will be disabled. ---")
        else:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            # Test connection
            gemini_model.generate_content("hello")
            print("Gemini API configured successfully for QA recommender.")
    except Exception as e:
        print(f"--- QA RAG ERROR: Could not configure Gemini API: {e} ---")
        gemini_model = None

    # 2. --- Load Datasets (Jobs and Courses) ---
    try:
        jobs_df = pd.read_csv(os.path.join(QA_DATASET_PATH, "job_skills.csv"))
        courses_df = pd.read_csv(os.path.join(QA_DATASET_PATH, "coursera_data.csv"))
        print("Job and Course CSV datasets loaded successfully.")
    except Exception as e:
        print(f"--- QA RAG ERROR: Could not load dataset files: {e}. The recommender will not work. ---")
        return # Stop loading if data is missing

    # 3. --- Initialize Embedding Model ---
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer embedding model loaded successfully.")
    except Exception as e:
        print(f"--- QA RAG ERROR: Could not load SentenceTransformer model: {e}. The recommender will not work. ---")
        return

    # 4. --- Create Contexts for FAISS ---
    def create_context_entry(row, context_type):
        text_parts = []
        if context_type == "job":
            text_parts.append(f"Job Title: {row.get('Title', '')}")
            text_parts.append(f"Company: {row.get('Company', '')}")
            text_parts.append(f"Responsibilities: {row.get('Responsibilities', '')}")
            text_parts.append(f"Minimum Qualifications: {row.get('Minimum Qualifications', '')}")
        elif context_type == "course":
            text_parts.append(f"Course Title: {row.get('course_title', '')}")
            text_parts.append(f"Organization: {row.get('course_organization', '')}")
            text_parts.append(f"Difficulty: {row.get('course_difficulty', '')}")
        
        text = ". ".join(filter(None, text_parts))
        if text.strip():
            return text, {"type": context_type, "original_data": row.to_dict()}
        return None, None

    print("Building contexts for FAISS index...")
    for _, row in jobs_df.iterrows():
        context_text, metadata = create_context_entry(row, "job")
        if context_text:
            all_contexts_for_faiss.append(context_text)
            all_context_metadata.append(metadata)

    for _, row in courses_df.iterrows():
        context_text, metadata = create_context_entry(row, "course")
        if context_text:
            all_contexts_for_faiss.append(context_text)
            all_context_metadata.append(metadata)

    if not all_contexts_for_faiss:
        print("--- QA RAG ERROR: No contexts were generated for FAISS. Check data processing. ---")
        return

    # 5. --- Build FAISS Index ---
    print(f"Encoding {len(all_contexts_for_faiss)} contexts for FAISS index...")
    context_embeddings = embedding_model.encode(all_contexts_for_faiss, show_progress_bar=True, convert_to_numpy=True).astype('float32')
    dimension = context_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(context_embeddings)
    print("FAISS index built successfully.")

    # 6. --- Load the Fine-tuned QA Model ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
        model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Fine-tuned QA model loaded successfully.")
    except Exception as e:
        print(f"--- QA RAG ERROR: Could not load fine-tuned QA model from '{settings.QA_MODEL_PATH}': {e} ---")
        qa_pipeline = None # Mark as unusable

# Load all resources when Django starts
_load_qa_rag_resources()


def get_qa_recommendations_with_rag(query: str, k: int = 5, item_type: str = None, num_retrieved_contexts: int = 10):
    """
    The core RAG function, adapted from the Colab script.
    It retrieves contexts using FAISS and then uses the fine-tuned QA model.
    """
    if not all([qa_pipeline, embedding_model, faiss_index]):
        return [], "The QA recommendation system is not available. Please check server logs."

    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    # Search the FAISS index
    distances, indices = faiss_index.search(query_embedding, num_retrieved_contexts)

    retrieved_qa_results = []
    for idx in indices[0]:
        if idx == -1: continue # Skip invalid indices

        item_metadata = all_context_metadata[idx]
        
        # Filter by item type if specified by the user
        if item_type and item_type != 'all' and item_metadata['type'] != item_type:
            continue
        
        context_text = all_contexts_for_faiss[idx]
        
        # Run the fine-tuned QA pipeline on the retrieved context
        qa_result = qa_pipeline(question=query, context=context_text)

        # Only include results with a reasonable confidence score
        if qa_result['score'] > 0.05 and qa_result['answer'].strip():
            retrieved_qa_results.append({
                "score": qa_result['score'],
                "answer": qa_result['answer'],
                "context_type": item_metadata["type"],
                "original_data": item_metadata["original_data"],
            })

    # Sort results by the QA model's confidence score
    retrieved_qa_results.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate results to avoid showing the same job/course multiple times
    final_recommendations = []
    seen_ids = set()
    for res in retrieved_qa_results:
        unique_id = None
        if res['context_type'] == 'job':
            unique_id = res['original_data'].get('Title', '') + res['original_data'].get('Company', '')
        elif res['context_type'] == 'course':
            unique_id = res['original_data'].get('course_title', '')
        
        if unique_id and unique_id not in seen_ids:
            final_recommendations.append(res)
            seen_ids.add(unique_id)
            if len(final_recommendations) >= k:
                break
    
    # Generate a summary using Gemini (instead of a local Ollama model)
    llm_summary = ""
    if gemini_model and final_recommendations:
        prompt_context = ""
        for i, rec in enumerate(final_recommendations):
            prompt_context += f"Recommendation {i+1} ({rec['context_type']}):\n"
            if rec['context_type'] == 'job':
                prompt_context += f"  - Title: {rec['original_data'].get('Title')}\n"
                prompt_context += f"  - Key Info: {rec['answer']}\n"
            else:
                prompt_context += f"  - Title: {rec['original_data'].get('course_title')}\n"
                prompt_context += f"  - Key Info: {rec['answer']}\n"
        
        summary_prompt = f"User asked: '{query}'. Based on these specific recommendations, write a short, encouraging summary explaining why these are a good fit and suggest a potential next step.\n\n{prompt_context}"
        try:
            response = gemini_model.generate_content(summary_prompt)
            llm_summary = response.text
        except Exception as e:
            llm_summary = f"Could not generate summary due to Gemini API error: {e}"

    return final_recommendations, llm_summary


def get_gemini_fallback_recommendations(query: str, item_type: str = None):
    """
    This function is called when the user is not satisfied with the local results.
    It uses Gemini for a more general, creative set of recommendations.
    """
    if not gemini_model:
        return "Gemini Fallback is not available. Please check server configuration."

    prompt = (
        f"The user is looking for recommendations related to: '{query}'.\n"
        f"The initial automated results were not satisfactory. Please provide a fresh, helpful set of recommendations for {item_type if item_type else 'jobs and courses'}. "
        f"Suggest specific job titles or course names and explain why they are a good match. Be actionable and encouraging."
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"


# --- This is your existing function for logging, no changes needed here ---
def log_feedback(query, source, feedback_text, qa_results, llm_response):
    """Logs the interaction and feedback to a file for later analysis."""
    feedback_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": query,
        "source": source,
        "feedback": feedback_text,
        "qa_results_preview": [{"answer": r.get('answer'), "score": r.get('score')} for r in qa_results[:3]],
        "llm_response_preview": llm_response[:300] + "..." if llm_response else "",
    }
    try:
        with open(FEEDBACK_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')
    except Exception as e:
        print(f"Error logging feedback: {e}")


# --- NEW: Add this new view to handle feedback from the buttons ---
def log_user_feedback(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        query = data.get('query')
        feedback = data.get('feedback') # e.g., "Helpful" or "Not Helpful"

        if not query or not feedback:
            return JsonResponse({'error': 'Missing query or feedback text'}, status=400)

        # Log this specific user feedback event.
        # We can use the existing log_feedback function with empty results.
        log_feedback(
            query=query,
            source="User_Feedback_Button",
            feedback_text=feedback,
            qa_results=[], # No QA results for this specific event
            llm_response="" # No LLM response for this event
        )
        
        return JsonResponse({'status': 'success', 'message': 'Feedback logged.'})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        print(f"Error in log_user_feedback view: {e}")
        return JsonResponse({'error': f'An unexpected server error occurred: {str(e)}'}, status=500)


def job_recommender(request):
    """Renders the main page for the job recommender."""
    auto_query = request.GET.get('query', '')
    context = {'auto_query': auto_query}
    return render(request, 'job_recommender.html', context)


def ask_question(request):
    """
    This is the main API endpoint for the QA recommender.
    It handles both initial requests and Gemini fallback requests.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'This endpoint only supports POST requests.'}, status=405)

    if not all([qa_pipeline, embedding_model, faiss_index]):
        return JsonResponse({'error': 'The QA system is not fully loaded. Please check server logs.'}, status=503)

    try:
        data = json.loads(request.body)
        question = data.get('question')
        item_type = data.get('item_type', 'all')
        # This flag determines if we should use the local RAG or go straight to Gemini
        use_gemini_fallback = data.get('use_gemini_fallback', False)

        if not question:
            return JsonResponse({'error': 'No question provided.'}, status=400)

        recommendations = []
        llm_summary = ""
        gemini_fallback_output = None
        source = ""

        if use_gemini_fallback:
            print(f"Requesting Gemini Fallback for query: '{question}'")
            source = "Gemini_Fallback"
            gemini_fallback_output = get_gemini_fallback_recommendations(question, item_type=item_type)
            log_feedback(question, source, "User requested Gemini", [], gemini_fallback_output)
        else:
            print(f"Requesting Local RAG for query: '{question}'")
            source = "Local_RAG"
            recommendations, llm_summary = get_qa_recommendations_with_rag(question, k=5, item_type=item_type)
            log_feedback(question, source, "Initial request", recommendations, llm_summary)

        return JsonResponse({
            'status': 'success',
            'recommendations': recommendations,        # Results from local RAG
            'llm_summary': llm_summary,              # Summary of local RAG results
            'gemini_fallback_output': gemini_fallback_output, # Results from Gemini fallback
            'source': source
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        print(f"Error in ask_question view: {e}")
        return JsonResponse({'error': f'An unexpected server error occurred: {str(e)}'}, status=500)