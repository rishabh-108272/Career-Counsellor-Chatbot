from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import json
import os
from django.conf import settings
import requests
from googlesearch import search



df_global_fallback = None
try:
    csv_path = os.path.join(settings.BASE_DIR,'career_consulting', 'career', 'skills_data', 'resume_skill_role_analysis.csv')
    if os.path.exists(csv_path):
        df_global_fallback = pd.read_csv(csv_path)
        print("--- Global fallback chatbot CSV loaded successfully. ---")
except Exception as e:
    print(f"--- Could not load global fallback chatbot data: {e} ---")


def get_huggingface_response(prompt):
    """Calls the Hugging Face Inference API and returns the text."""
    api_key = getattr(settings, 'HUGGINGFACE_API_KEY', None)
    if not api_key:
        return "Hugging Face API key is not configured on the server."

    # Using the same model you had in your original HTML
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "inputs": prompt,
        "parameters": { "max_new_tokens": 250 } # Limit the response length
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        
        # Extract the generated text, removing the original prompt from the response
        if result and isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].replace(prompt, "").strip()
        else:
            return "Received an unexpected response format from Hugging Face."
            
    except requests.exceptions.RequestException as e:
        # This will catch connection errors, timeouts, etc.
        print(f"Hugging Face API Error: {e}")
        return "Sorry, there was an issue contacting the Hugging Face model. It might be loading or unavailable."


def get_gemini_recommendations(user_query, item_type=None, api_key=None):
    """
    Uses Gemini API to provide general job/course recommendations based on a query.
    This serves as a fallback or a general overview.
    """
    if not api_key:
        return "Gemini API key is not configured for fallback recommendations."

    prompt_suffix = ""
    if item_type == 'job':
        prompt_suffix = " focusing specifically on job roles."
    elif item_type == 'course':
        prompt_suffix = " focusing specifically on relevant courses or learning paths."

    prompt = f"""
    Based on the query: "{user_query}", provide a concise list of 3-5 relevant career recommendations or courses{prompt_suffix}.
    For each recommendation, include a brief reason why it's relevant.
    Format your response as a bulleted list.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]} # No responseMimeType for free-form text
    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content", {}).get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Gemini could not generate recommendations at this time."
    except requests.exceptions.RequestException as e:
        return f"Failed to get Gemini recommendations due to API error: {e}"
    except Exception as e:
        return f"An unexpected error occurred during Gemini recommendation: {e}"

# --- CHATBOT VIEW (MODIFIED TO USE SESSION DATA) ---
# --- CHATBOT VIEW (CORRECTED TO DISPLAY SKILLS) ---
def chatbot(request):
    if request.method != 'POST':
        return render(request, 'chatbot.html')

    user_analysis_results = request.session.get('user_analysis_results', None)

    if user_analysis_results is None:
        return JsonResponse({"response": "Please upload and analyze your resume first on the Skill Analysis page to get personalized recommendations."})
    
    # Extract the necessary data for the chatbot's display
    current_job_title = user_analysis_results.get('user_job_title', 'N/A')
    matched_role = user_analysis_results.get('matched_role', 'unknown')
    matched_domain = user_analysis_results.get('matched_domain', 'unknown')
    similarity_score = user_analysis_results.get('similarity_score', 0.0)
    extracted_tech_skills = user_analysis_results.get('technical_skills', [])
    extracted_soft_skills = user_analysis_results.get('soft_skills', [])
    user_experience_summary = user_analysis_results.get('user_experience_summary', 'N/A')
    
    
    # Helper function modified to use the direct variables
    def format_current_recommendation():
        return (f"<strong>Job Title:</strong> {current_job_title}<br>"
                f"<strong>Matched Role:</strong> {matched_role}<br>"
                f"<strong>Domain:</strong> {matched_domain}<br>"
                f"<strong>Similarity Score:</strong> {similarity_score:.2f}<br><br>"
                "You can ask: 'Why this role?', 'Show me courses', or 'Next recommendation'.")

    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip().lower()
        model_choice = data.get("model_choice", "gemini")

        # Handle initial greeting
        if user_message in ["start", "hi", "hello", "hey"]:
            # For this simplified approach, the 'start' will always present
            # the analysis of the *most recently uploaded resume*.
            # We remove current_index and rejected_indices as we're not iterating a pre-made list in this way.
            # If you want 'next' to show *other* jobs from qa_data_df, that's a separate logic.
            return JsonResponse({
                "response": "Hi! Based on your most recent resume analysis, here is your primary recommended role:",
                "recommendation": format_current_recommendation()
            })

        # --- THIS IS THE CORRECTED SECTION FOR "WHY THIS ROLE?" ---
        if "why" in user_message:
            all_extracted_skills = list(set(extracted_tech_skills + extracted_soft_skills)) # Combine and deduplicate
            
            if all_extracted_skills:
                skills_text = "<br>".join([f"• {skill}" for skill in all_extracted_skills])
                response_text = (
                    "You were recommended this role because of the high similarity score and based on these key skills extracted from your resume:<br><br>"
                    f"<strong>Key Skills:</strong><br>{skills_text}"
                )
            else:
                response_text = "You were recommended this role because of the high similarity score. No specific skills were clearly extracted from your resume for this analysis."

            return JsonResponse({"response": response_text})

        # --- Handle 'next' to potentially suggest *other* roles/courses from qa_data_df ---
        # This part requires a strategy: do you want 'next' to iterate through *similar* jobs
        # from your QA data, or just signal no more direct recommendations from *this* resume?
        # For simplicity, let's make 'next' trigger a general recommendation if no more resume-specific ones.
        if any(word in user_message for word in ["next", "another", "not satisfied"]):
            # If you want to iterate through 'qa_data_df' for more recommendations:
            # You would need to store an index into qa_data_df in the session.
            # For now, let's just offer to ask Gemini for more.
            prompt_for_gemini = f"Given my resume's primary matched role is '{matched_role}' and skills like {', '.join(extracted_tech_skills + extracted_soft_skills)}, suggest a few other relevant career paths or learning areas."
            
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                return JsonResponse({"response": "Sorry, Gemini API key is not configured for providing more recommendations."})
            
            gemini_suggestions = get_gemini_recommendations(prompt_for_gemini, api_key=api_key)
            return JsonResponse({
                "response": "Understood. Here are some other suggestions I found for you based on your profile:",
                "recommendation": gemini_suggestions # Gemini's response is already formatted for display
            })


        # Handle 'courses'
        if "courses" in user_message or "improve" in user_message:
            try:
                target_role_for_courses = matched_role
                if target_role_for_courses == 'unknown' or not target_role_for_courses:
                    query = "online career development courses"
                else:
                    query = f"online courses for {target_role_for_courses}"
                
                course_links = list(search(query, num_results=3))
                links_html = "<br>".join([f"• <a href='{link}' target='_blank' rel='noopener noreferrer'>{link}</a>" for link in course_links])
                return JsonResponse({"response": f"Here are some top course search results that might help:<br><br>{links_html}"})
            except Exception as e:
                return JsonResponse({"response": f"Sorry, I had trouble finding courses right now. Error: {e}"})

        # Handle all other general questions with the selected AI model
        else:
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                return JsonResponse({"response": "Sorry, Gemini API key is not configured for general questions."})

            prompt_for_llm = f"You are a helpful and concise career advice chatbot. A user has asked the following question: '{user_message}'. Provide a helpful, bulleted response."
            
            ai_response = ""
            if model_choice == 'huggingface':
                print(f"Chatbot handling query with Hugging Face: '{user_message}'")
                ai_response = get_huggingface_response(prompt_for_llm)
            else: # Default to Gemini
                print(f"Chatbot handling query with Gemini: '{user_message}'")
                # For general questions, you need to use a model that can respond to any prompt.
                # Assuming you have a `gemini_model` configured for chat.
                # If not, you might repurpose `get_gemini_recommendations` for broader answers.
                # For truly general chat, you'd call Gemini's generate_content directly with `prompt_for_llm`.
                try:
                    # IMPORTANT: You'll need to ensure your `gemini_model` instance is loaded and accessible here.
                    # It's not defined in the provided snippet. If you have it globally, ensure it's loaded.
                    # Example assuming `gemini_model` is a loaded instance of Gemini:
                    # from google.generativeai import GenerativeModel
                    # gemini_model = GenerativeModel("gemini-1.5-flash-latest") # Loaded globally or passed in
                    
                    # For a simple text response to general questions:
                    response = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"contents": [{"parts": [{"text": prompt_for_llm}]}]})
                    )
                    response.raise_for_status()
                    result = response.json()
                    ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    ai_response = f"Sorry, I couldn't connect to the Gemini model for this general question. Error: {e}"

            return JsonResponse({"response": ai_response})

    except Exception as e:
        print(f"FATAL Error in chatbot view: {e}")
        return JsonResponse({"response": f"A server error occurred: {str(e)}"}, status=500)