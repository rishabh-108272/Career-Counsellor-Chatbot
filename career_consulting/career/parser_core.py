import json
import requests
import PyPDF2



# --- RESUME PARSER LOGIC (MODIFIED) ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")
        return None
    
def parse_resume_with_gemini(resume_text, api_key):
    if not resume_text:
        return {"error": "No resume text provided for parsing."}
    
    # MODIFIED PROMPT: Ask Gemini to also extract a 'keywords' list for direct skill analysis
    prompt = f"""
    You are a highly accurate resume parsing AI. From the following resume text, extract the 'skills', 'job_title', 'experience', and a comprehensive list of 'keywords'.
    'skills' should be a list of key technical and soft skills. 'job_title' should be the most recent or prominent job title.
    'experience' should be a concise summary of the work experience section.
    'keywords' should be a broad list of all important technical terms, tools, methodologies, and concepts found throughout the entire resume, explicitly excluding personal information like names, contact details, and locations. Focus on terms relevant to professional capabilities and domains.
    Provide the output in a JSON format.

    Resume Text: --- {resume_text} ---
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content", {}).get("parts"):
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            # It's crucial that Gemini returns valid JSON. Add a safety check.
            try:
                parsed_json = json.loads(json_string)
                # Ensure 'keywords' is a list, if not, convert or set empty
                if not isinstance(parsed_json.get('keywords'), list):
                    parsed_json['keywords'] = [] # Default to empty list if not provided or wrong type
                return parsed_json
            except json.JSONDecodeError:
                return {"error": f"Gemini returned invalid JSON: {json_string}"}
        else:
            return {"error": "Unexpected API response structure from Gemini."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during parsing: {e}"}

# --- NEW: Gemini Fallback Analysis Function ---
def get_gemini_fallback_analysis(resume_text, api_key):
    """
    Uses Gemini API to provide a general text-based analysis of the resume
    when the primary skill analysis yields low confidence.
    """
    if not resume_text:
        return "No resume text provided for detailed analysis."

    prompt = f"""
    Given the following resume text, provide a concise, general analysis of the candidate's profile.
    Focus on:
    1.  Overall strengths and key areas of experience.
    2.  Potential career paths or domains that seem suitable.
    3.  Suggestions for areas of improvement or skills to acquire based on common career trajectories.
    Do not mention specific job titles or companies from the resume unless absolutely necessary for context.
    Keep the response professional and encouraging.

    Resume Text:
    ---
    {resume_text}
    ---
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
            return "Gemini could not generate a detailed analysis at this time."
    except requests.exceptions.RequestException as e:
        return f"Failed to get Gemini analysis due to API error: {e}"
    except Exception as e:
        return f"An unexpected error occurred during Gemini analysis: {e}"