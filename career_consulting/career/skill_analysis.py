from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import json
import os
from django.conf import settings
import tempfile
import re
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import collections # For Counter
import seaborn as sns
import base64 # To encode images for embedding in HTML
from io import BytesIO # To save image to memory
from .parser_core import extract_text_from_pdf, parse_resume_with_gemini, get_gemini_fallback_analysis
from .views import _technical_skills,_soft_skills,_label_role,_role_embeddings,_role_model,_roles

SKILLS_JSON_PATH = os.path.join(settings.BASE_DIR,'career_consulting', 'career', 'skills_data', 'skills.json')
LABEL_ROLE_JSON_PATH = os.path.join(settings.BASE_DIR,'career_consulting', 'career', 'skills_data', 'label_role.json')


def _load_analyzer_resources():
    """Loads all necessary data and models for the skill analyzer.
        Called only once.
    """
    # Declare all global variables that will be MODIFIED within this function
    global _technical_skills, _soft_skills, _label_role, _role_model, _role_embeddings, _roles

    if _role_model is not None: # Already loaded
        return

    print("Loading skill analyzer resources...")
    try:
        with open(SKILLS_JSON_PATH) as f:
            skills_data = json.load(f)
        _technical_skills = set(skills_data.get("technical_skills", []))
        _soft_skills = set(skills_data.get("soft_skills", []))
        print(f"Loaded {len(_technical_skills)} technical and {len(_soft_skills)} soft skills.")

        with open(LABEL_ROLE_JSON_PATH) as f:
            _label_role = json.load(f)
        _roles = list(_label_role.keys())
        print(f"Loaded {len(_roles)} roles for matching.")

        _role_model = SentenceTransformer("all-MiniLM-L6-v2")
        _role_embeddings = _role_model.encode(_roles, convert_to_tensor=True)
        print("SentenceTransformer model for role matching loaded.")
    except FileNotFoundError as e:
        print(f"Error: Analyzer resource file not found: {e}. Please ensure '{SKILLS_JSON_PATH}' and '{LABEL_ROLE_JSON_PATH}' exist.")
        # Mark the model as unusable if files are not found
        _role_model = None # This assignment now correctly modifies the global _role_model
    except Exception as e:
        print(f"An unexpected error occurred while loading analyzer resources: {e}")
        # Mark the model as unusable if an unexpected error occurs
        _role_model = None # This assignment now correctly modifies the global _role_model

# Ensure resources are loaded when the Django app starts
_load_analyzer_resources()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b(summary|objective|experience|education|skills|projects|awards)\b[:\-]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,4}(\s*[-\/]\s*\d{1,4})?', '', text)  # remove years/durations like 2018-2022, 10
    text = re.sub(r'[^\w\s\.\,\-\/\#\+\&]', '', text) # remove most non-alphanumeric except common punctuation
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()


# Skill extraction function
def extract_skills_from_text(text):
    """
    Extracts technical and soft skills from a given text.
    Assumes _technical_skills and _soft_skills are loaded globally.
    """
    # _load_analyzer_resources() # No need to call here, as it's called once at module load
    found_tech = []
    found_soft = []
    if isinstance(text, str):
        text_lower = text.lower()
        for skill in _technical_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_tech.append(skill)
        for skill in _soft_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_soft.append(skill)
    return found_tech, found_soft


# --- Plotting Functions for Dashboard/Analytics ---

def plot_user_profile_dashboard(analysis_results: dict):
    """
    Plots a simple dashboard for the user's skill profile analysis.
    Generates base64 encoded strings of pie chart and bar plot.

    Args:
        analysis_results (dict): The dictionary containing analysis results from analyze_user_skills.

    Returns:
        tuple: A tuple containing (skill_distribution_plot_base64, top_skills_plot_base64)
               which are base64 encoded strings of the plots, or (None, None) if no data.
    """
    matched_role = analysis_results['Matched Role']
    sim_score = analysis_results['Similarity Score']
    tech_skills = analysis_results['Technical Skills']
    soft_skills = analysis_results['Soft Skills']

    # --- Pie Chart: Skill Type Distribution ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    skill_type_counts = {'Technical': len(tech_skills), 'Soft': len(soft_skills)}
    
    if sum(skill_type_counts.values()) > 0:
        ax1.pie(skill_type_counts.values(), labels=skill_type_counts.keys(), autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
        ax1.set_title("Distribution of Extracted Skill Types")
    else:
        ax1.text(0.5, 0.5, "No skills extracted", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title("Distribution of Extracted Skill Types")
        ax1.axis('off') # Hide axes for empty plot
    
    plt.tight_layout()
    pie_chart_buffer = BytesIO()
    plt.savefig(pie_chart_buffer, format='png')
    pie_chart_buffer.seek(0)
    pie_chart_base64 = base64.b64encode(pie_chart_buffer.getvalue()).decode('utf-8')
    plt.close(fig1) # Close the figure to free up memory

    # --- Bar Plot: Top N Technical and Soft Skills ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    all_extracted_skills = collections.Counter(tech_skills + soft_skills)
    top_n = 10 # Display top N skills

    if all_extracted_skills:
        # Sort skills by count for consistent plotting
        top_skills_df = pd.DataFrame(all_extracted_skills.most_common(top_n), columns=['Skill', 'Count'])
        sns.barplot(x='Count', y='Skill', data=top_skills_df, ax=ax2, palette='viridis')
        ax2.set_title(f"Top {top_n} Most Frequent Extracted Skills")
        ax2.set_xlabel("Frequency (in resume context)")
        ax2.set_ylabel("Skill")
    else:
        ax2.text(0.5, 0.5, "No specific skills to plot", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title(f"Top {top_n} Most Frequent Extracted Skills")
        ax2.axis('off') # Hide axes for empty plot

    plt.tight_layout()
    bar_chart_buffer = BytesIO()
    plt.savefig(bar_chart_buffer, format='png')
    bar_chart_buffer.seek(0)
    bar_chart_base64 = base64.b64encode(bar_chart_buffer.getvalue()).decode('utf-8')
    plt.close(fig2) # Close the figure to free up memory

    return pie_chart_base64, bar_chart_base64

# Role/domain matching function
def match_role_from_text(text):
    """
    Matches the input text to a predefined role and domain.
    Assumes _role_model, _role_embeddings, _roles, _label_role are loaded globally.
    """
    # _load_analyzer_resources() # No need to call here, as it's called once at module load
    if _role_model is None: # Check if model loaded successfully
        return "unknown", "unknown", 0.0

    if not text or str(text).strip() == "":
        return "unknown", "unknown", 0.0

    resume_embedding = _role_model.encode(str(text), convert_to_tensor=True)
    scores = util.cos_sim(resume_embedding, _role_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    matched_role = _roles[best_idx]
    matched_domain = _label_role.get(matched_role, "unknown").lower()

    # Set a confidence threshold for role matching
    if best_score < 0.4: # Adjust this threshold based on your data and desired precision
        return "unknown", "unknown", round(best_score, 3)
    return matched_role, matched_domain, round(best_score, 3)


def analyze_user_skills(parsed_resume_data: dict):
    """
    Analyzes a user's parsed resume data to extract skills and match roles.

    Args:
        parsed_resume_data (dict): A dictionary containing parsed resume information,
                                   e.g., {"skills": ["Python", "SQL"], "job_title": "Data Analyst", "experience": "Summary of work."}
                                   The 'skills' key can be a list or a string.
                                   The 'experience' key is used for overall role matching.

    Returns:
        dict: A dictionary containing:
            - Matched Role (str)
            - Matched Domain (str)
            - Similarity Score (float)
            - Technical Skills (list of str)
            - Soft Skills (list of str)
            - User Job Title (str, from input)
            - User Experience Summary (str, from input)
    """
    # _load_analyzer_resources() # No need to call here, as it's called once at module load
    if _role_model is None: # Ensure the model loaded, otherwise return empty results
        return {
            "Matched Role": "unknown",
            "Matched Domain": "unknown",
            "Similarity Score": 0.0,
            "Technical Skills": [],
            "Soft Skills": [],
            "User Job Title": parsed_resume_data.get('job_title', ''),
            "User Experience Summary": parsed_resume_data.get('experience', '')
        }

    user_job_title = parsed_resume_data.get('job_title', '')
    user_experience_summary = parsed_resume_data.get('experience', '')
    user_skills_from_parser = parsed_resume_data.get('skills', [])

    # Combine relevant text for role matching (job title + experience summary)
    # Clean the text before feeding to role matcher
    text_for_role_matching = clean_text(f"{user_job_title} {user_experience_summary}")

    matched_role, matched_domain, sim_score = match_role_from_text(text_for_role_matching)

    # Clean the raw skills text from the parser for skill extraction
    # The 'skills' from the parser is already a list, so combine it into a string for `extract_skills_from_text`
    if isinstance(user_skills_from_parser, list):
        skills_text_for_extraction = " ".join(user_skills_from_parser)
    else: # If it's a string, use it directly
        skills_text_for_extraction = str(user_skills_from_parser)

    tech_skills, soft_skills_found = extract_skills_from_text(skills_text_for_extraction)

    # Also include skills found from the experience summary itself for a more comprehensive list
    tech_skills_exp, soft_skills_exp = extract_skills_from_text(clean_text(user_experience_summary))

    # Combine and deduplicate skills
    combined_tech_skills = list(set(tech_skills + tech_skills_exp))
    combined_soft_skills = list(set(soft_skills_found + soft_skills_exp))

    return {
        "Matched Role": matched_role,
        "Matched Domain": matched_domain,
        "Similarity Score": sim_score,
        "Technical Skills": combined_tech_skills, # Now returns lists
        "Soft Skills": combined_soft_skills,     # Now returns lists
        "User Job Title": user_job_title,
        "User Experience Summary": user_experience_summary
    }

# --- skill_analysis VIEW (MODIFIED TO USE analyze_user_skills AND FALLBACK) ---
def skill_analysis(request):
    # Check if the skill analysis models loaded correctly
    if _role_model is None:
        return JsonResponse({'status': 'error', 'message': 'Skill analysis model resources are not available. Please check server logs.'}, status=503)
        
    context = {} # Context will be populated with analysis results
    resume_text = "" # Initialize resume_text here

    if request.method == 'POST' and request.FILES.get('resume'):
        try:
            uploaded_file = request.FILES['resume']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            resume_text = extract_text_from_pdf(tmp_path) # Assign extracted text to resume_text
            os.unlink(tmp_path) # Clean up the temporary file

            if resume_text is None:
                raise ValueError("Could not extract text from the PDF.")

            # Call Gemini parser to get structured data
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                raise Exception('GEMINI_API_KEY not configured in settings.')

            parsed_data = parse_resume_with_gemini(resume_text, api_key)
            if 'error' in parsed_data:
                raise Exception(parsed_data['error'])

            # Combine 'skills' and 'keywords' from parsed_data for the analyzer input
            parsed_skills = parsed_data.get('skills', [])
            extracted_keywords = parsed_data.get('keywords', [])
            combined_skills_for_analyzer = list(set(parsed_skills + extracted_keywords))

            analyzer_input = {
                "skills": combined_skills_for_analyzer, # This now includes keywords from Gemini
                "job_title": parsed_data.get('job_title', ''),
                "experience": parsed_data.get('experience', '')
            }
            
            # Use the comprehensive analyze_user_skills function
            analysis_results = analyze_user_skills(analyzer_input)
            request.session['user_analysis_results'] = {
                'matched_role': analysis_results['Matched Role'],
                'matched_domain': analysis_results['Matched Domain'],
                'similarity_score': analysis_results['Similarity Score'], # Keep as float for consistency
                'technical_skills': analysis_results['Technical Skills'],
                'soft_skills': analysis_results['Soft Skills'],
                'user_job_title': analysis_results['User Job Title'],
                'user_experience_summary': analysis_results['User Experience Summary']
            }
            print("--- SUCCESS: User's analysis results saved to session. ---")


            # --- Conditional Gemini Fallback Analysis ---
            gemini_fallback_analysis = None
            # Check if role/domain could not be confidently matched OR if no skills were extracted
            if analysis_results['Matched Role'] == "unknown" or \
               (not analysis_results['Technical Skills'] and not analysis_results['Soft Skills']):
                print("Primary analysis low confidence or no skills found. Requesting Gemini fallback analysis...")
                gemini_fallback_analysis = get_gemini_fallback_analysis(resume_text, api_key)
                context['gemini_fallback_analysis'] = gemini_fallback_analysis
            else:
                context['gemini_fallback_analysis'] = None # Ensure it's explicitly None if not used

            # Generate plots
            pie_chart_base64, bar_chart_base64 = plot_user_profile_dashboard(analysis_results)

            # --- Generate automatic prompt for job_recommender ---
            auto_query_parts = []
            if analysis_results['Matched Role'] != "unknown" and analysis_results['Matched Role']:
                auto_query_parts.append(f"recommendations for a {analysis_results['Matched Role']}")
                if analysis_results['Matched Domain'] != "unknown" and analysis_results['Matched Domain']:
                    auto_query_parts.append(f"in the {analysis_results['Matched Domain']} domain")
            else:
                auto_query_parts.append("career recommendations")

            if analysis_results['Technical Skills']:
                auto_query_parts.append(f"with technical skills like {', '.join(analysis_results['Technical Skills'])}")
            if analysis_results['Soft Skills']:
                auto_query_parts.append(f"and soft skills such as {', '.join(analysis_results['Soft Skills'])}")

            # Fallback if no specific role or skills, use general experience
            if not auto_query_parts or (len(auto_query_parts) == 1 and "career recommendations" in auto_query_parts[0]):
                if analysis_results['User Experience Summary']:
                    # Truncate summary for prompt brevity
                    summary_preview = analysis_results['User Experience Summary']
                    if len(summary_preview) > 150:
                        summary_preview = summary_preview[:150] + "..."
                    auto_query_parts.append(f"based on experience: {summary_preview}")
                else:
                    auto_query_parts = ["general career recommendations"] # Fallback if no info at all

            auto_prompt = "Find " + " ".join(auto_query_parts) + "."
            context['auto_prompt'] = auto_prompt # Add to context

            # Prepare context for rendering or JSON response
            context.update({
                'analysis_results': {
                    'matched_role': analysis_results['Matched Role'],
                    'matched_domain': analysis_results['Matched Domain'],
                    'similarity_percentage': analysis_results['Similarity Score'] * 100,
                    'technical_skills': analysis_results['Technical Skills'],
                    'soft_skills': analysis_results['Soft Skills'],
                    'user_job_title': analysis_results['User Job Title'],
                    'user_experience_summary': analysis_results['User Experience Summary']
                },
                'accuracy': "High" if analysis_results['Similarity Score'] >= 0.7 else "Medium" if analysis_results['Similarity Score'] >= 0.4 else "Low",
                'success': True,
                # Add plot data to context
                'pie_chart': pie_chart_base64,
                'bar_chart': bar_chart_base64,
            })

            # For AJAX requests, you might want to send plot URLs/data differently
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'success',
                    'analysis_results': context['analysis_results'],
                    'accuracy': context['accuracy'],
                    'pie_chart_data': pie_chart_base64, # Send base64 data to AJAX
                    'bar_chart_data': bar_chart_base64, # Send base64 data to AJAX
                    'gemini_fallback_analysis': gemini_fallback_analysis, # Send fallback analysis to AJAX
                    'auto_prompt': auto_prompt # Send auto_prompt to AJAX
                })
            return render(request, 'skill_analysis.html', context)
        
        except Exception as e:
            context['error'] = str(e)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
            return render(request, 'skill_analysis.html', context)
    
    # If not a POST request or no resume file, just render the empty form
    return render(request, 'skill_analysis.html', context)