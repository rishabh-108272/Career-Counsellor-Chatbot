from django.shortcuts import render
from django.http import JsonResponse
import os
from django.conf import settings
import uuid
from .skill_analysis import analyze_user_skills
from .parser_core import extract_text_from_pdf, parse_resume_with_gemini, get_gemini_fallback_analysis



# --- resume_upload VIEW (MODIFIED) ---
def resume_upload(request):
    if request.method == 'POST':
        resume_file = request.FILES.get('resume_file')
        if not resume_file:
            return JsonResponse({'error': 'No file was uploaded.'}, status=400)
        
        # Ensure MEDIA_ROOT exists
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)
        
        unique_filename = str(uuid.uuid4()) + '.pdf'
        temp_file_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
        
        try:
            with open(temp_file_path, 'wb+') as destination:
                for chunk in resume_file.chunks():
                    destination.write(chunk)
            
            resume_text = extract_text_from_pdf(temp_file_path)
            if resume_text is None:
                return JsonResponse({'error': 'Could not extract text from the PDF.'}, status=400)
            
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                return JsonResponse({'error': 'GEMINI_API_KEY not configured in settings.'}, status=500)
            
            parsed_data = parse_resume_with_gemini(resume_text, api_key)
            
            if 'error' in parsed_data:
                return JsonResponse({'error': parsed_data['error']}, status=500)
            else:
                # Pass the parsed_data (which now includes 'keywords') to analyze_user_skills
                # The analyze_user_skills function expects 'skills', 'job_title', 'experience'.
                # We can map 'keywords' from the parser to 'skills' for the analyzer.
                # Or, even better, pass the 'keywords' explicitly to the analyzer.
                
                # Let's adjust parsed_data to fit analyze_user_skills expecting 'skills'
                # and ensure 'keywords' are incorporated into the 'skills' field
                
                # The `analyze_user_skills` function already processes `parsed_resume_data.get('skills', [])`
                # If Gemini provides 'skills' and 'keywords', we should combine them for the analyzer.
                
                parsed_skills = parsed_data.get('skills', [])
                extracted_keywords = parsed_data.get('keywords', [])
                
                # Combine and deduplicate skills from both 'skills' and 'keywords' fields
                combined_skills_for_analyzer = list(set(parsed_skills + extracted_keywords))
                
                # Create the input for analyze_user_skills
                analyzer_input = {
                    "skills": combined_skills_for_analyzer,
                    "job_title": parsed_data.get('job_title', ''),
                    "experience": parsed_data.get('experience', '')
                }
                
                analysis_results = analyze_user_skills(analyzer_input)
                
                # Merge analysis results with original parsed data for the response
                final_response_data = {**parsed_data, **analysis_results}
                
                return JsonResponse({'parsed_data': final_response_data})
        except Exception as e:
            return JsonResponse({'error': f'An unexpected server error occurred: {str(e)}'}, status=500)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    return render(request, 'resume_upload.html')