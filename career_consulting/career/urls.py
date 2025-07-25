from django.urls import path
from . import views
from . import resume_parser
from . import skill_analysis
from . import career_recommender
from . import chatbot
# This app_name is important for the {% url ... %} tags in your templates
app_name = 'career'

urlpatterns = [
    # This will be your home page
    path('', views.home, name='home'),
    
    # The rest of your pages (unchanged)
    path('resume-upload/', resume_parser.resume_upload, name='resume_upload'),
    path('skill-analysis/', skill_analysis.skill_analysis, name='skill_analysis'),
    path('job-recommender/', career_recommender.job_recommender, name='job_recommender'),
    path('log-feedback/', career_recommender.log_user_feedback, name='log_user_feedback'),
    path('feedback/', views.feedback, name='feedback'),
    path('chatbot/', chatbot.chatbot, name='chatbot'),
    path('ask-question/', career_recommender.ask_question, name='ask_question'),
]
