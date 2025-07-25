from django.shortcuts import render
from .forms import FeedbackForm
from django.conf import settings

print("Settings Base Directory:",settings.BASE_DIR)

# --- Global resources (loaded once) ---
_technical_skills = set()
_soft_skills = set()
_label_role = {}
_role_model = None
_role_embeddings = None
_roles = []



def home(request):
    return render(request, 'home.html')

# --- FEEDBACK VIEW (UNCHANGED) ---
def feedback(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            return render(request, 'feedback.html', {'form': FeedbackForm(), 'success': True})
    else:
        form = FeedbackForm()
    return render(request, 'feedback.html', {'form': form})



