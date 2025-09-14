# Pathwise: AI Powered Career Buddy

This is a Django web application that provides career conselling services leveraging AI models. It offers a comprehensive suite of tools for individuals seeking to understand their professional profile and explore career opportunities.

## Features

-   **Resume Upload & Parsing:** Users can upload PDF resumes, from which the system extracts structured data such as skills, primary job title, and a summary of experience using the Google Gemini API.
-   **Skill Analysis:** The extracted information is analyzed to match the user's profile to predefined career roles and domains. It visualizes the distribution of technical and soft skills and provides a similarity score for the matched role. Includes a fallback analysis using Gemini for broader insights if initial analysis is low confidence.
-   **Job & Course Recommender (RAG System):** A Retrieval-Augmented Generation (RAG) system provides personalized job and course recommendations based on user queries. It utilizes FAISS for efficient retrieval of relevant contexts from a curated dataset, a fine-tuned Question Answering (QA) model to extract specific answers, and Google Gemini for summarizing recommendations and providing a general fallback.
-   **AI Chatbot:** An interactive chatbot offers career advice, answers general queries, and provides specific recommendations. It integrates with Google Gemini and Hugging Face models, and can retrieve real-time course links via Google Search.
-   **User Feedback System:** A mechanism to log user feedback on recommendations and interactions, aiding in continuous improvement of the system.

## Setup Instructions

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/).
* **pip**: Python's package installer (usually comes with Python).
* **Git**: For cloning the repository. Download from [git-scm.com](https://git-scm.com/downloads).

### 2. Clone the Repository

Open your command prompt and clone the repository:

```bash
git clone https://github.com/rishabh-108272/Career-Counsellor-Chatbot.git
```

### 3. Navigate to Project Root

Your Django project is likely nested. Navigate into the main project directory where `manage.py` is located:

```bash
cd Career-Counsellor-Chatbot
```

### 4. Set up Virtual Environment

It is highly recommended to create and activate a Python virtual environment to manage project dependencies.

```bash 
python -m venv venv
```

### Activate the virtual environment:

On Windows (CMD or PowerShell):

```bash 
.\venv\Scripts\activate
```

On Linux\MacOS\Git Bash:

```bash 
source venv/bin/activate
```

### 5. Install the dependencies

```bash 
pip install -r requirements.txt
```

### 6. Obtain and Configure API Keys

This project relies on external AI services. You'll need to obtain API keys for them.

* **Google Gemini API Key:**
    1.  Go to [Google AI Studio](https://aistudio.google.com/) or the Google Cloud Console.
    2.  Create a new API key for the Gemini API.
* **Hugging Face API Key:**
    1.  Go to [Hugging Face](https://huggingface.co/settings/tokens).
    2.  Generate a new "Access Token" (a "read" role is usually sufficient).

**Configure Environment Variables:**

Create a file named `.env` in your **project root directory** (`Career-Counsellor-Chatbot/` where `manage.py` is) and add your obtained API keys:

```dotenv
# .env
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY_HERE"
HUGGINGFACE_API_KEY="YOUR_HUGGING_FACE_API_KEY_HERE"
```

Important: The `.env` file is listed in `.gitignore` and will *not* be committed to version control for security reasons. Your `settings.py` is configured to load these.

### 7. Download AI Models (Crucial Step)

The `career_recommender` component uses a local QA model. The model weight files are often very large and are excluded from the Git repository. You must download them separately.

**Instructions to download the QA model:**

*(You will need to replace this placeholder with specific instructions for your QA model. For example:)*

1.  Navigate to the Hugging Face model page for `your-qa-model-name` (e.g., `https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad`).
2.  Go to the "Files and versions" tab.
3.  Download all necessary files (e.g., `config.json`, `model.safetensors` or `pytorch_model.bin`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`).
4.  Place all these downloaded files into the `career_consulting_project/career/qa_model/` directory. Ensure the directory looks like this:

    ```
    career/qa_model/
    ├── config.json
    ├── model.safetensors # or pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
    # (plus the qa_test_dataset and qa_train_dataset directories if they are part of the model bundle)
    ```

### 8. Prepare Data Files

The project relies on several CSV and JSON datasets. Ensure these are present in their respective locations:

* `career_consulting_project/career/qa_data/coursera_data.csv`
* `career_consulting_project/career/qa_data/job_skills.csv`
* `career_consulting_project/career/skill_data/skills.json`
* `career_consulting_project/career/skill_data/label_role.json`
* `career_consulting_project/career/skill_data/resume_skill_role_analysis.csv` (used for global fallback in chatbot, if applicable)

### 9. Run Database Migrations

Apply the initial database migrations required by Django:

```bash
python manage.py migrate
```

### 10. Start the Development Server

Finally, you can start the Django development server:

```bash
python manage.py runserver
```

The application will now be running. Open your web browser and navigate to `http://127.0.0.1:8000/career/`.

---

## Usage

Once the server is running, you can explore the application's features:

* **Home Page:** Access the main landing page at `http://127.0.0.1:8000/career/`.
  <img width="1018" height="516" alt="image" src="https://github.com/user-attachments/assets/6b39bdf6-cd39-4e6b-9246-2d8ff83e4023" />
* **Resume Upload:** Go to `/career/resume-upload/` to upload your PDF resume for parsing.
  <img width="1018" height="526" alt="image" src="https://github.com/user-attachments/assets/0e84cbc3-11a6-4c12-8df4-253ea9903472" />
* **Analysis Results** View the Resume Parser Analysis results.
  <img width="1004" height="477" alt="image" src="https://github.com/user-attachments/assets/2ea87fc2-b46c-450a-a542-6adadcaad728" />
* **Skill Analysis:** After successfully uploading a resume, proceed to `/career/skill-analysis/` to view the extracted skills, matched career role, and visual analytics.
  <img width="1007" height="520" alt="image" src="https://github.com/user-attachments/assets/dc99bea7-49c6-4a9a-8634-37a3b05512d9" />
* **Technical and Soft Skill Analysis:**
  <img width="996" height="519" alt="image" src="https://github.com/user-attachments/assets/7f7bcfd6-f005-494f-92a7-fe4fcda6cad9" />
* **Job Recommender:** Navigate to `/career/job-recommender/` to get personalized job and course recommendations based on your profile or custom queries.
  <img width="940" height="473" alt="image" src="https://github.com/user-attachments/assets/0189735c-672a-4808-8cd4-26a8943b4cea" />
* **AI Chatbot:** Use the interactive chatbot at `/career/chatbot/` for general career advice or to explore your analysis results interactively.
  <img width="764" height="384" alt="image" src="https://github.com/user-attachments/assets/71203c94-3498-438f-a738-028f5efe545d" />

## 🧪 Testing

This project includes a suite of automated tests to ensure code quality and reliability. Tests are written using the **Pytest** framework.

### Running Tests

1.  Navigate to the project's root directory.
2.  Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the test suite:
    ```bash
    pytest
    ```

### Test Coverage

To generate a report of which code is covered by the tests, run:
```bash
pytest --cov=career
```

## Dataset References
* [Resume Dataset](https://www.kaggle.com/code/warazubairkhan/pdf-resume-text-extraction-analysis-framework)
* [Job Title Description](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)
* [Coursera Course Dataset](https://www.kaggle.com/datasets/siddharthm1698/coursera-course-dataset)









