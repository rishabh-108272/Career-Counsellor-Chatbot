# career/tests.py
import pytest
import pandas as pd
from django.test import Client 
from django.urls import reverse 
# We copy the function here to test it in isolation.
def create_context_entry(row, context_type):
    """
    Creates a text context and metadata dictionary from a pandas Series (row).
    This function is copied from career.career_recommender._load_qa_rag_resources
    for isolated unit testing.
    """
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

# --- Test Case 1: Testing with Job Data ---
def test_create_context_entry_for_job():
    # 1. Arrange: Create a sample dictionary representing a row from a jobs DataFrame
    job_data = {
        'Title': 'Software Engineer',
        'Company': 'Tech Innovations Inc.',
        'Responsibilities': 'Develop and maintain web applications.',
        'Minimum Qualifications': 'B.S. in Computer Science.'
    }
    job_row = pd.Series(job_data) # Convert dict to pandas Series, just like in the real code

    expected_text = "Job Title: Software Engineer. Company: Tech Innovations Inc.. Responsibilities: Develop and maintain web applications.. Minimum Qualifications: B.S. in Computer Science."
    expected_metadata = {"type": "job", "original_data": job_data}

    # 2. Act: Call the function with the sample data
    actual_text, actual_metadata = create_context_entry(job_row, "job")

    # 3. Assert: Check if the outputs are correct
    assert actual_text == expected_text
    assert actual_metadata == expected_metadata

# --- Test Case 2: Testing with Course Data ---
def test_create_context_entry_for_course():
    # 1. Arrange: Create sample course data
    course_data = {
        'course_title': 'Introduction to Python',
        'course_organization': 'Coursera',
        'course_difficulty': 'Beginner'
    }
    course_row = pd.Series(course_data)

    expected_text = "Course Title: Introduction to Python. Organization: Coursera. Difficulty: Beginner"
    expected_metadata = {"type": "course", "original_data": course_data}

    # 2. Act: Call the function
    actual_text, actual_metadata = create_context_entry(course_row, "course")

    # 3. Assert: Check the results
    assert actual_text == expected_text
    assert actual_metadata == expected_metadata
    
    
#---Test Case 3: Testing the Job Recommender View ---
@pytest.mark.django_db
def test_job_recommender_view():
    #Set up test client 
    client=Client()
    url=reverse('career:job_recommender')
    
    #Make a GET request to the url
    response=client.get(url)
    
    #Assert: Check that if the response code is 200 OK
    assert response.status_code == 200
    