import os
import spacy
from pdfminer.high_level import extract_text as pdf_extract_text
import openai

# Load spaCy model (make sure you've run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def allowed_file(filename, allowed_extensions):
    """
    Check if the file extension is allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_resume_text(file_path):
    """
    Extract text from a resume file. Supports PDF and TXT.
    """
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        try:
            text = pdf_extract_text(file_path)
        except Exception:
            text = ""
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = ""
    return text

def score_resume(resume_text, job_description):
    """
    Use spaCy to process the resume and job description,
    then compute a matching score based on common keywords.
    """
    doc_resume = nlp(resume_text.lower())
    doc_job = nlp(job_description.lower())

    # Get sets of lemma tokens (excluding stop words and non-alpha tokens)
    resume_tokens = {token.lemma_ for token in doc_resume if not token.is_stop and token.is_alpha}
    job_tokens = {token.lemma_ for token in doc_job if not token.is_stop and token.is_alpha}

    common_tokens = resume_tokens.intersection(job_tokens)
    score = len(common_tokens) / len(job_tokens) if job_tokens else 0
    return score, list(common_tokens)

def generate_interview_questions(resume_text):
    """
    Generate interview questions based on the resume text using the newer ChatCompletion API.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "OpenAI API key not set."

    openai.api_key = openai_api_key

    # We'll provide a system message that sets the context, and a user message with the resume excerpt.
    prompt = (
        f"Generate 5 interview questions for a candidate based on the following resume excerpt:\n\n"
        f"{resume_text[:500]}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another ChatGPT model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates interview questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150,
            temperature=0.7,
        )
        # The chat response is now in response.choices[0].message["content"]
        questions = response.choices[0].message["content"].strip()
    except Exception as e:
        questions = f"Error generating questions: {str(e)}"
    return questions
