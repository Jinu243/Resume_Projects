from flask import Blueprint, render_template, request, current_app
from werkzeug.utils import secure_filename
import os
from .utils import allowed_file, extract_resume_text, score_resume, generate_interview_questions

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Check for resume file in the request
        if 'resume' not in request.files:
            return "No resume file provided", 400

        file = request.files['resume']
        job_description = request.form.get('job_description', '')
        
        if file.filename == '':
            return "No selected file", 400

        if file and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            # Save the file securely in the uploads folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract resume text and calculate matching score
            resume_text = extract_resume_text(file_path)
            score, common_keywords = score_resume(resume_text, job_description)
            interview_questions = generate_interview_questions(resume_text)

            result = {
                "score": round(score, 2),
                "common_keywords": common_keywords,
                "interview_questions": interview_questions
            }
    return render_template('index.html', result=result)
