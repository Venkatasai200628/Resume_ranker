import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import requests
import json
from flask import Flask, request, render_template, send_file, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# xAI API configuration (replace with your API key)
XAI_API_KEY = "replace your api keys"
XAI_API_URL = "api keys endpoint"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# Preprocess text with SpaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# AI-enhanced keyword extraction using xAI API
def get_ai_keywords(job_description):
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": f"Extract key skills and qualifications from this job description:\n{job_description}\nReturn a list of keywords.",
        "max_tokens": 100
    }
    try:
        response = requests.post(XAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        keywords = result.get("choices", [{}])[0].get("text", "").split(", ")
        return [kw.strip() for kw in keywords if kw.strip()]
    except Exception as e:
        print(f"Error with API: {e}")
        return []

# Scoring algorithm
def score_resumes(resume_texts, job_description):
    # Preprocess job description and resumes
    job_text = preprocess_text(job_description)
    processed_resumes = [preprocess_text(text) for text in resume_texts]
    
    # Get AI-enhanced keywords
    ai_keywords = get_ai_keywords(job_description)
    
    # Combine job description with AI keywords for vectorization
    combined_job_text = job_text + " " + " ".join(ai_keywords)
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    all_texts = [combined_job_text] + processed_resumes
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    
    # Boost scores based on AI keyword matches
    scores = []
    for i, resume in enumerate(resume_texts):
        keyword_matches = sum(1 for kw in ai_keywords if kw.lower() in resume.lower())
        score = similarities[i] * 1 + (keyword_matches / max(len(ai_keywords), 1)) * 0.5
        scores.append(score)
    
    return scores

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resumes' not in request.files or 'job_description' not in request.form:
            return redirect(url_for('index'))  # Silently redirect on error
        
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')
        resume_texts = []
        resume_names = []
        
        for resume in resume_files:
            if resume and resume.filename.endswith('.pdf'):
                filename = secure_filename(resume.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume.save(resume_path)
                text = extract_text_from_pdf(resume_path)
                if text:
                    resume_texts.append(text)
                    resume_names.append(filename)
                os.remove(resume_path)  # Clean up
        
        if not resume_texts:
            return redirect(url_for('index'))  # Silently redirect on error
        
        # Score and rank resumes
        scores = score_resumes(resume_texts, job_description)
        ranked_resumes = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)
        
        # Generate CSV report
        df = pd.DataFrame(ranked_resumes, columns=['Resume', 'Score'])
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.csv')
        df.to_csv(report_path, index=False)
        
        return render_template('results.html', results=ranked_resumes, report_path='report.csv')
    
    return render_template('index.html')

@app.route('/download/<path:filename>')
def download_report(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':

    app.run(debug=True)
