# 🧠 AI-Powered Resume Ranker

A smart web application that ranks candidate resumes against a job description using NLP, TF-IDF, and AI-driven keyword extraction (powered by the Groq API).

## 🚀 Features
- Upload multiple PDF resumes
- Input a job description
- Automatically extract text and preprocess using SpaCy
- Rank resumes using TF-IDF vectorization and cosine similarity
- Boost relevance scoring with AI-generated keywords
- Downloadable CSV report with ranked candidates
- Beautiful and responsive web UI with TailwindCSS

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **NLP:** SpaCy, Scikit-learn (TF-IDF)
- **AI API:** [Groq API](https://groq.com/)
- **PDF Parsing:** PyPDF2
- **Frontend:** HTML, TailwindCSS
- **Data Handling:** Pandas

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/your-username/ai-resume-ranker.git
cd ai-resume-ranker

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
XAI_API_KEY = "your_groq_api_key"
python resume_ranker.py

├── templates/
│   ├── index.html        # Resume upload form
│   └── results.html      # Ranking results page
├── uploads/              # Temporary file storage
├── resume_ranker.py      # Main Flask application
├── requirements.txt      # Dependencies list
└── README.md             # Project info
