# 📄 ResumeIQ

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)

**AI-powered, job-specific resume analysis. Upload a resume, paste a job description, and get an evidence-based score, ATS check, and actionable feedback.**

ResumeIQ extracts content from PDF resumes and compares it against a target job description. Instead of a vague "good/bad" verdict, it shows *why* a score was given, which keywords are missing, and what to improve, without encouraging fabricated experience.

---

## ✨ Features

- 📑 **Resume parsing:** extracts skills, education, experience, projects, and certifications from PDF resumes
- 🎯 **JD analysis:** identifies required skills, preferred skills, responsibilities, and experience requirements
- 🧠 **Smart skill matching:** exact, normalized, and semantic matching (e.g. related terms count as evidence)
- 📊 **ResumeIQ Score:** a weighted, explainable score for resume-to-JD fit
- 🤖 **ATS analysis:** keyword coverage, resume structure, and contact information checks
- 🔍 **Keyword gap view:** matched, missing, and related keywords at a glance
- 💡 **AI insights:** strengths, weaknesses, and recommendations
- 🧾 **Transparent evidence:** every score is backed by what was found in the resume

---

## 🏗 How It Works

```
PDF Resume ──► PyMuPDF text extraction ─┐
                                        ├─► Gemini (parse into structured data, Pydantic)
Job Description ────────────────────────┘
                                                │
                                                ▼
                        Python scoring engine (deterministic)
              Sentence Transformers (all-MiniLM-L6-v2) + cosine similarity
                                                │
                                                ▼
              ResumeIQ Score  •  ATS Score  •  Keyword Gaps  •  AI Insights
```

**Design choice:** Python computes all numeric scores deterministically. Gemini is used only to parse the resume/JD into structured data and to explain the results, so the same inputs always give the same score.

### ResumeIQ Score

| Component | Weight |
|---|---|
| Required skills | 50% |
| Preferred skills | 15% |
| Experience | 15% |
| Education | 10% |
| Projects | 10% |

Project relevance is judged using semantic evidence, responsibility evidence, and technology coverage.

### ATS Score

| Component | Weight |
|---|---|
| Keyword coverage | 60% |
| Resume structure | 25% |
| Contact information | 15% |

ATS keyword matching is intentionally stricter than skill matching: related semantic matches are not counted as literal keyword coverage, since real ATS systems match keywords.

---

## 🛠 Tech Stack

| Area | Tools |
|---|---|
| Language & UI | Python, Streamlit |
| PDF parsing | PyMuPDF |
| LLM | Google Gemini API |
| Structured output | Pydantic |
| Semantic matching | Sentence Transformers (`all-MiniLM-L6-v2`), cosine similarity |
| Config | python-dotenv |
| Tools | Git, GitHub |

---

## 📂 Project Structure

```
ResumeIQ
│
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py             # PDF text extraction
│   ├── resume_parser.py          # Resume → structured data
│   ├── jd_parser.py              # Job description → structured data
│   ├── prompts.py                # Gemini prompt templates
│   ├── matcher.py                # Exact / normalized skill matching
│   ├── semantic_matcher.py       # Embedding-based semantic matching
│   ├── scorer.py                 # ResumeIQ weighted score
│   ├── ats_analyzer.py           # ATS score (keywords, structure, contact)
│   ├── analyzer.py               # Orchestrates the analysis pipeline
│   └── insights.py               # Strengths, weaknesses, recommendations
├── app.py                        # Streamlit application
├── requirements.txt
├── test_ats_analyzer.py
├── test_insight_generator.py
├── test_insights.py
├── test_keyword_gap_analysis.py
├── test_keyword_gaps.py
├── test_resumeiq.py
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shubhamjha007/ResumeIQ
cd ResumeIQ
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key
```

Get a key from [Google AI Studio](https://aistudio.google.com/).

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 🧪 Running Tests

```bash
python test_ats_analyzer.py
python test_keyword_gap_analysis.py
python test_insight_generator.py
```

---

## 🎯 Future Improvements

- Smarter JD parser that separates technical skills, soft skills, domain knowledge, responsibilities, and experience requirements
- Downloadable optimization report (PDF)
- Support for DOCX resumes
- Side-by-side comparison for multiple job descriptions
- Hosted live demo

---

## 👨‍💻 Author

**Shubham Kumar Jha**

- GitHub: [shubhamjha007](https://github.com/shubhamjha007)
- LinkedIn: [shubhamjha](https://www.linkedin.com/in/shubhamjha07/)
