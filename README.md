# AI-Based Resume Screener

An interactive Streamlit app that analyzes resumes against a job description using classic NLP and an optional LLM-assisted analysis. Produces ranked candidates, detailed per-candidate insights, and exportable CSV/JSON reports.

## Features
- Resume ingestion from PDF, DOCX, and TXT
- Text extraction and preprocessing (tokenization, stopword removal, stemming)
- Skill/experience extraction and a transparent rule-based scoring fallback
- Optional OpenAI analysis with robust fallback when API is unavailable
- Visual dashboards (bar chart of scores, recommendation pie chart)
- Detailed per-candidate strengths, weaknesses, matched skills, and recommendation
- One-click export to CSV and JSON

## Project Structure
- `resume_screener.py`: Streamlit app and core logic
- `requirements.txt`: Python dependencies
- `docs/Project_Report.md`: Detailed report (methodology, implementation, results)
- `slides/Presentation.md`: Slide outline summarizing findings

## Prerequisites
- Python 3.9–3.12
- Windows, macOS, or Linux

## Setup
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# PowerShell (Windows)
.\\.venv\\Scripts\\Activate.ps1
# bash/zsh (macOS/Linux)
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

## OpenAI Configuration (Optional)
The app can use an LLM for enhanced analysis when an API key is available. Otherwise it automatically falls back to a transparent, rule-based method.

Recommended: set an environment variable before launching the app and update the code to read it (or use Streamlit secrets):
```powershell
# PowerShell
setx OPENAI_API_KEY "<hemant's-api-key>"
```
In Streamlit, you can also store secrets in `.streamlit/secrets.toml`:
```toml
[general]
OPENAI_API_KEY = "<your-key>"
```
Note: The provided script includes a placeholder for the API key. Replace it with a secure method (env var or secrets) before sharing.

## Run the App
```bash
streamlit run resume_screener.py
```
Then open the URL printed in the terminal (typically `http://localhost:8501`).

## How to Use
1. Paste your job description in the left panel.
2. Upload one or more resumes (PDF, DOCX, TXT).
3. Click "Analyze Resumes".
4. Review ranked candidates, inspect details, and export CSV/JSON reports.

## Methodology (Brief)
- Extract text via `PyPDF2`/`python-docx` or raw TXT
- Preprocess with NLTK: lowercase, tokenize, remove stopwords, stem
- Extract skills with curated category lists and detect years of experience via regex
- Compute a score: 70% skill match + up to 30% experience bonus
- If OpenAI is available, request a structured JSON assessment with strengths, weaknesses, and recommendation

For a deep dive, see `docs/Project_Report.md`.

## Limitations and Ethics
- Keyword-based matching can miss synonyms and nuanced expertise
- LLM outputs may vary; validate important decisions and watch for bias
- Do not use personally sensitive data beyond what is necessary; comply with local laws

## Submission Checklist
- Source code and documentation (this repository + README)
- Detailed report (`docs/Project_Report.md`)
- Presentation slides (`slides/Presentation.md`)

## License
MIT (adjust as needed)
