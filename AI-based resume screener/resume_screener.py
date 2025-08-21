# AI-Based Resume Screener
# Complete Python Application for VS Code

import os
import json
import re
import streamlit as st
from pathlib import Path
import PyPDF2
import docx
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIError, APITimeoutError

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ResumeScreener:
    def __init__(self):
        self.api_status = "unknown"  # Track API status: 'active', 'no_quota', 'error'
        self.setup_openai()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def setup_openai(self):
        """
        🔧 CONFIGURATION POINT 1: OpenAI API Setup
        """
        # Initialize the OpenAI client with your API key
        # REPLACE THIS WITH YOUR VALID API KEY OR SET AS st.secrets["OPENAI_API_KEY"]
        api_key = "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"  # Replace this
        
        if api_key and not api_key.startswith("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"):
            try:
                self.client = OpenAI(api_key=api_key)
                # Make a simple, cheap test call to check the API key and quota
                test_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say 'Hello'"}],
                    max_tokens=5,
                )
                self.api_status = "active"
                st.sidebar.success("✅ OpenAI API is connected and active.")
            except (RateLimitError, APIError) as e:
                if "quota" in str(e).lower() or "insufficient_quota" in str(e).lower():
                    self.api_status = "no_quota"
                    st.sidebar.warning("⚠️ OpenAI API key has no quota. Switching to Basic Analysis.")
                else:
                    self.api_status = "error"
                    st.sidebar.warning(f"⚠️ OpenAI API error: {e}. Switching to Basic Analysis.")
            except (APIConnectionError, APITimeoutError, Exception) as e:
                self.api_status = "error"
                st.sidebar.warning(f"⚠️ Could not connect to OpenAI: {e}. Switching to Basic Analysis.")
        else:
            self.api_status = "no_key"
            st.sidebar.info("ℹ️ No valid API key found. Using Basic Analysis mode.")

    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded resume files"""
        try:
            if uploaded_file.type == "application/pdf":
                return self.extract_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self.extract_from_docx(uploaded_file)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
                return ""
        except Exception as e:
            st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
            return ""

    def extract_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF files"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def extract_from_docx(self, docx_file) -> str:
        """Extract text from DOCX files"""
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def preprocess_text(self, text: str) -> List[str]:
        """Text Preprocessing"""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        return processed_tokens

    def extract_skills_and_experience(self, resume_text: str) -> Dict[str, Any]:
        """Skills and Experience Extraction"""
        skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'trello']
        }
        
        found_skills = {}
        resume_lower = resume_text.lower()
        
        for category, skills in skill_categories.items():
            found_skills[category] = [skill for skill in skills if skill in resume_lower]
        
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:in|with)',
            r'experience[:\s]*(\d+)\+?\s*years?'
        ]
        
        years_experience = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_lower)
            if matches:
                years_experience = max(years_experience, int(matches[0]))
        
        return {
            'skills': found_skills,
            'years_experience': years_experience,
            'total_skills_found': sum(len(skills) for skills in found_skills.values())
        }

    def analyze_with_openai(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """OpenAI Analysis with robust error handling"""
        # Check API status before attempting call
        if self.api_status != "active":
            st.warning("OpenAI API not available. Using enhanced fallback analysis.")
            return self.enhanced_fallback_analysis(resume_text, job_description)
            
        prompt = f"""
        Analyze this resume against the job description and provide a detailed assessment.
        
        Job Description:
        {job_description}
        
        Resume:
        {resume_text[:3000]}
        
        Please provide a JSON response with:
        1. overall_score: numerical score from 0-100
        2. skills_match: list of matched skills from the resume
        3. experience_level: assessment of experience level
        4. strengths: list of candidate's strengths
        5. weaknesses: list of areas for improvement
        6. recommendation: hire/interview/reject recommendation
        7. summary: brief 2-sentence summary
        
        Respond only in valid JSON format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError:
            st.warning("OpenAI response was not valid JSON. Using enhanced analysis.")
            return self.enhanced_fallback_analysis(resume_text, job_description)
        except (RateLimitError, APIError) as e:
            if "quota" in str(e).lower():
                self.api_status = "no_quota"
                st.warning("OpenAI quota exceeded. Using enhanced analysis.")
            return self.enhanced_fallback_analysis(resume_text, job_description)
        except Exception as e:
            st.warning(f"Analysis issue: {e}. Using enhanced analysis.")
            return self.enhanced_fallback_analysis(resume_text, job_description)

    def enhanced_fallback_analysis(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """A much better fallback analysis that uses the job description"""
        resume_skills_data = self.extract_skills_and_experience(resume_text)
        job_skills_data = self.extract_skills_and_experience(job_description)
        
        # Calculate match by comparing skills found in resume vs job description
        job_skills_flat = [skill for skills in job_skills_data['skills'].values() for skill in skills]
        resume_skills_flat = [skill for skills in resume_skills_data['skills'].values() for skill in skills]
        
        matched_skills = [skill for skill in job_skills_flat if skill in resume_skills_flat]
        missing_skills = [skill for skill in job_skills_flat if skill not in resume_skills_flat]
        
        # Calculate score based on skill match and experience
        skill_match_ratio = len(matched_skills) / len(job_skills_flat) if job_skills_flat else 0
        base_score = int(skill_match_ratio * 70)  # 70% weight on skills
        
        # Experience bonus (up to 30%)
        experience_bonus = min(resume_skills_data['years_experience'] * 3, 30)
        total_score = min(base_score + experience_bonus, 100)
        
        # Generate recommendations based on score
        if total_score >= 75:
            recommendation = "interview"
            summary = f"Strong candidate with {len(matched_skills)}/{len(job_skills_flat)} required skills and {resume_skills_data['years_experience']} years of experience."
        elif total_score >= 50:
            recommendation = "review"
            summary = f"Potential candidate with {len(matched_skills)}/{len(job_skills_flat)} required skills. Needs further review."
        else:
            recommendation = "reject"
            summary = f"Weak candidate match. Only {len(matched_skills)}/{len(job_skills_flat)} required skills found."
        
        return {
            'overall_score': total_score,
            'skills_match': matched_skills,
            'experience_level': f"{resume_skills_data['years_experience']} years",
            'strengths': matched_skills[:3] if matched_skills else ["Quick learner", "Adaptable"],
            'weaknesses': missing_skills[:3] if missing_skills else ["Limited relevant experience"],
            'recommendation': recommendation,
            'summary': summary
        }

    def generate_report(self, results: List[Dict]) -> pd.DataFrame:
        """Report Generation"""
        report_data = []
        for result in results:
            report_data.append({
                'Candidate': result['name'],
                'Score': result['analysis']['overall_score'],
                'Experience': result['analysis']['experience_level'],
                'Skills Count': len(result['analysis']['skills_match']),
                'Recommendation': result['analysis']['recommendation'].upper(),
                'Top Skills': ', '.join(result['analysis']['skills_match'][:5])
            })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('Score', ascending=False)

def main():
    """Main Streamlit Application"""
    st.set_page_config(
        page_title="AI Resume Screener",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI-Based Resume Screener</h1>
        <p>Automate your hiring process with intelligent resume analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    screener = ResumeScreener()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show analysis mode based on API status
        if screener.api_status == "active":
            analysis_mode = st.selectbox(
                "Analysis Mode",
                ["Full AI Analysis", "Basic Analysis"]
            )
        else:
            analysis_mode = "Basic Analysis"
            st.info("ℹ️ Using Basic Analysis (OpenAI not available)")
        
        min_score_threshold = st.slider("Minimum Score Threshold", 0, 100, 60)
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Enter job description
        2. Upload resume files
        3. Click 'Analyze Resumes'
        4. Review ranked results
        5. Export reports
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Job Description")
        job_description = st.text_area(
            "Enter the job requirements, skills, and qualifications:",
            height=200,
            value="""We are seeking a Senior Data Engineer to design and build scalable data pipelines and analytics platforms.

Responsibilities:
- Architect, build, and optimize data pipelines and ETL/ELT workflows
- Collaborate with analytics and data science teams
- Ensure data quality, reliability, and governance

Required Skills:
- Strong proficiency in Python and SQL
- Experience with cloud platforms (AWS, Azure, or GCP)
- Familiarity with orchestration and transformation tools (Airflow, dbt)
- Working knowledge of data warehouses (Snowflake, BigQuery, or Redshift)
- Experience with Spark or distributed computing frameworks
- CI/CD and version control (GitHub/GitLab), containerization (Docker)"""
        )
    
    with col2:
        st.header("📂 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Select resume files:",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
    
    if st.button("🔍 Analyze Resumes", type="primary", use_container_width=True):
        if not job_description.strip():
            st.error("Please enter a job description")
            return
            
        if not uploaded_files:
            st.error("Please upload at least one resume")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {uploaded_file.name}...")
            
            resume_text = screener.extract_text_from_file(uploaded_file)
            
            if resume_text:
                if analysis_mode == "Full AI Analysis" and screener.api_status == "active":
                    analysis = screener.analyze_with_openai(resume_text, job_description)
                else:
                    analysis = screener.enhanced_fallback_analysis(resume_text, job_description)
                
                results.append({
                    'name': uploaded_file.name,
                    'text': resume_text,
                    'analysis': analysis
                })
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            results.sort(key=lambda x: x['analysis']['overall_score'], reverse=True)
            
            st.header("📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Candidates", len(results))
            with col2:
                avg_score = sum(r['analysis']['overall_score'] for r in results) / len(results)
                st.metric("Average Score", f"{avg_score:.1f}")
            with col3:
                qualified = len([r for r in results if r['analysis']['overall_score'] >= min_score_threshold])
                st.metric("Qualified Candidates", qualified)
            with col4:
                top_score = max(r['analysis']['overall_score'] for r in results)
                st.metric("Top Score", f"{top_score}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                scores = [r['analysis']['overall_score'] for r in results]
                names = [r['name'] for r in results]
                
                fig = px.bar(
                    x=names, y=scores,
                    title="Candidate Scores",
                    labels={'x': 'Candidate', 'y': 'Score'},
                    color=scores,
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                recommendations = [r['analysis']['recommendation'] for r in results]
                rec_counts = pd.Series(recommendations).value_counts()
                
                fig = px.pie(
                    values=rec_counts.values,
                    names=rec_counts.index,
                    title="Recommendation Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.header("🎯 Detailed Candidate Analysis")
            
            for i, result in enumerate(results):
                with st.expander(f"#{i+1} {result['name']} - Score: {result['analysis']['overall_score']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Summary")
                        st.write(result['analysis']['summary'])
                        
                        st.subheader("💪 Strengths")
                        for strength in result['analysis']['strengths']:
                            st.write(f"• {strength}")
                        
                        st.subheader("⚠️ Areas for Improvement")
                        for weakness in result['analysis']['weaknesses']:
                            st.write(f"• {weakness}")
                    
                    with col2:
                        st.subheader("🛠️ Matched Skills")
                        skills_text = ", ".join(result['analysis']['skills_match'])
                        st.write(skills_text if skills_text else "No specific skills extracted")
                        
                        st.subheader("📈 Experience Level")
                        st.write(result['analysis']['experience_level'])
                        
                        st.subheader("✅ Recommendation")
                        recommendation = result['analysis']['recommendation'].upper()
                        color = "green" if recommendation == "INTERVIEW" else "orange" if recommendation == "REVIEW" else "red"
                        st.markdown(f"<span style='color: {color}; font-weight: bold;'>{recommendation}</span>", 
                                  unsafe_allow_html=True)
            
            st.header("📤 Export Results")
            
            report_df = screener.generate_report(results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_report = {
                    'analysis_date': datetime.now().isoformat(),
                    'job_description': job_description,
                    'total_candidates': len(results),
                    'candidates': results
                }
                
                st.download_button(
                    label="📋 Download JSON Report",
                    data=json.dumps(json_report, indent=2),
                    file_name=f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    st.markdown("---")

if __name__ == "__main__":
    main()