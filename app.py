import base64
import json
import os

import streamlit as st
from google import genai


MODEL_NAME = "gemini-3.6-flash"


def get_api_key():
    """Read the Gemini API key from Streamlit Secrets or an environment variable."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret_key = None

    return secret_key or os.getenv("GEMINI_API_KEY")


def analyze_resume(pdf_bytes, job_description=""):
    """Send the uploaded PDF resume to Gemini and return structured analysis."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to Streamlit Secrets."
        )

    client = genai.Client(api_key=api_key)

    job_context = (
        job_description.strip()
        if job_description.strip()
        else "No job description was provided. Evaluate the resume against general ATS-friendly resume practices."
    )

    prompt = f"""
You are an ATS resume analyst.

Analyze the uploaded resume PDF. Give an ESTIMATED ATS compatibility score, not a claim
that this is the score produced by any real applicant-tracking system.

Use these scoring categories:
- Formatting and ATS readability: 20
- Contact information and basic resume structure: 10
- Skills and keyword coverage: 20
- Experience/project impact and measurable achievements: 20
- Education and relevant qualifications: 10
- Clarity, grammar, consistency, and conciseness: 10
- Overall relevance to the target role: 10

Target job description:
{job_context}

Return ONLY valid JSON with this exact structure:
{{
  "ats_score": 0,
  "score_label": "Strong",
  "summary": "short overall assessment",
  "category_scores": {{
    "formatting": 0,
    "contact_and_structure": 0,
    "skills_and_keywords": 0,
    "experience_and_impact": 0,
    "education": 0,
    "clarity_and_consistency": 0,
    "role_relevance": 0
  }},
  "strengths": ["...", "...", "..."],
  "improvements": [
    {{
      "priority": "High",
      "area": "Keywords",
      "issue": "specific issue found",
      "recommendation": "specific improvement"
    }}
  ],
  "missing_or_weak_keywords": ["...", "..."],
  "ats_risks": ["...", "..."],
  "action_plan": ["step 1", "step 2", "step 3"]
}}

Rules:
- Keep ats_score between 0 and 100.
- Make category scores respect their maximum weights and add up to ats_score.
- Do not invent jobs, degrees, skills, dates, companies, or achievements that are not in the resume.
- If something is missing, say it is missing rather than guessing.
- Give practical, specific recommendations.
- Do not rewrite the entire resume.
"""

    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=[
            {
                "type": "document",
                "data": encoded_pdf,
                "mime_type": "application/pdf",
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    )

    raw_text = (interaction.output_text or "").strip()

    # Be tolerant if the model accidentally wraps JSON in markdown fences.
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "", 1).replace("```", "", 1).strip()

    return json.loads(raw_text)


def score_color(score):
    if score >= 80:
        return "🟢"
    if score >= 60:
        return "🟡"
    return "🔴"


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume ATS Analyzer")
st.write(
    "Upload a PDF resume to get an estimated ATS score, strengths, ATS risks, "
    "and specific improvements using Gemini Flash."
)

st.info(
    "Note: ATS scores vary between real systems. This app provides an AI-based "
    "estimate using a transparent scoring rubric."
)

uploaded_file = st.file_uploader(
    "Upload your resume (PDF only)",
    type=["pdf"],
)

job_description = st.text_area(
    "Optional: paste the job description",
    height=180,
    placeholder="Adding the job description makes keyword and relevance analysis much more useful.",
)

if uploaded_file is not None:
    st.caption(f"Selected: {uploaded_file.name}")

if st.button("🔍 Analyze Resume", type="primary", disabled=uploaded_file is None):
    with st.spinner("Gemini is analyzing your resume..."):
        try:
            result = analyze_resume(uploaded_file.getvalue(), job_description)

            score = int(result.get("ats_score", 0))
            score = max(0, min(100, score))

            st.subheader("ATS Score")
            st.metric("Estimated ATS Compatibility", f"{score}/100")
            st.write(f"{score_color(score)} **{result.get('score_label', 'Assessment')}**")
            st.write(result.get("summary", ""))

            st.subheader("Category Breakdown")
            category_scores = result.get("category_scores", {})
            cols = st.columns(4)

            category_labels = [
                ("Formatting", "formatting"),
                ("Contact & Structure", "contact_and_structure"),
                ("Skills & Keywords", "skills_and_keywords"),
                ("Experience & Impact", "experience_and_impact"),
                ("Education", "education"),
                ("Clarity & Consistency", "clarity_and_consistency"),
                ("Role Relevance", "role_relevance"),
            ]

            for index, (label, key) in enumerate(category_labels):
                with cols[index % 4]:
                    st.metric(label, f"{category_scores.get(key, 0)}")

            left, right = st.columns(2)

            with left:
                st.subheader("✅ Strengths")
                for item in result.get("strengths", []):
                    st.write(f"- {item}")

                st.subheader("🔑 Missing / Weak Keywords")
                keywords = result.get("missing_or_weak_keywords", [])
                if keywords:
                    for item in keywords:
                        st.write(f"- {item}")
                else:
                    st.write("No major keyword gaps were identified.")

            with right:
                st.subheader("⚠️ ATS Risks")
                risks = result.get("ats_risks", [])
                if risks:
                    for item in risks:
                        st.write(f"- {item}")
                else:
                    st.write("No major ATS risks were identified.")

                st.subheader("🛠 Improvement Plan")
                for step in result.get("action_plan", []):
                    st.write(f"- {step}")

            st.subheader("Detailed Improvements")
            improvements = result.get("improvements", [])

            if improvements:
                for item in improvements:
                    priority = item.get("priority", "Medium")
                    area = item.get("area", "General")
                    issue = item.get("issue", "")
                    recommendation = item.get("recommendation", "")

                    with st.expander(f"{priority} priority • {area}"):
                        st.write(f"**Issue:** {issue}")
                        st.write(f"**Recommendation:** {recommendation}")
            else:
                st.write("No detailed improvements were returned.")

        except json.JSONDecodeError:
            st.error(
                "Gemini returned an unexpected response format. Please try again."
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "Check that your Gemini API key is configured and that the uploaded file is a valid PDF."
            )

st.divider()
st.caption("Built with Streamlit + Google Gemini Flash.")
