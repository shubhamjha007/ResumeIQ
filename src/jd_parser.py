from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv


load_dotenv()


# -----------------------------
# Job Description Data Models
# -----------------------------

class JobDescription(BaseModel):
    job_title: str = Field(default="",
        description="Job title mentioned in the job description"
    )

    required_skills: List[str] = Field(default_factory=list,
        description="Skills explicitly required for the job"
    )

    preferred_skills: List[str] = Field(default_factory=list,
        description="Skills explicitly listed as preferred or nice-to-have"
    )

    responsibilities: List[str] = Field(default_factory=list,
        description="Main responsibilities mentioned in the job description"
    )

    experience_requirements: List[str] = Field(default_factory=list,
        description="Required or preferred experience"
    )

    education_requirements: List[str] = Field(default_factory=list,
        description="Required or preferred educational qualifications"
    )

    certifications: List[str] = Field(default_factory=list,
        description="Certifications explicitly required or preferred"
    )


# -----------------------------
# Gemini Client
# -----------------------------

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Add it to .env and restart the app.")
    return genai.Client(api_key=api_key)


# -----------------------------
# JD Analysis
# -----------------------------

def analyze_job_description(jd_text: str) -> JobDescription:
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description is empty.")

    prompt = f"""
You are an expert job description analysis system.

Analyze the job description below and extract structured
information from it.

IMPORTANT RULES:

1. Extract only information explicitly present.
2. Do not invent skills or requirements.
3. Separate required skills from preferred skills.
4. Keep responsibilities as concise statements.
5. Preserve the meaning of the original job description.
6. If a category is not present, return an empty list.
7. Do not treat a company name as a skill.

Job Description:

{jd_text}
"""

    try:
        # Keep the client alive through parsing the response, then explicitly
        # close it so Streamlit reruns never reuse a stale connection.
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JobDescription,
            ),
        )
        return JobDescription.model_validate_json(response.text)
    except Exception as exc:
        raise RuntimeError(f"Gemini could not analyze the job description: {exc}") from exc
    finally:
        if "client" in locals():
            client.close()
