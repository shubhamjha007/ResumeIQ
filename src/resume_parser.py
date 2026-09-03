from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import os
import re
import logging
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)


# -----------------------------
# Resume data structures
# -----------------------------

class Experience(BaseModel):
    company: str = Field(default="", description="Company or organization name")
    role: str = Field(default="", description="Job title or role")
    duration: str = Field(default="", description="Employment duration if available")
    description: List[str] = Field(default_factory=list,
        description="Important responsibilities or achievements"
    )


class Project(BaseModel):
    name: str = Field(default="", description="Project name")
    description: str = Field(default="", description="Project description")
    technologies: List[str] = Field(default_factory=list,
        description="Technologies, frameworks, or tools used"
    )


class Education(BaseModel):
    degree: str = Field(default="", description="Degree or qualification")
    institution: str = Field(default="", description="College, university, or school")
    duration: str = Field(default="", description="Duration or graduation year if available")


class ResumeData(BaseModel):
    name: str = Field(default="", description="Candidate's full name")
    email: str = Field(default="", description="Candidate email if available")
    phone: str = Field(default="", description="Candidate phone number if available")

    skills: List[str] = Field(default_factory=list,
        description="Technical and professional skills explicitly mentioned"
    )

    education: List[Education] = Field(default_factory=list,
        description="Educational qualifications"
    )

    experience: List[Experience] = Field(default_factory=list,
        description="Work experience and internships"
    )

    projects: List[Project] = Field(default_factory=list,
        description="Projects mentioned in the resume"
    )

    certifications: List[str] = Field(default_factory=list,
        description="Certifications explicitly mentioned"
    )


# -----------------------------
# Gemini client
# -----------------------------

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Add it to .env and restart the app.")
    return genai.Client(api_key=api_key)


# -----------------------------
# Resume analysis function
# -----------------------------

def build_resume_prompt(resume_text: str) -> str:
    """Build the Gemini request from uploaded-PDF text only."""
    return f"""
You are an expert resume information extraction system.

Analyze the resume text below and extract only information
that is explicitly present in the resume.

For every skill, require a literal supporting mention in the resume text.
Do not include model names, libraries, or technologies from your own
knowledge unless the candidate's resume itself names them.

Do not invent:
- skills
- companies
- job titles
- achievements
- certifications
- technologies
- dates

If information is not available, return an empty string
or an empty list as appropriate.

Resume:

{resume_text}
"""


def _source_normalize(value: str) -> str:
    text = str(value or "").casefold()
    # Keep language symbols such as C++/C#, but turn punctuation into word
    # boundaries so a skill at the end of a comma-separated list is retained.
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def skill_is_grounded_in_resume(skill: str, resume_text: str) -> bool:
    """Require a literal skill or canonical alias to appear in PDF text.

    This is source validation, not a technology blacklist. It blocks any
    hallucinated extraction equally while retaining a stated alias such as
    "Microsoft Excel" when the resume says "Excel".
    """
    from src.matcher import normalize_skill
    source = f" {_source_normalize(resume_text)} "
    variants = {_source_normalize(skill), normalize_skill(skill)}
    return any(variant and f" {variant} " in source for variant in variants)


def ground_resume_skills(skills, resume_text: str):
    """Deduplicate and retain only skills evidenced by the uploaded resume."""
    grounded, seen = [], set()
    for skill in skills or []:
        label = str(skill).strip()
        key = _source_normalize(label)
        if key and key not in seen and skill_is_grounded_in_resume(label, resume_text):
            seen.add(key)
            grounded.append(label)
    return grounded


def analyze_resume(resume_text: str) -> ResumeData:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    prompt = build_resume_prompt(resume_text)

    try:
        # Keep a strong reference for the whole request.  Constructing a client
        # inline can let it be finalized/closed during Streamlit reruns.
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeData,
            ),
        )
        parsed = ResumeData.model_validate_json(response.text)
        parsed.skills = ground_resume_skills(parsed.skills, resume_text)
        logger.info("Grounded ResumeData.skills extracted from uploaded PDF: %s", parsed.skills)
        return parsed
    except Exception as exc:
        raise RuntimeError(f"Gemini could not analyze the resume: {exc}") from exc
    finally:
        if "client" in locals():
            client.close()
