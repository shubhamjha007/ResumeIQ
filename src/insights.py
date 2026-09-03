import os
from typing import List

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


class ResumeInsight(BaseModel):
    category: str
    message: str
    priority: str = "medium"


class ResumeInsights(BaseModel):
    summary: str

    strengths: List[ResumeInsight] = Field(
        default_factory=list
    )

    weaknesses: List[ResumeInsight] = Field(
        default_factory=list
    )

    recommendations: List[ResumeInsight] = Field(
        default_factory=list
    )

    overall_assessment: str

def generate_insights(
    resume_data,
    jd_data,
    score_data,
    match_result
):
    """
    Generate AI-powered resume insights using Gemini.

    Gemini explains the deterministic evidence produced
    by ResumeIQ. It does not calculate the ResumeIQ score.
    """

    prompt = f"""
You are ResumeIQ, an AI-powered resume analysis assistant.

Your job is to analyze the provided resume evidence against
the provided job description and explain the results.

IMPORTANT RULES:

1. Do NOT invent information.
2. Do NOT claim the candidate has a skill unless the evidence
   supports it.
3. Do NOT change or recalculate the ResumeIQ score.
4. Use the provided scores as authoritative.
5. Distinguish clearly between:
   - strengths
   - weaknesses/gaps
   - recommendations
6. Recommendations should be actionable and specific.
7. Never recommend adding a skill that the candidate does not
   actually have. Instead say that they should add it only if
   they genuinely possess that skill.
8. Keep the analysis professional and concise.
9. Do not introduce qualifications that are not explicitly
   present in the job description, such as "full-time",
   "professional", "senior", "production", etc.

10. When describing an experience gap, distinguish between:
    - total experience
    - relevant role experience
    - required experience range

11. Do not infer that an internship, leadership role, or project
    is equivalent to analyst work unless the provided evidence
    supports that conclusion.

12. When discussing a missing skill, use the exact normalized
    skill/requirement from the matching evidence whenever possible.

13. Do not treat semantic similarity alone as proof that the
    candidate possesses a skill.

14. Recommendations must be conditional when the resume does
    not contain evidence. Use wording such as:
    "If you have this experience, consider highlighting it."

--------------------------------
RESUME DATA
--------------------------------

Name:
{resume_data.name}

Skills:
{resume_data.skills}

Education:
{resume_data.education}

Experience:
{resume_data.experience}

Projects:
{resume_data.projects}

Certifications:
{resume_data.certifications}


--------------------------------
JOB DESCRIPTION
--------------------------------

Job Title:
{jd_data.job_title}

Required Skills:
{jd_data.required_skills}

Preferred Skills:
{jd_data.preferred_skills}

Experience Requirements:
{jd_data.experience_requirements}

Education Requirements:
{jd_data.education_requirements}

Responsibilities:
{jd_data.responsibilities}

Certifications:
{jd_data.certifications}


--------------------------------
RESUMEIQ SCORES
--------------------------------

Overall Score:
{score_data["overall_score"]}

Required Skills:
{score_data["required_skill_score"]}

Preferred Skills:
{score_data["preferred_skill_score"]}

Experience:
{score_data["experience_score"]}

Education:
{score_data["education_score"]}

Projects:
{score_data["project_score"]}


--------------------------------
SKILL MATCHING EVIDENCE
--------------------------------

{match_result}


--------------------------------
OUTPUT REQUIREMENTS
--------------------------------

Return a structured ResumeInsights object.

summary:
Provide a 1-2 sentence summary of the candidate's
overall alignment with this particular job.

strengths:
Identify the strongest evidence in the resume.

weaknesses:
Identify the most important gaps between the resume
and this job description.

recommendations:
Provide practical resume/job-preparation recommendations.

overall_assessment:
Provide a concise final assessment of the candidate's
current fit for this particular role.

Prioritize the most important issues rather than listing
every minor observation.

Each insight must contain:
- category
- message
- priority

Priority must be one of:
"high"
"medium"
"low"
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ResumeInsights,
        },
    )

    return ResumeInsights.model_validate_json(
        response.text
    )