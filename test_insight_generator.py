"""Manual Gemini insight-generation smoke test; not run during unittest discovery."""
from src.insights import generate_insights


class MockData:
    name = "Test Candidate"
    skills = ["Python", "SQL", "Excel"]
    education = []
    experience = []
    projects = []
    certifications = []


class MockJD:
    job_title = "Data Analyst"
    required_skills = ["SQL", "Excel", "Data Analysis"]
    preferred_skills = ["Python"]
    experience_requirements = ["1-2 years of experience"]
    education_requirements = ["Bachelor's degree"]
    responsibilities = ["Analyze data", "Prepare reports"]
    certifications = []


def run_manual_smoke_test():
    score_data = {"overall_score": 65.0, "required_skill_score": 66.67,
                  "preferred_skill_score": 100.0, "experience_score": 50.0,
                  "education_score": 80.0, "project_score": 40.0}
    match_result = {"matched_required": ["SQL", "Excel"], "missing_required": ["Data Analysis"],
                    "matched_preferred": ["Python"]}
    result = generate_insights(MockData(), MockJD(), score_data, match_result)
    print(result)


if __name__ == "__main__":
    run_manual_smoke_test()
