from src.ats_analyzer import (
    ATSIssue,
    ATSAnalysis
)


result = ATSAnalysis(
    ats_score=75.0,
    keyword_coverage=70.0,

    strengths=[
        "Clear resume structure",
        "Good technical keyword coverage"
    ],

    issues=[
        ATSIssue(
            category="Keywords",
            issue="Some job-specific keywords are missing.",
            recommendation=(
                "Add relevant keywords if they accurately "
                "represent your experience."
            ),
            priority="high"
        )
    ],

    recommendations=[
        "Quantify project achievements.",
        "Use relevant job terminology where truthful."
    ],

    summary=(
        "The resume has a generally ATS-friendly structure "
        "but could improve keyword alignment."
    )
)


print(result)

from src.ats_analyzer import (
    detect_resume_sections,
    detect_contact_information,
    calculate_keyword_coverage,
    calculate_structure_score,
    calculate_contact_score,
    calculate_ats_score,
)


class MockJD:

    required_skills = [
        "Python",
        "SQL",
        "Excel"
    ]

    preferred_skills = [
        "Power BI"
    ]


resume_text = """
John Doe

john@example.com
+919876543210
linkedin.com/in/johndoe
github.com/johndoe

SKILLS

Python
SQL
Excel

EDUCATION

Bachelor of Technology

EXPERIENCE

Software Intern

PROJECTS

Customer Churn Prediction

CERTIFICATIONS

Python Certification
"""


print("\nSECTIONS")

print(
    detect_resume_sections(
        resume_text
    )
)


print("\nCONTACT")

print(
    detect_contact_information(
        resume_text
    )
)


print("\nKEYWORD COVERAGE")

print(
    calculate_keyword_coverage(
        resume_text,
        MockJD()
    )
)


print("\nSTRUCTURE SCORE")

print(
    calculate_structure_score(
        resume_text
    )
)


print("\nCONTACT SCORE")

print(
    calculate_contact_score(
        resume_text
    )
)


print("\nATS SCORE")

print(
    calculate_ats_score(
        resume_text,
        MockJD()
    )
)