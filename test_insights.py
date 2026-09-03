from src.insights import (
    ResumeInsight,
    ResumeInsights
)


insight = ResumeInsight(
    category="Technical Skills",
    message="Strong SQL and Excel alignment.",
    priority="high"
)


result = ResumeInsights(
    summary="The resume has good analytical foundations.",

    strengths=[
        insight
    ],

    weaknesses=[
        ResumeInsight(
            category="Required Skills",
            message="Database management is not strongly evidenced.",
            priority="high"
        )
    ],

    recommendations=[
        ResumeInsight(
            category="Resume Improvement",
            message="Highlight reporting and analytics experience.",
            priority="medium"
        )
    ],

    overall_assessment=(
        "The candidate has strong analytical fundamentals "
        "but several role-specific gaps remain."
    )
)


print(result)