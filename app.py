import streamlit as st
import logging

from src.pdf_parser import extract_text_from_pdf
from src.resume_parser import analyze_resume
from src.jd_parser import analyze_job_description
from src.matcher import match_skills
from src.semantic_matcher import semantic_match_skills
from src.scorer import generate_resume_score
from src.insights import generate_insights
from src.ats_analyzer import calculate_ats_score


logger = logging.getLogger(__name__)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ResumeIQ",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📄 ResumeIQ")

st.subheader("AI-Powered Resume Analyzer")

st.write(
    "Upload your resume and let ResumeIQ extract "
    "structured information using AI."
)


# --------------------------------------------------
# Resume Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)


# --------------------------------------------------
# Job Description Input
# --------------------------------------------------

st.subheader("Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=300,
    placeholder="Paste the complete job description here..."
)


# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

if uploaded_file is not None:

    st.success("Resume uploaded successfully!")

    if st.button("Analyze Resume"):

        # ------------------------------------------
        # Check Job Description
        # ------------------------------------------

        if not job_description.strip():

            st.warning(
                "Please paste a job description first."
            )

        else:

            # --------------------------------------
            # Extract Resume Text
            # --------------------------------------

            with st.spinner("Extracting resume text..."):

                resume_text = extract_text_from_pdf(
                    uploaded_file
                )

            if not resume_text.strip():

                st.error(
                    "No text could be extracted from this PDF."
                )

            else:

                # ----------------------------------
                # Analyze Resume
                # ----------------------------------

                with st.spinner(
                    "Analyzing resume with Gemini..."
                ):

                    try:
                        resume_data = analyze_resume(resume_text)
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"Resume analysis failed: {exc}")
                        st.stop()

                # ----------------------------------
                # Analyze Job Description
                # ----------------------------------

                with st.spinner(
                    "Analyzing job description..."
                ):

                    try:
                        jd_data = analyze_job_description(job_description)
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"Job-description analysis failed: {exc}")
                        st.stop()

                with st.spinner(
                    "Performing semantic skill matching..."
                ):

                    try:
                        logger.info("ResumeData.skills passed to skill matching: %s", resume_data.skills)
                        match_result = semantic_match_skills(
                            resume_data.skills, jd_data.required_skills, jd_data.preferred_skills
                        )
                    except Exception as exc:
                        st.error(f"Skill matching failed: {exc}")
                        st.stop()

                with st.spinner("Calculating ResumeIQ score..."):

                    try:

                        score_data = generate_resume_score(
                            resume_data,
                            jd_data,
                            match_result
                        )

                        ats_data = calculate_ats_score(
                            resume_text,
                            jd_data,
                            match_result
                        )

                        with st.spinner("Generating AI insights..."):

                            insights = generate_insights(
                                resume_data,
                                jd_data,
                                score_data,
                                match_result
                            )

                    except Exception as exc:

                        st.error(
                            f"Scoring failed: {exc}"
                        )

                        st.stop()

                    from src.scorer import calculate_candidate_experience

                    candidate_years = calculate_candidate_experience(
                        resume_data
                    )

                    st.write(
                        "Calculated candidate experience:",
                        candidate_years,
                        "years"
                    )


                st.success(
                    "Resume and job description "
                    "analyzed successfully!"
                )

                with st.expander("Skill-matching input audit"):
                    st.caption("Exact grounded ResumeData.skills passed to matching")
                    st.json(resume_data.skills)

                # ==================================
                # RESUMEIQ SCORE
                # ==================================

                st.divider()

                st.header("🏆 ResumeIQ Score")

                score = score_data["overall_score"]

                st.metric(
                    "Overall Resume Score",
                    f"{score}/100"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Required Skills",
                        f"{score_data['required_skill_score']}%"
                    )

                with col2:

                    st.metric(
                        "Preferred Skills",
                        f"{score_data['preferred_skill_score']}%"
                    )

                with col3:

                    st.metric(
                        "Experience",
                        f"{score_data['experience_score']}%"
                    )

                with col4:

                    st.metric(
                        "Education",
                        f"{score_data['education_score']}%"
                    )

                st.metric(
                    "Projects",
                    f"{score_data['project_score']}%"
                )

                st.divider()

                st.header("🤖 AI Insights")

                st.subheader("📋 Summary")

                st.write(
                    insights.summary
                )


                st.subheader("🔥 Strengths")

                if insights.strengths:

                    for insight in insights.strengths:

                        st.write(
                            f"**{insight.category}** — "
                            f"{insight.message}"
                        )

                else:

                    st.write("No major strengths identified.")


                st.subheader("⚠️ Weaknesses")

                if insights.weaknesses:

                    for insight in insights.weaknesses:

                        st.write(
                            f"**{insight.category}** — "
                            f"{insight.message}"
                        )

                else:

                    st.write("No major weaknesses identified.")


                st.subheader("💡 Recommendations")

                if insights.recommendations:

                    for insight in insights.recommendations:

                        st.write(
                            f"**{insight.category}** — "
                            f"{insight.message}"
                        )

                else:

                    st.write("No recommendations generated.")


                st.subheader("🎯 Overall Assessment")

                st.write(
                    insights.overall_assessment
                )

                # ==================================
                # ATS ANALYSIS
                # ==================================

                st.divider()

                st.header("📋 ATS Analysis")

                st.metric(
                    "ATS Score",
                    f"{ats_data['ats_score']}/100"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Keyword Coverage",
                        f"{ats_data['keyword_coverage']}%"
                    )

                with col2:

                    st.metric(
                        "Resume Structure",
                        f"{ats_data['structure_score']}%"
                    )

                with col3:

                    st.metric(
                        "Contact Information",
                        f"{ats_data['contact_score']}%"
                    )

                st.subheader("🔑 ATS Keyword Analysis")

                keyword_gaps = ats_data["keyword_gaps"]


                st.write("### Required Keywords")


                if keyword_gaps.matched_required:

                    st.write("#### ✅ Matched")

                    for item in keyword_gaps.matched_required:

                        st.write(
                            f"• {item.requirement}"
                        )

                else:

                    st.write(
                        "No required keywords matched literally."
                    )


                if keyword_gaps.missing_required:

                    st.write("#### ❌ Missing")

                    for item in keyword_gaps.missing_required:

                        st.write(
                            f"• {item.requirement}"
                        )

                else:

                    st.write(
                        "No required keywords are missing."
                    )


                if keyword_gaps.related_required:

                    st.write("#### 🟡 Related but not exact")

                    for item in keyword_gaps.related_required:

                        st.write(
                            f"• **{item.requirement}**"
                        )

                        if item.resume_evidence:

                            st.caption(
                                f"Resume evidence: "
                                f"{item.resume_evidence}"
                            )


                st.write("### Preferred Keywords")


                if keyword_gaps.matched_preferred:

                    st.write("#### ✅ Matched")

                    for item in keyword_gaps.matched_preferred:

                        st.write(
                            f"• {item.requirement}"
                        )

                else:

                    st.write(
                        "No preferred keywords matched literally."
                    )


                if keyword_gaps.missing_preferred:

                    st.write("#### ❌ Missing")

                    for item in keyword_gaps.missing_preferred:

                        st.write(
                            f"• {item.requirement}"
                        )

                else:

                    st.write(
                        "No preferred keywords are missing."
                    )


                if keyword_gaps.related_preferred:

                    st.write("#### 🟡 Related but not exact")

                    for item in keyword_gaps.related_preferred:

                        st.write(
                            f"• **{item.requirement}**"
                        )

                        if item.resume_evidence:

                            st.caption(
                                f"Resume evidence: "
                                f"{item.resume_evidence}"
                            )

                st.subheader("📑 Resume Sections")

                for section, detected in ats_data["sections"].items():

                    if detected:

                        st.write(
                            f"✅ {section.title()}"
                        )

                    else:

                        st.write(
                            f"❌ {section.title()}"
                        )

                st.subheader("📞 Contact Information")

                for item, detected in (
                    ats_data["contact_information"].items()
                ):

                    if detected:

                        st.write(
                            f"✅ {item.title()}"
                        )

                    else:

                        st.write(
                            f"❌ {item.title()}"
                        )

                # ==================================
                # SKILL MATCHING
                # ==================================

                st.divider()

                st.header("🎯 Skill Matching")


                # ----------------------------------
                # Overall Required Skill Score
                # ----------------------------------

                st.subheader("Required Skill Match")

                score = match_result["required_score"]

                policy = match_result.get("matching_policy")
                if policy:
                    st.caption(
                        "Matching policy — exact/normalised aliases: 1.0; "
                        f"required semantic: ≥ {policy['required_semantic_threshold']:.2f}; "
                        f"preferred semantic: ≥ {policy['preferred_semantic_threshold']:.2f}."
                    )

                st.metric(
                    "Required Skill Score",
                    f"{score}%"
                )


                # ----------------------------------
                # Matched Required Skills
                # ----------------------------------

                st.subheader("✅ Matched Required Skills")

                if match_result["matched_required"]:

                    for match in match_result["matched_required"]:

                        st.write(
                            f"✓ **{match['jd_skill']}**"
                        )

                        st.caption(
                            f"Resume skill: {match['resume_skill']} "
                            f"| Similarity: {match['similarity']} | Method: {match['match_type']}"
                        )

                else:

                    st.write(
                        "No required skills matched."
                    )


                # ----------------------------------
                # Partially Matched Required Skills
                # ----------------------------------

                if match_result.get("partial_required"):
                    st.subheader("🟡 Partially Matched Required Skills")
                    for match in match_result["partial_required"]:
                        st.write(f"≈ **{match['skill']}** — {match['coverage']:.0%} component coverage")
                        st.caption(
                            f"Whole-skill similarity: {match['similarity']} | "
                            f"Threshold: {match['threshold']} | Method: compound"
                        )
                        for component in match["components"]:
                            evidence = component.get("resume_skill", "No supporting resume skill")
                            st.caption(
                                f"{component['status'].title()}: {component['skill']} → {evidence} "
                                f"({component['similarity']})"
                            )

                # ----------------------------------
                # Missing Required Skills
                # ----------------------------------

                st.subheader("❌ Missing Required Skills")

                if match_result["missing_required"]:

                    for match in match_result["missing_required"]:

                        st.write(
                            f"✗ **{match['skill']}**"
                        )

                        st.caption(
                            f"Status: missing | Method: {match.get('match_type', 'semantic')} | "
                            f"Best semantic similarity: {match['similarity']} | "
                            f"Threshold: {match.get('threshold', policy['required_semantic_threshold'] if policy else 'n/a')}"
                        )
                        if match.get("components"):
                            st.caption(f"Compound coverage: {match.get('coverage', 0):.0%}")
                            for component in match["components"]:
                                evidence = component.get("resume_skill", "No supporting resume skill")
                                st.caption(
                                    f"{component['status'].title()}: {component['skill']} → {evidence} "
                                    f"({component['similarity']})"
                                )

                else:

                    st.write(
                        "No required skills are missing!"
                    )


                # ----------------------------------
                # Matched Preferred Skills
                # ----------------------------------

                st.subheader("⭐ Matched Preferred Skills")

                if match_result["matched_preferred"]:

                    for match in match_result["matched_preferred"]:

                        st.write(
                            f"✓ **{match['jd_skill']}**"
                        )

                        st.caption(
                            f"Resume skill: {match['resume_skill']} "
                            f"| Similarity: {match['similarity']} | Method: {match['match_type']}"
                        )

                else:

                    st.write(
                        "No preferred skills matched."
                    )


                # ----------------------------------
                # Partially Matched Preferred Skills
                # ----------------------------------

                if match_result.get("partial_preferred"):
                    st.subheader("🟡 Partially Matched Preferred Skills")
                    for match in match_result["partial_preferred"]:
                        st.write(f"≈ **{match['skill']}** — {match['coverage']:.0%} component coverage")
                        for component in match["components"]:
                            evidence = component.get("resume_skill", "No supporting resume skill")
                            st.caption(
                                f"{component['status'].title()}: {component['skill']} → {evidence} "
                                f"({component['similarity']})"
                            )

                # ----------------------------------
                # Missing Preferred Skills
                # ----------------------------------

                st.subheader("📌 Missing Preferred Skills")

                if match_result["missing_preferred"]:

                    for match in match_result["missing_preferred"]:

                        st.write(
                            f"• **{match['skill']}**"
                        )

                        st.caption(
                            f"Status: missing | Method: {match.get('match_type', 'semantic')} | "
                            f"Best semantic similarity: {match['similarity']} | "
                            f"Threshold: {match.get('threshold', policy['preferred_semantic_threshold'] if policy else 'n/a')}"
                        )
                        if match.get("components"):
                            st.caption(f"Compound coverage: {match.get('coverage', 0):.0%}")

                else:

                    st.write(
                        "No preferred skills are missing!"
                    )
                # ==================================
                # RESUME ANALYSIS
                # ==================================

                st.divider()

                st.header("📄 Resume Analysis")

                # ----------------------------------
                # Candidate Information
                # ----------------------------------

                st.subheader("Candidate Information")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write("**Name**")
                    st.write(resume_data.name)

                with col2:

                    st.write("**Email**")
                    st.write(resume_data.email)

                with col3:

                    st.write("**Phone**")
                    st.write(resume_data.phone)

                # ----------------------------------
                # Skills
                # ----------------------------------

                st.divider()

                st.subheader("Skills")

                if resume_data.skills:

                    st.write(
                        ", ".join(resume_data.skills)
                    )

                else:

                    st.write("No skills found.")

                # ----------------------------------
                # Education
                # ----------------------------------

                st.divider()

                st.subheader("Education")

                if resume_data.education:

                    for education in resume_data.education:

                        st.write(
                            f"**{education.degree}** — "
                            f"{education.institution}"
                        )

                        if education.duration:

                            st.caption(
                                education.duration
                            )

                else:

                    st.write("No education information found.")

                # ----------------------------------
                # Experience
                # ----------------------------------

                st.divider()

                st.subheader("Experience")

                if resume_data.experience:

                    for experience in resume_data.experience:

                        st.write(
                            f"### {experience.role} — "
                            f"{experience.company}"
                        )

                        if experience.duration:

                            st.caption(
                                experience.duration
                            )

                        for point in experience.description:

                            st.write(
                                f"- {point}"
                            )

                else:

                    st.write("No experience found.")

                # ----------------------------------
                # Projects
                # ----------------------------------

                st.divider()

                st.subheader("Projects")

                if resume_data.projects:

                    for project in resume_data.projects:

                        st.write(
                            f"### {project.name}"
                        )

                        st.write(
                            project.description
                        )

                        if project.technologies:

                            st.write(
                                "**Technologies:** "
                                + ", ".join(
                                    project.technologies
                                )
                            )

                else:

                    st.write("No projects found.")

                # ----------------------------------
                # Certifications
                # ----------------------------------

                st.divider()

                st.subheader("Certifications")

                if resume_data.certifications:

                    for certification in (
                        resume_data.certifications
                    ):

                        st.write(
                            f"- {certification}"
                        )

                else:

                    st.write(
                        "No certifications found."
                    )

                # ==================================
                # JOB DESCRIPTION ANALYSIS
                # ==================================

                st.divider()

                st.header("💼 Job Description Analysis")

                # ----------------------------------
                # Job Title
                # ----------------------------------

                st.write(
                    f"**Job Title:** "
                    f"{jd_data.job_title}"
                )

                # ----------------------------------
                # Required Skills
                # ----------------------------------

                st.subheader("Required Skills")

                if jd_data.required_skills:

                    for skill in jd_data.required_skills:

                        st.write(
                            f"✓ {skill}"
                        )

                else:

                    st.write(
                        "No required skills identified."
                    )

                # ----------------------------------
                # Preferred Skills
                # ----------------------------------

                st.subheader("Preferred Skills")

                if jd_data.preferred_skills:

                    for skill in jd_data.preferred_skills:

                        st.write(
                            f"• {skill}"
                        )

                else:

                    st.write(
                        "No preferred skills identified."
                    )

                # ----------------------------------
                # Responsibilities
                # ----------------------------------

                st.subheader("Responsibilities")

                if jd_data.responsibilities:

                    for responsibility in (
                        jd_data.responsibilities
                    ):

                        st.write(
                            f"- {responsibility}"
                        )

                else:

                    st.write(
                        "No responsibilities identified."
                    )

                # ----------------------------------
                # Experience Requirements
                # ----------------------------------

                st.subheader(
                    "Experience Requirements"
                )

                if jd_data.experience_requirements:

                    for requirement in (
                        jd_data.experience_requirements
                    ):

                        st.write(
                            f"- {requirement}"
                        )

                else:

                    st.write(
                        "No specific experience "
                        "requirements identified."
                    )

                # ----------------------------------
                # Education Requirements
                # ----------------------------------

                st.subheader(
                    "Education Requirements"
                )

                if jd_data.education_requirements:

                    for requirement in (
                        jd_data.education_requirements
                    ):

                        st.write(
                            f"- {requirement}"
                        )

                else:

                    st.write(
                        "No specific education "
                        "requirements identified."
                    )

                # ----------------------------------
                # Certifications
                # ----------------------------------

                st.subheader("Certifications")

                if jd_data.certifications:

                    for certification in (
                        jd_data.certifications
                    ):

                        st.write(
                            f"- {certification}"
                        )

                else:

                    st.write(
                        "No certifications identified."
                    )
