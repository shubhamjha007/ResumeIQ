import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from src.matcher import SKILL_MATCH_CONFIG, match_skills, normalize_skill, split_compound_skill
from src.semantic_matcher import semantic_match_skills
from src.resume_parser import build_resume_prompt, ground_resume_skills, skill_is_grounded_in_resume
from src.scorer import (calculate_candidate_experience, calculate_education_score,
                        calculate_project_score, convert_duration_to_years,
                        extract_required_years, parse_experience_period)


def resume(**kwargs):
    values = {"skills": [], "experience": [], "education": [], "projects": []}
    values.update(kwargs)
    return SimpleNamespace(**values)


def jd(**kwargs):
    values = {"job_title": "", "required_skills": [], "preferred_skills": [], "responsibilities": [],
              "experience_requirements": [], "education_requirements": []}
    values.update(kwargs)
    return SimpleNamespace(**values)


class MatchingTests(unittest.TestCase):
    def test_exact_case_and_hyphen_matching(self):
        result = match_skills(["Python", "scikit-learn"], ["python", "Scikit Learn"], [])
        self.assertEqual(result["required_score"], 100)
        self.assertEqual(normalize_skill("Machine-Learning"), "machine learning")

    def test_empty_skill_lists(self):
        self.assertEqual(match_skills([], ["Python"], ["SQL"])["required_score"], 0)
        self.assertEqual(match_skills(["Python"], [], [])["required_score"], 0)

    def test_preferred_is_separate(self):
        result = match_skills(["Python"], ["Python", "SQL"], ["Python", "Docker"])
        self.assertEqual(result["required_score"], 50)
        self.assertEqual(len(result["matched_preferred"]), 1)

    def test_safe_aliases_are_normalized_matches(self):
        result = match_skills(
            ["Microsoft Excel", "Statistical Analysis", "RCA"],
            ["Excel", "Basic statistics", "Root cause analysis"], [],
        )
        self.assertEqual(result["required_score"], 100)
        self.assertTrue(all(item["match_type"] == "normalized" for item in result["matched_required"]))

    def test_compounds_are_decomposed_without_broad_generic_components(self):
        compound = "Database management, reporting and analysis"
        result = match_skills(["Data analysis"], [compound], [])
        self.assertEqual(result["required_score"], 0)
        self.assertEqual(result["missing_required"][0]["components"], ["Database management", "reporting"])
        self.assertEqual(split_compound_skill("Root cause analysis"), [])
        self.assertEqual(split_compound_skill("Verbal and written communication"), [])
        self.assertEqual(split_compound_skill("Multi-tasking and prioritization"), [])

    def test_high_confidence_semantic_skill_match(self):
        class FakeModel:
            def encode(self, values):
                vectors = {"Predictive Modeling": [1, 0], "Machine Learning": [1, 0], "Python": [0, 1]}
                return np.array([vectors[value] for value in values])
        with patch("src.semantic_matcher.get_embedding_model", return_value=FakeModel()):
            result = semantic_match_skills(["Machine Learning"], ["Predictive Modeling"], [])
        self.assertEqual(result["required_score"], 100)
        self.assertEqual(result["matched_required"][0]["match_type"], "semantic")

    def test_required_and_preferred_semantic_thresholds_are_separate(self):
        similarity = 0.66
        class FakeModel:
            def encode(self, values):
                vectors = {
                    "Required capability": [1, 0], "Preferred capability": [1, 0],
                    "Related resume skill": [similarity, (1 - similarity ** 2) ** .5],
                }
                return np.array([vectors[value] for value in values])
        with patch("src.semantic_matcher.get_embedding_model", return_value=FakeModel()):
            result = semantic_match_skills(["Related resume skill"], ["Required capability"], ["Preferred capability"])
        self.assertEqual(result["required_score"], 0)
        self.assertEqual(len(result["matched_preferred"]), 1)
        self.assertEqual(result["matching_policy"]["required_semantic_threshold"], SKILL_MATCH_CONFIG["required_semantic_threshold"])
        self.assertEqual(result["matching_policy"]["preferred_semantic_threshold"], SKILL_MATCH_CONFIG["preferred_semantic_threshold"])

    def test_one_supported_component_is_partial_not_a_full_match(self):
        class FakeModel:
            def encode(self, values):
                return np.array([[1, 0] for _ in values])
        with patch("src.semantic_matcher.get_embedding_model", return_value=FakeModel()):
            result = semantic_match_skills(["Data analysis"], ["Database management, reporting and analysis"], [])
        self.assertEqual(len(result["matched_required"]), 0)
        self.assertEqual(len(result["partial_required"]), 1)
        self.assertEqual(result["partial_required"][0]["coverage"], .5)

    def test_all_supported_compound_components_produce_a_compound_match(self):
        class FakeModel:
            def encode(self, values):
                vectors = {
                    "Data analysis and visualization techniques": [.8, .6],
                    "Data analysis": [1, 0],
                    "visualization techniques": [0, 1],
                    "Statistical Analysis": [1, 0],
                    "Data Visualization": [0, 1],
                }
                return np.array([vectors[value] for value in values])
        with patch("src.semantic_matcher.get_embedding_model", return_value=FakeModel()):
            result = semantic_match_skills(
                ["Statistical Analysis", "Data Visualization"],
                ["Data analysis and visualization techniques"], [],
            )
        match = result["matched_required"][0]
        self.assertEqual(match["match_type"], "compound")
        self.assertEqual(match["coverage"], 1.0)
        self.assertEqual([component["status"] for component in match["components"]], ["matched", "matched"])

    def test_low_semantic_similarity_remains_missing(self):
        class FakeModel:
            def encode(self, values):
                vectors = {"Unrelated requirement": [1, 0], "Unrelated resume skill": [0, 1]}
                return np.array([vectors[value] for value in values])
        with patch("src.semantic_matcher.get_embedding_model", return_value=FakeModel()):
            result = semantic_match_skills(["Unrelated resume skill"], ["Unrelated requirement"], [])
        self.assertEqual(result["missing_required"][0]["similarity"], 0.0)
        self.assertNotIn("resume_skill", result["missing_required"][0])

    def test_soft_skill_compound_does_not_emit_irrelevant_component_candidates(self):
        class FakeModel:
            def encode(self, values):
                vectors = {"Verbal and written communication": [1, 0], "Sentence Transformers": [0, 1]}
                return np.array([vectors[value] for value in values])
        with patch("src.semantic_matcher.get_embedding_model", return_value=FakeModel()):
            result = semantic_match_skills(["Sentence Transformers"], ["Verbal and written communication"], [])
        missing = result["missing_required"][0]
        self.assertNotIn("components", missing)
        self.assertNotIn("resume_skill", missing)


class ResumeGroundingTests(unittest.TestCase):
    def test_hallucinated_or_external_skill_is_removed_without_a_blacklist(self):
        resume_text = "Skills: Python, SQL, Excel. Built Power BI dashboards."
        grounded = ground_resume_skills(
            ["Python", "SQL", "Microsoft Excel", "Power BI", "Sentence Transformers"], resume_text)
        self.assertEqual(grounded, ["Python", "SQL", "Microsoft Excel", "Power BI"])
        self.assertFalse(skill_is_grounded_in_resume("Sentence Transformers", resume_text))

    def test_resume_prompt_contains_only_the_supplied_resume_source(self):
        resume_text = "Candidate skill: Python"
        prompt = build_resume_prompt(resume_text)
        self.assertIn(resume_text, prompt)
        self.assertNotIn("Required job skill: Sentence Transformers", prompt)


class ExperienceTests(unittest.TestCase):
    def test_explicit_and_date_durations(self):
        self.assertEqual(convert_duration_to_years("2 years"), 2)
        self.assertEqual(convert_duration_to_years("6 months"), .5)
        self.assertEqual(convert_duration_to_years("1 year 6 months"), 1.5)
        self.assertEqual(convert_duration_to_years("May 2025 - July 2026"), 1.17)
        self.assertEqual(convert_duration_to_years("May 2025 – July 2026"), 1.17)
        self.assertIsNotNone(convert_duration_to_years("May 2025 - Present"))

    def test_requirement_ranges(self):
        self.assertEqual(extract_required_years(["1 to 3 years of experience"]), 1)
        self.assertEqual(extract_required_years(["2-4 years"]), 2)
        self.assertEqual(extract_required_years(["3+ years"]), 3)

    def test_overlapping_and_non_overlapping_dates(self):
        overlap = resume(experience=[SimpleNamespace(duration="Jan 2025 - Dec 2025"), SimpleNamespace(duration="Jun 2025 - May 2026")])
        separate = resume(experience=[SimpleNamespace(duration="Jan 2023 - Jan 2024"), SimpleNamespace(duration="Jan 2025 - Jan 2026")])
        self.assertEqual(calculate_candidate_experience(overlap), 1.33)
        self.assertEqual(calculate_candidate_experience(separate), 2)
        self.assertIsNone(parse_experience_period("not a date"))


class ScoringTests(unittest.TestCase):
    def test_btech_satisfies_generic_bachelor(self):
        candidate = resume(education=[SimpleNamespace(degree="Bachelor of Technology in Electronics and Communication Engineering")])
        self.assertEqual(calculate_education_score(candidate, jd(education_requirements=["Bachelor's degree or equivalent"])), 100)

    def test_field_specific_education_and_empty_education(self):
        candidate = resume(education=[SimpleNamespace(degree="B.Tech Mechanical Engineering")])
        self.assertLess(calculate_education_score(candidate, jd(education_requirements=["Bachelor's degree in Computer Science"])), 100)
        self.assertEqual(calculate_education_score(resume(), jd(education_requirements=["Bachelor's degree"])), 0)

    def test_projects_empty_and_skill_relevance(self):
        target = jd(required_skills=["Python", "Machine Learning"], responsibilities=["Build machine learning models"])
        self.assertEqual(calculate_project_score(resume(), target), 0)
        relevant = resume(projects=[SimpleNamespace(name="Churn Prediction", description="Built a machine learning model in Python", technologies=["Python", "Machine Learning"])])
        unrelated = resume(projects=[SimpleNamespace(name="Shop", description="Built an ecommerce website", technologies=["HTML", "CSS"])])
        class FakeModel:
            def encode(self, values):
                return np.array([[1, 0] if "machine learning" in value.lower() or "python" in value.lower() else [0, 1] for value in values])
        with patch("src.scorer.get_embedding_model", return_value=FakeModel()) as loader:
            relevant_details = calculate_project_score(relevant, target, include_details=True)
            unrelated_details = calculate_project_score(unrelated, target, include_details=True)
        self.assertGreater(relevant_details["score"], unrelated_details["score"])
        self.assertEqual(len(relevant_details["projects"][0]["matches"]), 3)
        self.assertIn("Technologies: Python, Machine Learning", relevant_details["projects"][0]["text"])
        self.assertEqual(loader.call_count, 2)

    def test_malformed_project_is_reported_not_embedded(self):
        details = calculate_project_score(resume(projects=[SimpleNamespace(name="", description="", technologies=[])]),
                                          jd(responsibilities=["Analyze data"]), include_details=True)
        self.assertEqual(details["score"], 0)
        self.assertEqual(len(details["ignored_projects"]), 1)


if __name__ == "__main__":
    unittest.main()
