import unittest

from src.ats_analyzer import analyze_keyword_gaps


class MockJD:
    def __init__(self, required=None, preferred=None):
        self.required_skills = required or []
        self.preferred_skills = preferred or []


class KeywordGapAnalysisTests(unittest.TestCase):
    def test_exact_keyword_is_matched(self):
        result = analyze_keyword_gaps("Skills: SQL", MockJD(required=["SQL"]))
        self.assertEqual([item.requirement for item in result.matched_required], ["SQL"])
        self.assertFalse(result.missing_required)

    def test_case_and_punctuation_differences_are_matched(self):
        result = analyze_keyword_gaps(
            "PYTHON; power bi; CI/CD",
            MockJD(required=["Python", "Power-BI", "CI CD"]),
        )
        self.assertEqual(len(result.matched_required), 3)

    def test_multi_word_keyword_is_matched(self):
        result = analyze_keyword_gaps(
            "Performed root-cause analysis for production incidents.",
            MockJD(required=["Root Cause Analysis"]),
        )
        self.assertEqual(result.matched_required[0].requirement, "Root Cause Analysis")

    def test_substring_is_not_a_match(self):
        result = analyze_keyword_gaps("Built a NoSQLite migration utility.", MockJD(required=["SQL"]))
        self.assertFalse(result.matched_required)
        self.assertEqual(result.missing_required[0].requirement, "SQL")

    def test_missing_keyword_is_reported(self):
        result = analyze_keyword_gaps("Python", MockJD(required=["Excel"]))
        self.assertEqual(result.missing_required[0].requirement, "Excel")

    def test_required_and_preferred_are_kept_separate(self):
        result = analyze_keyword_gaps("Python", MockJD(required=["Python"], preferred=["Docker"]))
        self.assertEqual([item.requirement for item in result.matched_required], ["Python"])
        self.assertEqual([item.requirement for item in result.missing_preferred], ["Docker"])

    def test_compound_requirement_retains_original_and_reports_components(self):
        requirement = "Database management, reporting and analysis"
        result = analyze_keyword_gaps("Database-management", MockJD(required=[requirement]))
        item = result.missing_required[0]
        self.assertEqual(item.requirement, requirement)
        self.assertEqual(item.components, ["Database management", "reporting and analysis"])
        self.assertEqual(item.matched_components, ["Database management"])
        self.assertEqual(item.missing_components, ["reporting and analysis"])

    def test_semantic_evidence_is_related_not_literal(self):
        semantic_result = {
            "matched_required": [{
                "jd_skill": "Basic statistics",
                "resume_skill": "Statistical Analysis",
                "match_type": "semantic",
                "similarity": 0.82,
            }],
            "matched_preferred": [],
        }
        result = analyze_keyword_gaps(
            "Skills: Statistical Analysis",
            MockJD(required=["Basic statistics"]),
            semantic_result,
        )
        self.assertEqual(result.missing_required[0].requirement, "Basic statistics")
        self.assertEqual(result.related_required[0].resume_evidence, "Statistical Analysis")
        self.assertEqual(result.related_required[0].match_type, "semantic")


if __name__ == "__main__":
    unittest.main()
