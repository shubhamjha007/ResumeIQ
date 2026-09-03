import unittest
from types import SimpleNamespace

from src.ats_analyzer import analyze_keyword_gaps, calculate_ats_score


def jd(required=(), preferred=()):
    return SimpleNamespace(required_skills=list(required), preferred_skills=list(preferred))


class KeywordGapTests(unittest.TestCase):
    def test_exact_case_punctuation_and_multiword_keywords(self):
        data = jd(required=["SQL", "Power-BI", "Data Analysis"])
        gaps = analyze_keyword_gaps("Skills: sql, POWER BI; data   analysis.", data)
        self.assertEqual([item["keyword"] for item in gaps["required"]["matched"]],
                         ["SQL", "Power-BI", "Data Analysis"])
        self.assertEqual(gaps["required"]["missing"], [])

    def test_substrings_do_not_count_as_ats_keyword_matches(self):
        gaps = analyze_keyword_gaps("Built a NoSQL database service.", jd(required=["SQL"]))
        self.assertEqual(gaps["required"]["matched"], [])
        self.assertEqual(gaps["required"]["missing"][0]["keyword"], "SQL")

    def test_required_and_preferred_are_kept_separate(self):
        gaps = analyze_keyword_gaps("Python", jd(required=["Python", "SQL"], preferred=["Docker"]))
        self.assertEqual([item["keyword"] for item in gaps["required"]["matched"]], ["Python"])
        self.assertEqual([item["keyword"] for item in gaps["required"]["missing"]], ["SQL"])
        self.assertEqual([item["keyword"] for item in gaps["preferred"]["missing"]], ["Docker"])

    def test_compound_requirement_preserves_original_and_reports_partial_coverage(self):
        keyword = "Database management, reporting and analysis"
        gaps = analyze_keyword_gaps("Created reporting and analysis dashboards.", jd(required=[keyword]))
        partial = gaps["required"]["partial"][0]
        self.assertEqual(partial["keyword"], keyword)
        self.assertEqual(partial["coverage"], 0.5)
        self.assertEqual(partial["components"][1]["keyword"], "reporting and analysis")

    def test_related_evidence_is_not_literal_coverage(self):
        semantic_result = {
            "matched_required": [], "partial_required": [],
            "matched_preferred": [{"jd_skill": "Basic statistics", "resume_skill": "Statistical Analysis",
                                    "match_type": "normalized", "similarity": 1.0}],
            "partial_preferred": [],
        }
        gaps = analyze_keyword_gaps("Statistical Analysis", jd(preferred=["Basic statistics"]), semantic_result)
        self.assertEqual(gaps["preferred"]["matched"], [])
        self.assertEqual(gaps["preferred"]["missing"][0]["keyword"], "Basic statistics")
        self.assertEqual(gaps["preferred"]["related"][0]["resume_evidence"], "Statistical Analysis")

    def test_existing_ats_score_calculation_is_unchanged(self):
        data = jd(required=["Python", "SQL"], preferred=[])
        score = calculate_ats_score("Skills: Python", data)
        self.assertEqual(score["keyword_coverage"], 50.0)
        self.assertEqual(score["ats_score"], 36.25)


if __name__ == "__main__":
    unittest.main()
