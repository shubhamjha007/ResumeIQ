"""Deterministic, conservative skill matching primitives.

The alias table is deliberately small: entries are only added when the two
terms are interchangeable for a hiring requirement, not merely related.
"""
import re


SKILL_ALIASES = {
    "py": "python", "js": "javascript", "nodejs": "node js", "node.js": "node js",
    "sklearn": "scikit learn", "scikit learn": "scikit learn", "ms excel": "excel",
    "microsoft excel": "excel", "microsoft office excel": "excel", "rca": "root cause analysis",
    # Statistical analysis demonstrates at least the basic-statistics knowledge
    # commonly requested by entry-level roles.
    "basic statistics": "statistics", "statistics": "statistics",
    "statistical analysis": "statistics", "statistical analyses": "statistics",
}

SKILL_MATCH_CONFIG = {
    "required_semantic_threshold": 0.70,
    "preferred_semantic_threshold": 0.65,
    "compound_policy": "Each meaningful component needs independent evidence. All components matched is a compound match; some matched is a partial match scored by coverage.",
}

# Splitting into one of these creates an unhelpfully broad requirement.
GENERIC_COMPOUND_COMPONENTS = frozenset({"analysis", "management", "skills", "ability"})


def _format_skill(skill: str) -> str:
    text = str(skill or "").casefold().strip()
    text = re.sub(r"[\-_/\.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_skill(skill: str) -> str:
    """Return a conservative canonical form suitable for direct matching."""
    text = _format_skill(skill)
    return SKILL_ALIASES.get(text, text)


def split_compound_skill(skill: str):
    """Return independently meaningful components of a compound requirement.

    Comma/semicolon lists are treated as separate requirements.  A phrase
    joined only by ``and``/``or`` is split only when every segment is itself a
    multi-word competency.  This keeps phrases such as "verbal and written
    communication" and "multi-tasking and prioritization" intact instead of
    comparing isolated modifiers to arbitrary technical skills.
    """
    text = str(skill or "").strip()
    list_separator = re.search(r"[,;]", text)
    conjunction = re.search(r"\s+(?:and|or)\s+", text, flags=re.IGNORECASE)
    if not list_separator and not conjunction:
        return []
    parts = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", text, flags=re.IGNORECASE)
    meaningful = [part.strip() for part in parts if normalize_skill(part) not in GENERIC_COMPOUND_COMPONENTS]
    if not list_separator and any(len(part.split()) < 2 for part in meaningful):
        return []
    return meaningful if len(meaningful) > 1 else []


def _match_type(jd_skill: str, resume_skill: str, normalized: str) -> str:
    """Distinguish literal equality from formatting/alias normalisation."""
    return ("exact" if _format_skill(jd_skill) == _format_skill(resume_skill) == normalized
            else "normalized")


def direct_skill_match(jd_skill, resume_skills):
    """Return direct exact/normalised evidence for one requirement, if present."""
    normalized = normalize_skill(jd_skill)
    for resume_skill in resume_skills or []:
        if normalized and normalized == normalize_skill(resume_skill):
            return {"resume_skill": str(resume_skill).strip(), "similarity": 1.0,
                    "match_type": _match_type(jd_skill, resume_skill, normalized)}
    return None


def match_skills(resume_skills, required_skills, preferred_skills):
    """Match explicit skills by exact text, normalisation, and safe aliases."""
    def classify(skills):
        matched, missing, seen = [], [], set()
        for skill in skills or []:
            normalized = normalize_skill(skill)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            direct = direct_skill_match(skill, resume_skills)
            if direct:
                matched.append({"jd_skill": str(skill).strip(), **direct})
            else:
                missing.append({"skill": str(skill).strip(), "similarity": 0.0,
                                "components": split_compound_skill(skill)})
        return matched, missing

    matched_required, missing_required = classify(required_skills)
    matched_preferred, missing_preferred = classify(preferred_skills)
    total_required = len(matched_required) + len(missing_required)
    return {"matched_required": matched_required, "missing_required": missing_required,
            "matched_preferred": matched_preferred, "missing_preferred": missing_preferred,
            "partial_required": [], "partial_preferred": [],
            "required_score": round(100 * len(matched_required) / total_required, 2) if total_required else 0.0}
