import re
import unicodedata
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.matcher import (
    SKILL_ALIASES,
    normalize_skill,
)

class ATSIssue(BaseModel):
    category: str
    issue: str
    recommendation: str
    priority: str = "medium"


class ATSAnalysis(BaseModel):
    ats_score: float = Field(
        ge=0,
        le=100
    )

    keyword_coverage: float = Field(
        ge=0,
        le=100
    )

    strengths: List[str] = Field(
        default_factory=list
    )

    issues: List[ATSIssue] = Field(
        default_factory=list
    )

    recommendations: List[str] = Field(
        default_factory=list
    )

    summary: str


class KeywordGapItem(BaseModel):
    """One JD requirement, retaining its original wording for the UI."""

    requirement: str
    components: List[str] = Field(default_factory=list)
    matched_components: List[str] = Field(default_factory=list)
    missing_components: List[str] = Field(default_factory=list)
    resume_evidence: Optional[str] = None
    match_type: Optional[str] = None


class KeywordGapAnalysis(BaseModel):
    """Literal ATS keyword evidence, with optional non-scoring context."""

    matched_required: List[KeywordGapItem] = Field(default_factory=list)
    missing_required: List[KeywordGapItem] = Field(default_factory=list)
    matched_preferred: List[KeywordGapItem] = Field(default_factory=list)
    missing_preferred: List[KeywordGapItem] = Field(default_factory=list)
    related_required: List[KeywordGapItem] = Field(default_factory=list)
    related_preferred: List[KeywordGapItem] = Field(default_factory=list)


# These entries in the existing matcher intentionally represent related
# competencies.  They are useful semantic context, but are not literal ATS
# evidence (e.g. "Statistical Analysis" is not the phrase "Basic statistics").
_CONCEPTUAL_ALIAS_CANONICALS = frozenset({"statistics"})


def _normalize_literal_text(value: str) -> str:
    """Normalise harmless presentation differences without using substrings."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("++", " plus plus ").replace("#", " sharp ")
    text = re.sub(r"[\-_/.]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _literal_variants(requirement: str) -> List[str]:
    """Return formatting/safe-alias variants accepted as literal ATS evidence."""
    canonical = normalize_skill(requirement)
    variants = {_normalize_literal_text(requirement)}
    if canonical not in _CONCEPTUAL_ALIAS_CANONICALS:
        for alias, alias_canonical in SKILL_ALIASES.items():
            if alias_canonical == canonical:
                variants.add(_normalize_literal_text(alias))
    return [variant for variant in variants if variant]


def _literal_present(normalized_resume: str, requirement: str) -> Optional[str]:
    """Find a whole-token literal/formatting match; never a raw substring."""
    for variant in _literal_variants(requirement):
        # Normalisation makes phrases space-separated, so boundaries prevent
        # SQL matching "nosqlite" and preserve multi-word phrase order.
        pattern = rf"(?<!\w){re.escape(variant)}(?!\w)"
        if re.search(pattern, normalized_resume):
            return variant
    return None


def _requirement_components(requirement: str) -> List[str]:
    """Split list-like compound requirements while retaining meaningful phrases.

    A comma/semicolon separates list items but does not break a phrase such as
    "reporting and analysis" into the overly broad word "analysis".
    """
    text = str(requirement or "").strip()
    if not re.search(r"[,;]", text):
        return [text] if text else []
    return [part.strip(" \t,;") for part in re.split(r"\s*[,;]\s*", text)
            if part.strip(" \t,;")]


def _semantic_evidence(requirement: str, semantic_match_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Use existing matcher output only as labelled, non-literal context."""
    if not semantic_match_result:
        return None
    wanted = normalize_skill(requirement)
    for bucket in ("matched_required", "matched_preferred"):
        for item in semantic_match_result.get(bucket, []):
            jd_skill = item.get("jd_skill", "")
            if normalize_skill(jd_skill) != wanted:
                continue
            evidence = item.get("resume_skill")
            if evidence and item.get("match_type") in {"semantic", "normalized"}:
                return {"resume_evidence": str(evidence), "match_type": item.get("match_type")}
    return None


def analyze_keyword_gaps(
    resume_text: str,
    jd_data: Any,
    semantic_match_result: Optional[Dict[str, Any]] = None,
) -> KeywordGapAnalysis:
    """Classify required/preferred JD skills by literal ATS evidence.

    This deliberately does not feed into ``calculate_keyword_coverage`` or
    any ResumeIQ scoring.  Semantic matcher output, if supplied, is surfaced
    as "related but not exact" only for requirements that remain literal gaps.
    """
    normalized_resume = _normalize_literal_text(resume_text)

    def classify(requirements: List[str]):
        matched, missing, related, seen = [], [], [], set()
        for raw_requirement in requirements or []:
            requirement = str(raw_requirement or "").strip()
            normalized_requirement = normalize_skill(requirement)
            if not requirement or normalized_requirement in seen:
                continue
            seen.add(normalized_requirement)
            components = _requirement_components(requirement)
            matched_components = [component for component in components
                                  if _literal_present(normalized_resume, component)]
            missing_components = [component for component in components
                                  if component not in matched_components]
            item = KeywordGapItem(
                requirement=requirement,
                components=components,
                matched_components=matched_components,
                missing_components=missing_components,
                match_type="literal" if not missing_components else None,
            )
            if not missing_components:
                matched.append(item)
                continue

            # Semantic/normalised matches are shown as related context,
            # not as missing keywords. This prevents the same requirement
            # from appearing in both Missing and Related.
            semantic = _semantic_evidence(requirement, semantic_match_result)
            if semantic:
                related.append(item.model_copy(update=semantic))
            else:
                missing.append(item)
        return matched, missing, related

    required = classify(getattr(jd_data, "required_skills", []))
    preferred = classify(getattr(jd_data, "preferred_skills", []))
    return KeywordGapAnalysis(
        matched_required=required[0], missing_required=required[1], related_required=required[2],
        matched_preferred=preferred[0], missing_preferred=preferred[1], related_preferred=preferred[2],
    )

def detect_resume_sections(resume_text: str) -> Dict[str, bool]:
    """
    Detect common resume sections.
    """

    text = resume_text.lower()

    sections = {
        "summary": False,
        "skills": False,
        "education": False,
        "experience": False,
        "projects": False,
        "certifications": False,
    }

    section_aliases = {
        "summary": [
            "summary",
            "professional summary",
            "profile"
        ],
        "skills": [
            "skills",
            "technical skills",
            "key skills"
        ],
        "education": [
            "education",
            "academic background"
        ],
        "experience": [
            "experience",
            "work experience",
            "professional experience"
        ],
        "projects": [
            "projects",
            "academic projects",
            "personal projects"
        ],
        "certifications": [
            "certifications",
            "certificates"
        ],
    }

    for section, aliases in section_aliases.items():

        for alias in aliases:

            pattern = rf"\b{re.escape(alias)}\b"

            if re.search(pattern, text):

                sections[section] = True
                break

    return sections

def detect_contact_information(
    resume_text: str
) -> Dict[str, bool]:
    """
    Detect common contact information.
    """

    email_found = bool(
        re.search(
            r"\b[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            resume_text
        )
    )

    phone_found = bool(
        re.search(
            r"(?:\+91[\s-]?)?"
            r"\b[6-9]\d{9}\b",
            resume_text
        )
    )

    linkedin_found = (
        "linkedin.com" in resume_text.lower()
    )

    github_found = (
        "github.com" in resume_text.lower()
    )

    return {
        "email": email_found,
        "phone": phone_found,
        "linkedin": linkedin_found,
        "github": github_found,
    }

def calculate_keyword_coverage(
    resume_text: str,
    jd_data
) -> float:
    """
    Calculate literal ATS keyword coverage.

    Only exact/normalized literal evidence counts toward
    ATS keyword coverage. Semantic similarity does not.
    """

    if not resume_text:
        return 0.0

    keywords = (
        getattr(jd_data, "required_skills", [])
        + getattr(jd_data, "preferred_skills", [])
    )

    keywords = [
        str(keyword).strip()
        for keyword in keywords
        if str(keyword).strip()
    ]

    if not keywords:
        return 100.0

    normalized_resume = _normalize_literal_text(
        resume_text
    )

    matched = 0
    seen = set()

    for keyword in keywords:

        canonical = normalize_skill(keyword)

        if canonical in seen:
            continue

        seen.add(canonical)

        if _literal_present(
            normalized_resume,
            keyword
        ):
            matched += 1

    return round(
        matched / len(seen) * 100,
        2
    )

def calculate_structure_score(
    resume_text: str
) -> float:
    """
    Calculate an ATS-friendly structure score.
    """

    sections = detect_resume_sections(
        resume_text
    )

    important_sections = [
        "skills",
        "education",
        "experience",
        "projects"
    ]

    present = sum(
        sections[section]
        for section in important_sections
    )

    return round(
        present
        / len(important_sections)
        * 100,
        2
    )

def calculate_contact_score(
    resume_text: str
) -> float:
    """
    Calculate contact information completeness.
    """

    contact = detect_contact_information(
        resume_text
    )

    important = [
        "email",
        "phone"
    ]

    optional = [
        "linkedin",
        "github"
    ]

    score = 0

    for item in important:

        if contact[item]:

            score += 25

    for item in optional:

        if contact[item]:

            score += 12.5

    return round(
        min(score, 100),
        2
    )

def calculate_ats_score(
    resume_text: str,
    jd_data,
    semantic_match_result=None
) -> dict:
    """
    Calculate the deterministic ATS score.
    """

    keyword_score = calculate_keyword_coverage(
        resume_text,
        jd_data
    )

    structure_score = calculate_structure_score(
        resume_text
    )

    contact_score = calculate_contact_score(
        resume_text
    )

    keyword_gaps = analyze_keyword_gaps(
    resume_text,
    jd_data,
    semantic_match_result
    )

    ats_score = (
        keyword_score * 0.60
        + structure_score * 0.25
        + contact_score * 0.15
    )

    return {
        "ats_score": round(
            ats_score,
            2
        ),

        "keyword_coverage": keyword_score,

        "structure_score": structure_score,

        "contact_score": contact_score,

        "sections": detect_resume_sections(
            resume_text
        ),

        "contact_information":
            detect_contact_information(
                resume_text
            ),

        "keyword_gaps": keyword_gaps,
    }
