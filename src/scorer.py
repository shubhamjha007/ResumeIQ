"""Deterministic, explainable ResumeIQ scoring."""
import calendar
import re
from datetime import date

from sklearn.metrics.pairwise import cosine_similarity
from src.matcher import normalize_skill
from src.semantic_matcher import get_embedding_model


DEGREE_LEVELS = {"high school": 1, "secondary": 1, "diploma": 2, "associate": 3,
                 "bachelor": 4, "master": 5, "doctorate": 6}


def extract_required_years(requirements):
    """Return the minimum required years (the lower bound for a stated range)."""
    text = " ".join(requirements or []).lower()
    values = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:to|[-–—])\s*(\d+(?:\.\d+)?)\s*years?", text):
        values.append(float(match.group(1)))
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:to|[-–—])\s*\d+(?:\.\d+)?\s*years?", "", text)
    values.extend(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text))
    return max(values, default=0.0)


def _parse_month_year(text):
    match = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{4})", text.strip())
    if not match:
        return None
    try:
        month = list(calendar.month_name).index(match.group(1).title())
    except ValueError:
        try:
            month = list(calendar.month_abbr).index(match.group(1).title()[:3])
        except ValueError:
            return None
    return date(int(match.group(2)), month, 1)


def parse_experience_period(duration, today=None):
    """Parse a date interval as inclusive month indexes, or return ``None``."""
    if not duration:
        return None
    parts = re.split(r"\s*[-–—]\s*", str(duration).strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    start = _parse_month_year(parts[0])
    end_text = parts[1].strip().lower()
    if end_text in {"present", "current", "ongoing", "now"}:
        end = today or date.today()
        end = date(end.year, end.month, 1)
    else:
        end = _parse_month_year(parts[1])
    if not start or not end or end < start:
        return None
    return start.year * 12 + start.month, end.year * 12 + end.month


def convert_duration_to_years(duration):
    """Convert explicit duration or a date range to years; invalid input is safe."""
    if not duration:
        return None
    text = str(duration).lower()
    years = sum(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*years?", text))
    months = sum(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*months?", text))
    if years or months:
        return round(years + months / 12, 2)
    period = parse_experience_period(duration)
    return round((period[1] - period[0]) / 12, 2) if period else None


def _experience_entries(resume_data):
    return getattr(resume_data, "experience", []) or []


def calculate_candidate_experience(resume_data):
    """Sum employment intervals without double-counting overlapping months.

    Undated explicit durations are included after date-based periods because
    their overlap cannot be established; this limitation is surfaced in UI
    explanations rather than guessed away.
    """
    periods, undated_years = [], 0.0
    for entry in _experience_entries(resume_data):
        duration = getattr(entry, "duration", "")
        period = parse_experience_period(duration)
        if period:
            periods.append(period)
        else:
            value = convert_duration_to_years(duration)
            if value is not None:
                undated_years += value
    months, merged = 0, []
    for start, end in sorted(periods):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    for start, end in merged:
        months += end - start
    return round(months / 12 + undated_years, 2)


def calculate_preferred_score(match_result):
    matched = len(match_result.get("matched_preferred", []))
    total = matched + len(match_result.get("missing_preferred", []))
    return round(100 * matched / total, 2) if total else 0.0


def _role_relevance(resume_data, jd_data):
    target = [getattr(jd_data, "job_title", "")] + list(getattr(jd_data, "experience_requirements", []) or [])
    target = [item for item in target if item]
    roles = [" ".join(filter(None, [getattr(e, "role", ""), " ".join(getattr(e, "description", []) or [])])) for e in _experience_entries(resume_data)]
    if not roles or not target:
        return 1.0
    similarities = cosine_similarity(get_embedding_model().encode(roles), get_embedding_model().encode(target))
    # Role title/category is semantic evidence, not a sole gate; avoid claiming relevance from weak similarity.
    return max(0.0, min(1.0, (float(similarities.max()) - 0.25) / 0.45))


def calculate_experience_score(resume_data, jd_data, include_details=False):
    required = extract_required_years(getattr(jd_data, "experience_requirements", []))
    candidate = calculate_candidate_experience(resume_data)
    amount = 100.0 if required == 0 else min(100.0, 100 * candidate / required)
    relevance = _role_relevance(resume_data, jd_data) if required and candidate else (1.0 if required == 0 else 0.0)
    score = round(amount * (0.4 + 0.6 * relevance), 2) if required else 100.0
    details = {"score": score, "candidate_years": candidate, "required_years": required,
               "amount_score": round(amount, 2), "role_relevance": round(relevance * 100, 2)}
    return details if include_details else score


def _degree_level(text):
    value = str(text or "").lower()
    if re.search(r"\b(b\.?tech|b\.?e\.?|b\.?s\.?|bachelor)\b", value): return 4
    if re.search(r"\b(m\.?tech|m\.?e\.?|m\.?s\.?|master)\b", value): return 5
    if re.search(r"\b(ph\.?d|doctorate|doctoral)\b", value): return 6
    for label, level in DEGREE_LEVELS.items():
        if label in value: return level
    return 0


def calculate_education_score(resume_data, jd_data, include_details=False):
    requirements = getattr(jd_data, "education_requirements", []) or []
    education = getattr(resume_data, "education", []) or []
    if not requirements: return {"score": 100.0, "reason": "No education requirement stated."} if include_details else 100.0
    if not education: return {"score": 0.0, "reason": "No education listed."} if include_details else 0.0
    required_level = max((_degree_level(item) for item in requirements), default=0)
    candidate_level = max((_degree_level(getattr(item, "degree", "")) for item in education), default=0)
    if required_level == 0: score, reason = 75.0, "Requirement does not name a degree level."
    elif candidate_level < required_level: score, reason = round(50 * candidate_level / required_level, 2), "Candidate degree level is below the stated requirement."
    else: score, reason = 100.0, "Candidate meets or exceeds the required degree level."
    # Field relevance only applies when a field is explicitly named after an in/with phrase.
    required_text = " ".join(requirements).lower()
    field_match = re.search(r"(?:in|with)\s+([a-z ]{3,40})(?:,|\.|$)", required_text)
    if score and field_match:
        field = field_match.group(1).strip()
        degrees = " ".join(getattr(item, "degree", "") for item in education).lower()
        if field not in degrees and field not in {"a", "any field", "any discipline"}:
            score, reason = round(score * 0.8, 2), reason + " Field differs from the explicitly requested field."
    details = {"score": score, "required_level": required_level, "candidate_level": candidate_level, "reason": reason}
    return details if include_details else score


def _calibrate_project_similarity(similarity):
    """Map MiniLM similarity to relevance evidence without treating it as a percent.

    Values near 0.15 are weak topical overlap; 0.65 is strong evidence for a
    short-project/short-requirement comparison.  The linear middle preserves
    ranking and is intentionally exposed in the result explanation.
    """
    return max(0.0, min(1.0, (float(similarity) - 0.15) / 0.50))


def calculate_project_score(resume_data, jd_data, include_details=False):
    """Score projects from semantic evidence, responsibility coverage and explicit tools.

    Responsibilities are the primary semantic targets. Required skills are
    retained as separate targets in the audit matrix and as explicit
    technology coverage, avoiding a long list of terse skills drowning out a
    project's description.
    """
    raw_projects = getattr(resume_data, "projects", []) or []
    responsibilities = [str(x).strip() for x in (getattr(jd_data, "responsibilities", []) or []) if str(x).strip()]
    required_skills = [str(x).strip() for x in (getattr(jd_data, "required_skills", []) or []) if str(x).strip()]
    targets = ([{"text": item, "type": "responsibility"} for item in responsibilities] +
               [{"text": item, "type": "required_skill"} for item in required_skills])
    ignored, projects = [], []
    for project in raw_projects:
        name, description = str(getattr(project, "name", "") or "").strip(), str(getattr(project, "description", "") or "").strip()
        technologies = [str(t).strip() for t in (getattr(project, "technologies", []) or []) if str(t).strip()]
        text = ". ".join(part for part in (name, description, f"Technologies: {', '.join(technologies)}" if technologies else "") if part)
        if not text:
            ignored.append({"name": name or "Unnamed project", "reason": "Missing name, description, and technologies."})
        else:
            projects.append({"name": name or "Unnamed project", "text": text, "technologies": technologies})
    if not projects:
        details = {"score": 0.0, "reason": "No usable projects listed.", "ignored_projects": ignored,
                   "projects": [], "jd_targets": targets}
        return details if include_details else 0.0
    if not targets:
        details = {"score": 100.0, "reason": "No project-related JD criteria stated.", "ignored_projects": ignored,
                   "projects": projects, "jd_targets": []}
        return details if include_details else 100.0

    # One shared cached MiniLM instance; descriptions and technologies are both embedded.
    model = get_embedding_model()
    matrix = cosine_similarity(model.encode([p["text"] for p in projects]), model.encode([t["text"] for t in targets]))
    project_details = []
    for project, similarities in zip(projects, matrix):
        values = [float(value) for value in similarities]
        top_values = sorted(values, reverse=True)[:min(2, len(values))]
        semantic_evidence = _calibrate_project_similarity(sum(top_values) / len(top_values))
        matches = [{**target, "similarity": round(value, 3)} for target, value in zip(targets, values)]
        project_details.append({**project, "matches": matches, "best_similarity": round(max(values), 3),
                                "average_similarity": round(sum(values) / len(values), 3),
                                "top_two_average": round(sum(top_values) / len(top_values), 3),
                                "semantic_evidence": round(semantic_evidence * 100, 2)})

    best_semantic = max(item["semantic_evidence"] for item in project_details) / 100
    responsibility_indexes = [i for i, target in enumerate(targets) if target["type"] == "responsibility"]
    if responsibility_indexes:
        responsibility_evidence = sum(_calibrate_project_similarity(matrix[:, index].max()) for index in responsibility_indexes) / len(responsibility_indexes)
    else:
        responsibility_evidence = best_semantic
    jd_skills = {normalize_skill(skill) for skill in required_skills if normalize_skill(skill)}
    technologies = {normalize_skill(tech) for project in projects for tech in project["technologies"] if normalize_skill(tech)}
    technology_coverage = len(jd_skills & technologies) / len(jd_skills) if jd_skills else 0.0
    # Semantic relevance remains the largest component; coverage rewards breadth
    # continuously rather than discarding sub-0.45 yet meaningful similarities.
    score = round(100 * (0.60 * best_semantic + 0.20 * responsibility_evidence + 0.20 * technology_coverage), 2)
    details = {"score": score, "formula": "60% best-project semantic evidence + 20% responsibility evidence + 20% exact technology coverage",
               "calibration": "similarity <= 0.15 => 0 evidence; >= 0.65 => 100 evidence; linear between",
               "best_project_semantic_evidence": round(best_semantic * 100, 2),
               "responsibility_evidence": round(responsibility_evidence * 100, 2),
               "technology_coverage": round(technology_coverage * 100, 2), "jd_targets": targets,
               "projects": project_details, "ignored_projects": ignored}
    return details if include_details else score


def calculate_overall_score(required_skill_score, preferred_skill_score, experience_score, education_score, project_score):
    return round(required_skill_score * .50 + preferred_skill_score * .15 + experience_score * .15 + education_score * .10 + project_score * .10, 2)


def generate_resume_score(resume_data, jd_data, match_result):
    required = float(match_result.get("required_score", 0.0))
    preferred = calculate_preferred_score(match_result)
    experience = calculate_experience_score(resume_data, jd_data, True)
    education = calculate_education_score(resume_data, jd_data, True)
    projects = calculate_project_score(resume_data, jd_data, True)
    return {"overall_score": calculate_overall_score(required, preferred, experience["score"], education["score"], projects["score"]),
            "required_skill_score": required, "preferred_skill_score": preferred, "experience_score": experience["score"], "education_score": education["score"], "project_score": projects["score"],
            "explanations": {"experience": experience, "education": education, "projects": projects}}
