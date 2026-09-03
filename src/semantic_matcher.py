"""Explainable semantic fallback for skills not matched deterministically."""
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity

from src.matcher import (SKILL_MATCH_CONFIG, direct_skill_match, match_skills,
                         normalize_skill)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the shared model only when a semantic comparison is needed."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _best_semantic_match(query, resume_skills, resume_embeddings, model, threshold, excluded=()):
    """Return the strongest eligible semantic evidence for one atomic skill."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, resume_embeddings)[0]
    eligible = [index for index in range(len(resume_skills)) if index not in excluded]
    if not eligible:
        return {"similarity": 0.0, "threshold": threshold, "match_type": "semantic"}, None
    best_index = max(eligible, key=lambda index: similarities[index])
    similarity = float(similarities[best_index])
    evidence = {"resume_skill": resume_skills[best_index], "similarity": round(similarity, 3),
                "threshold": threshold, "match_type": "semantic"}
    return evidence, best_index


def _component_evidence(component, resume_skills, resume_embeddings, model, threshold, used_indices):
    """Prefer exact/normalised evidence, otherwise use the category threshold."""
    direct = direct_skill_match(component, resume_skills)
    if direct:
        index = resume_skills.index(direct["resume_skill"])
        if index not in used_indices:
            return {"skill": component, "status": "matched", **direct}, index
    semantic, index = _best_semantic_match(component, resume_skills, resume_embeddings, model,
                                           threshold, used_indices)
    status = "matched" if semantic["similarity"] >= threshold else "missing"
    if status == "missing":
        # A rejected nearest neighbour is not candidate evidence. Showing it
        # made arbitrary implementation terms appear to support soft skills.
        semantic.pop("resume_skill", None)
    return {"skill": component, "status": status, **semantic}, index if status == "matched" else None


def _compound_evidence(item, resume_skills, resume_embeddings, model, threshold, whole_similarity):
    """Assess each component independently and report coverage, never inference alone."""
    used_indices, evidence = set(), []
    for component in item["components"]:
        component_result, resume_index = _component_evidence(
            component, resume_skills, resume_embeddings, model, threshold, used_indices)
        evidence.append(component_result)
        if resume_index is not None:
            used_indices.add(resume_index)
    matched = [component for component in evidence if component["status"] == "matched"]
    coverage = len(matched) / len(evidence) if evidence else 0.0
    status = "matched" if coverage == 1.0 else "partial" if coverage else "missing"
    return {
        "skill": item["skill"], "status": status, "match_type": "compound",
        "similarity": round(whole_similarity, 3), "threshold": threshold,
        "coverage": round(coverage, 3), "components": evidence,
        "resume_skills": [component["resume_skill"] for component in matched],
        "compound_policy": SKILL_MATCH_CONFIG["compound_policy"],
    }


def semantic_match_skills(resume_skills, required_skills, preferred_skills,
                          threshold=None, preferred_threshold=None):
    """Apply direct matching then category-specific semantic/compound evaluation.

    Required and preferred semantic thresholds are intentionally separate.
    Compounds are only fully matched when every meaningful component has
    independent evidence; partial coverage is reported and proportionally
    reflected in the skill score.
    """
    required_threshold = SKILL_MATCH_CONFIG["required_semantic_threshold"] if threshold is None else threshold
    preferred_threshold = (SKILL_MATCH_CONFIG["preferred_semantic_threshold"]
                           if preferred_threshold is None else preferred_threshold)
    resume_skills = [str(skill).strip() for skill in (resume_skills or []) if normalize_skill(skill)]
    exact = match_skills(resume_skills, required_skills or [], preferred_skills or [])
    policy = {**SKILL_MATCH_CONFIG, "required_semantic_threshold": required_threshold,
              "preferred_semantic_threshold": preferred_threshold}
    if not resume_skills:
        return {**exact, "required_score": 0.0, "matching_policy": policy}

    missing = exact["missing_required"] + exact["missing_preferred"]
    if not missing:
        return {**exact, "matching_policy": policy}

    model = get_embedding_model()
    resume_embeddings = model.encode(resume_skills)
    whole_embeddings = model.encode([item["skill"] for item in missing])
    whole_matrix = cosine_similarity(whole_embeddings, resume_embeddings)
    matched_required, missing_required, partial_required = list(exact["matched_required"]), [], []
    matched_preferred, missing_preferred, partial_preferred = list(exact["matched_preferred"]), [], []

    def classify(items, offset, acceptance_threshold, matched, partial, unmatched):
        for index, item in enumerate(items):
            similarities = whole_matrix[offset + index]
            best_index = similarities.argmax()
            whole_similarity = float(similarities[best_index])
            if item.get("components"):
                compound = _compound_evidence(item, resume_skills, resume_embeddings, model,
                                               acceptance_threshold, whole_similarity)
                if compound["status"] == "matched":
                    matched.append({"jd_skill": item["skill"], "resume_skill": ", ".join(compound["resume_skills"]),
                                    "similarity": compound["similarity"], "match_type": "compound",
                                    "coverage": compound["coverage"], "components": compound["components"]})
                elif compound["status"] == "partial":
                    partial.append(compound)
                else:
                    unmatched.append(compound)
                continue

            semantic, _ = _best_semantic_match(item["skill"], resume_skills, resume_embeddings,
                                                model, acceptance_threshold)
            if semantic["similarity"] >= acceptance_threshold:
                matched.append({"jd_skill": item["skill"], **semantic})
            else:
                semantic.pop("resume_skill", None)
                unmatched.append({"skill": item["skill"], **semantic, "status": "missing"})

    classify(exact["missing_required"], 0, required_threshold, matched_required, partial_required, missing_required)
    classify(exact["missing_preferred"], len(exact["missing_required"]), preferred_threshold,
             matched_preferred, partial_preferred, missing_preferred)
    total_required = len(matched_required) + len(partial_required) + len(missing_required)
    required_coverage = len(matched_required) + sum(item["coverage"] for item in partial_required)
    return {"matched_required": matched_required, "partial_required": partial_required,
            "missing_required": missing_required, "matched_preferred": matched_preferred,
            "partial_preferred": partial_preferred, "missing_preferred": missing_preferred,
            "required_score": round(100 * required_coverage / total_required, 2) if total_required else 0.0,
            "matching_policy": policy}
