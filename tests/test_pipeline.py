"""Run with:  python -m pytest -q   (from the project root)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ml_model import compute_match_score, compute_similarity
from pdf_extractor import PDFValidationError, extract_text_from_pdf
from preprocessing import clean_text, preprocess
from recommendations import build_recommendations, build_roadmap
from sample_data import SAMPLE_JOBS, load_sample_resume_bytes
from skill_extractor import extract_jd_skills, extract_skills
from skill_gap import run_pipeline


def test_preprocessing_keeps_tech_tokens_and_removes_stopwords():
    assert "cpp" in clean_text("I know C++ and Node.js").split()
    p = preprocess("The students are building machine learning models")
    assert "the" not in p.filtered and "are" not in p.filtered
    assert p.processed


def test_skill_extraction_boundaries():
    found = extract_skills("Worked with MySQL, JavaScript, Node.js and C++. Able to excel in teams.")
    assert "MySQL" in found and "JavaScript" in found and "C++" in found
    assert "SQL" not in found          # 'mysql' must not trigger 'sql'
    assert "Java" not in found         # 'javascript' must not trigger 'java'
    assert "Excel" not in found        # the verb 'excel' is not the tool


def test_required_vs_preferred_detection():
    jd = extract_jd_skills("Requirements:\n- Python and SQL\n\nPreferred qualifications:\n- Docker\n- Knowledge of AWS is a plus")
    assert jd["Python"]["importance"] == "required"
    assert jd["Docker"]["importance"] == "preferred" and jd["AWS"]["importance"] == "preferred"


def test_cosine_is_higher_for_related_text():
    resume = "Python developer with pandas numpy scikit-learn machine learning projects and SQL databases " * 3
    related = "Looking for python developer with pandas scikit-learn machine learning and SQL " * 3
    unrelated = "Hospital nurse caring for patients administering medicine and supporting doctors " * 3
    refs = list(SAMPLE_JOBS.values())
    assert compute_similarity(resume, related, refs).cosine > compute_similarity(resume, unrelated, refs).cosine


def test_score_formula_and_weight_redistribution():
    full = compute_match_score(0.5, 1.0, 1.0)
    assert full.final == 100.0
    no_pref = compute_match_score(0.25, 0.5, None)       # weights 0.25 / 0.60 re-normalised
    expected = 100 * (0.25 / 0.85 * 0.5 + 0.60 / 0.85 * 0.5)
    assert abs(no_pref.final - round(expected, 1)) < 0.11


def test_pdf_validation_errors():
    for bad in (b"", b"not a pdf", b"%PDF-1.4 garbage"):
        with pytest.raises(PDFValidationError):
            extract_text_from_pdf(bad)


def test_end_to_end_on_sample_data():
    text = extract_text_from_pdf(load_sample_resume_bytes()).text
    res = run_pipeline(text, SAMPLE_JOBS["Machine Learning Intern"])
    assert 0 <= res.score.final <= 100
    assert {"Python", "Pandas", "Scikit-learn"} <= {m["skill"] for m in res.gap.matched}
    assert "TensorFlow" in {m["skill"] for m in res.gap.missing}
    recs = build_recommendations(res.gap)
    assert recs and build_roadmap(recs)
    # a different job must give a different score (no hard-coded results)
    other = run_pipeline(text, SAMPLE_JOBS["DevOps / Cloud Engineer (Entry Level)"])
    assert other.score.final < res.score.final


def test_short_inputs_rejected():
    with pytest.raises(ValueError):
        run_pipeline("too short", SAMPLE_JOBS["Data Analyst"])


# ---------------------------------------------------------------- generic / insufficient job descriptions
from skill_extractor import extract_jd_categories
from skill_gap import (INSUFFICIENT_JD_MESSAGE, InsufficientJobDescriptionError, analyze_skill_gap)

GENERIC_JD = """Software Engineer Trainee
Requirements:
- Knowledge of modern programming languages
- Understanding of database technologies
- Familiarity with web technologies
- Exposure to cloud platforms
- Experience in software development
- Testing and debugging of applications
"""
NO_TECH_JD = ("Office Support Associate. We are looking for a reliable person to join our growing organisation. The role involves attending "
              "daily meetings, coordinating with departments, preparing weekly reports, handling phone calls and emails, managing schedules, "
              "organising documents and supporting the manager. Candidates should be punctual, well organised and friendly.")


def _resume_skills(*names):
    from skill_extractor import SKILL_CATEGORY
    return {n: {"category": SKILL_CATEGORY[n], "count": 1} for n in names}


def test_generic_phrases_become_category_requirements():
    cats = extract_jd_categories(GENERIC_JD, {})
    assert {"Programming languages", "Databases", "Web technologies", "Cloud platforms", "Software development", "Testing practices"} <= set(cats)


def test_named_skills_replace_the_generic_phrase():
    jd = "Requirements:\n- Knowledge of programming languages such as Python\n- Databases such as MySQL\n- Cloud platforms"
    from skill_extractor import extract_jd_skills
    cats = extract_jd_categories(jd, extract_jd_skills(jd))
    assert "Programming languages" not in cats and "Databases" not in cats and "Cloud platforms" in cats


def test_one_skill_does_not_fully_match_a_category():
    cats = extract_jd_categories("Requirements:\n- Knowledge of programming languages", {})
    one = analyze_skill_gap(_resume_skills("Python"), {}, cats)
    assert len(one.partial) == 1 and not one.matched and one.partial[0]["credit"] == 0.5
    two = analyze_skill_gap(_resume_skills("Python", "Java"), {}, cats)
    assert len(two.matched) == 1 and not two.partial
    none = analyze_skill_gap(_resume_skills("HTML"), {}, cats)
    assert len(none.missing) == 1


def test_only_relevant_resume_skills_are_compared():
    cats = extract_jd_categories("Requirements:\n- Cloud platforms", {})
    res = analyze_skill_gap(_resume_skills("Python", "Java", "SQL"), {}, cats)     # no cloud skill present
    assert len(res.missing) == 1 and not res.matched and not res.partial


def test_generic_jd_gives_real_comparison():
    text = extract_text_from_pdf(load_sample_resume_bytes()).text
    res = run_pipeline(text, GENERIC_JD)
    assert res.gap.matched and res.gap.missing                  # not empty, not "no gaps"
    assert "Cloud platforms" in {m["skill"] for m in res.gap.missing}


def test_insufficient_jd_raises_warning_not_a_score():
    text = extract_text_from_pdf(load_sample_resume_bytes()).text
    with pytest.raises(InsufficientJobDescriptionError):
        run_pipeline(text, NO_TECH_JD)
    assert "Insufficient job description information" in INSUFFICIENT_JD_MESSAGE


def test_soft_skills_alone_are_insufficient():
    text = extract_text_from_pdf(load_sample_resume_bytes()).text
    jd = ("Requirements:\n- Excellent communication with clients and colleagues\n- Teamwork and leadership in a growing organisation\n"
          "- Problem solving and time management during busy weeks\n- Attention to detail and adaptability for daily work in our friendly office")
    with pytest.raises(InsufficientJobDescriptionError):
        run_pipeline(text, jd)


def test_no_score_without_requirements_and_no_division_by_zero():
    with pytest.raises(ValueError):
        compute_match_score(0.3, None, None)
    res = analyze_skill_gap({}, {}, {})                          # empty everything must not crash
    assert res.required_cov is None and res.preferred_cov is None
