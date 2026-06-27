#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranker
====================================
Multi-signal scoring system for the Senior AI Engineer JD.

Architecture:
  1. JD Understanding  — parse the role into explicit criteria dimensions
  2. Profile Scoring   — score each candidate on 6 dimensions
  3. Behavioral Multiplier — apply Redrob platform signals
  4. Honeypot Filter   — detect and penalize impossible profiles
  5. Final Ranking     — sort, normalize, resolve ties

Design philosophy:
  - No LLM API calls during ranking (offline, CPU-only)
  - Transparent, inspectable scores per dimension
  - Reasoning generated from actual scoring evidence
  - Runs <5 min for 100K candidates on 16 GB CPU machine

Author: Hackathon submission
"""

import json
import csv
import re
import math
import time
import gzip
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ─── JD CRITERIA ─────────────────────────────────────────────────────────────

# Skills that are must-have vs nice-to-have vs red-flag for this role
MUST_HAVE_SKILLS = {
    # embeddings & retrieval
    "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5",
    "faiss", "vector search", "vector database", "vector db", "semantic search",
    "dense retrieval", "hybrid retrieval", "hybrid search",
    # vector databases
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    # ranking & IR
    "information retrieval", "learning to rank", "ranking", "reranking", "re-ranking",
    "ndcg", "mrr", "bm25", "retrieval", "recommendation system", "recommender",
    # core ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "transformers", "bert", "fine-tuning", "fine tuning",
    # python
    "python",
    # evaluation
    "a/b testing", "ab testing", "offline evaluation", "online evaluation",
    # LLM/AI
    "llm", "large language model", "rag", "retrieval augmented generation",
    "langchain", "llama", "gpt", "openai"
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "xgboost", "lightgbm", "pytorch", "tensorflow",
    "kubernetes", "docker", "spark", "kafka", "distributed systems",
    "open source", "github", "huggingface", "mlflow", "wandb", "weights & biases",
    "airflow", "dbt", "data engineering"
}

# Skills that are off-target for this specific role
OFF_TARGET_SKILLS = {
    "computer vision", "image classification", "object detection", "image segmentation",
    "speech recognition", "tts", "text to speech", "asr", "robotics", "unity",
    "photoshop", "figma", "ui/ux", "frontend", "react", "angular", "vue",
    "tailwind", "css", "html", "wordpress", "seo", "digital marketing",
    "salesforce", "erp", "sap", "tableau", "power bi"
}

# Consulting/pure-services companies (JD explicitly disqualifies pure consulting background)
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "lt infotech", "niit", "mastech", "syntel", "zensar", "persistent systems"
}

# Honeypot detection patterns
EXPERT_SKILLS_0_MONTHS_THRESHOLD = 5   # If ≥ N expert skills all have 0 duration → honeypot
MIN_COMPANY_AGE_SIGNALS = {
    # If a company was founded in year X, experience there before X is impossible
    # We detect this via duration vs career dates consistency
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "delhi ncr", "ncr", "hyderabad", "mumbai",
    "bangalore", "bengaluru", "gurugram", "gurgaon", "india"
}

REFERENCE_DATE = date(2025, 6, 1)  # approximate evaluation date


@dataclass
class CandidateScore:
    candidate_id: str
    # Component scores (0-1)
    skill_score: float = 0.0
    career_score: float = 0.0
    experience_score: float = 0.0
    location_score: float = 0.0
    education_score: float = 0.0
    behavioral_multiplier: float = 1.0
    honeypot_penalty: float = 1.0
    # Diagnostic
    raw_score: float = 0.0
    final_score: float = 0.0
    evidence: dict = field(default_factory=dict)

    def compute_final(self):
        # Weighted combination of components
        base = (
            0.35 * self.skill_score +
            0.35 * self.career_score +
            0.15 * self.experience_score +
            0.10 * self.location_score +
            0.05 * self.education_score
        )
        self.raw_score = base
        self.final_score = base * self.behavioral_multiplier * self.honeypot_penalty
        return self.final_score


# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def normalize(val: float, lo: float, hi: float) -> float:
    """Clamp-normalize val to [0, 1]."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def text_lower(s: Optional[str]) -> str:
    return (s or "").lower()


def months_since(date_str: Optional[str]) -> float:
    """Return months between date_str and REFERENCE_DATE. Negative means future."""
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        delta = REFERENCE_DATE - d
        return delta.days / 30.44
    except Exception:
        return 999


def parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None


# ─── SCORING COMPONENTS ──────────────────────────────────────────────────────

def score_skills(candidate: dict) -> tuple[float, dict]:
    """
    Score based on skills list with proficiency × endorsement × duration weighting.
    Bonus for assessment scores. Penalty for off-target skill concentration.
    """
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    must_have_hits = []
    nice_hits = []
    off_target_hits = []
    total_skill_weight = 0.0
    must_have_weight = 0.0
    nice_weight = 0.0

    PROFICIENCY_MULT = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

    for sk in skills:
        name_raw = sk.get("name", "")
        name = name_raw.lower()
        prof = PROFICIENCY_MULT.get(sk.get("proficiency", "beginner"), 0.3)
        endorsements = min(sk.get("endorsements", 0), 100) / 100.0
        duration = min(sk.get("duration_months", 0), 60) / 60.0
        assessment = assessment_scores.get(name_raw, assessment_scores.get(name, -1))
        assess_bonus = max(0.0, assessment / 100.0) * 0.2 if assessment >= 0 else 0.0

        # Skill weight = proficiency + endorsements (trust signal) + duration (depth) + assessment
        weight = prof * 0.5 + endorsements * 0.25 + duration * 0.25 + assess_bonus

        # Check category
        in_must = any(mh in name or name in mh for mh in MUST_HAVE_SKILLS)
        in_nice = any(nh in name or name in nh for nh in NICE_TO_HAVE_SKILLS) and not in_must
        in_off = any(ot in name for ot in OFF_TARGET_SKILLS) and not in_must

        if in_must:
            must_have_weight += weight
            must_have_hits.append(name_raw)
        elif in_nice:
            nice_weight += weight * 0.4
            nice_hits.append(name_raw)
        elif in_off:
            off_target_hits.append(name_raw)

    # Normalize: ideal is ~10 strong must-have skills (weight ~7) + some nice-to-haves
    must_norm = min(1.0, must_have_weight / 6.0)
    nice_norm = min(1.0, nice_weight / 2.0) * 0.3

    # Penalty if off-target skills dominate (> 40% of all skills)
    off_ratio = len(off_target_hits) / max(1, len(skills))
    off_penalty = max(0.0, 1.0 - off_ratio * 1.5) if off_ratio > 0.4 else 1.0

    skill_score = (must_norm * 0.8 + nice_norm * 0.2) * off_penalty

    evidence = {
        "must_have_hits": must_have_hits[:8],
        "nice_hits": nice_hits[:4],
        "off_target_hits": off_target_hits[:4],
        "must_have_weight": round(must_have_weight, 2),
    }
    return skill_score, evidence


def score_career(candidate: dict) -> tuple[float, dict]:
    """
    Score based on career history.

    Key signals:
    - Product-company experience vs pure consulting
    - Relevant AI/ML roles in career history
    - Role trajectory (building → not just operating)
    - Company size progression (startup + scale-up preferred)
    - Recency of relevant experience
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = text_lower(profile.get("current_title", ""))
    current_industry = text_lower(profile.get("current_industry", ""))

    AI_ML_TITLES = {
        "ml engineer", "machine learning engineer", "ai engineer", "data scientist",
        "nlp engineer", "search engineer", "ranking engineer", "applied scientist",
        "research engineer", "mlops", "backend engineer", "software engineer",
        "senior engineer", "staff engineer", "principal engineer"
    }
    STRONG_AI_KEYWORDS = {
        "ranking", "retrieval", "embedding", "search", "recommendation", "nlp",
        "machine learning", "deep learning", "llm", "vector", "model", "inference",
        "rag", "fine-tun", "bert", "transformer", "neural", "semantic"
    }
    PRODUCT_COMPANY_INDICATORS = {
        "startup", "saas", "fintech", "edtech", "healthtech", "e-commerce",
        "product", "platform", "series"
    }

    career_score = 0.0
    consulting_only = True
    has_product_company = False
    relevant_months = 0
    career_evidence = []

    # Sort by recency
    sorted_career = sorted(
        career,
        key=lambda x: parse_date(x.get("start_date")) or date(2000, 1, 1),
        reverse=True
    )

    for i, job in enumerate(sorted_career):
        title = text_lower(job.get("title", ""))
        company = text_lower(job.get("company", ""))
        description = text_lower(job.get("description", ""))
        industry = text_lower(job.get("industry", ""))
        company_size = job.get("company_size", "")
        duration = job.get("duration_months", 0)
        is_current = job.get("is_current", False)

        # Recency weight (more recent = more weight)
        recency_weight = 1.0 if i == 0 else max(0.3, 1.0 - i * 0.15)

        # Check consulting-only pattern
        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        if not is_consulting:
            consulting_only = False

        # Product company detection
        size_map = {"1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
                    "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8}
        size_num = size_map.get(company_size, 3)
        is_product = (size_num <= 6 and not is_consulting) or \
                     any(pk in industry for pk in ["technology", "software", "saas", "ai", "fintech"])

        if is_product:
            has_product_company = True

        # Role relevance — check title + description
        title_relevance = sum(1 for kw in AI_ML_TITLES if kw in title)
        desc_relevance = sum(1 for kw in STRONG_AI_KEYWORDS if kw in description)

        role_score = min(1.0, title_relevance * 0.3 + desc_relevance * 0.07)
        duration_factor = min(1.0, duration / 24.0)  # 2 years = max
        job_contribution = role_score * duration_factor * recency_weight

        if role_score > 0.2:
            relevant_months += duration
            career_evidence.append({
                "title": job.get("title"),
                "company": job.get("company"),
                "role_score": round(role_score, 2),
                "is_product": is_product
            })

        career_score += job_contribution

    # Normalize by expected ideal score (~3 relevant jobs * 0.7 avg * 1.0 duration * avg recency)
    career_score = min(1.0, career_score / 2.5)

    # Penalties
    if consulting_only:
        career_score *= 0.4  # JD explicitly says pure consulting is disqualifying

    # Bonus for product company experience
    if has_product_company:
        career_score = min(1.0, career_score * 1.15)

    # Penalty: current title totally off (e.g. "marketing manager")
    off_titles = {"marketing", "sales", "hr", "finance", "accounting", "legal"}
    if any(ot in current_title for ot in off_titles):
        career_score *= 0.3

    evidence = {
        "consulting_only": consulting_only,
        "has_product_company": has_product_company,
        "relevant_months": relevant_months,
        "top_jobs": career_evidence[:3]
    }
    return career_score, evidence


def score_experience(candidate: dict) -> tuple[float, dict]:
    """
    Score years of experience against the JD sweet spot (5-9 years, ideal 6-8).
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)

    # JD says 5-9 years, ideal 6-8
    if 6 <= yoe <= 8:
        exp_score = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        exp_score = 0.85
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        exp_score = 0.65
    elif 3 <= yoe < 4 or 11 < yoe <= 15:
        exp_score = 0.45
    elif yoe >= 15:
        exp_score = 0.35  # overqualified / likely won't join
    else:
        exp_score = max(0.0, yoe / 4.0 * 0.4)  # very junior

    return exp_score, {"years_of_experience": yoe}


def score_location(candidate: dict) -> tuple[float, dict]:
    """
    Score based on location fit. JD prefers Pune/Noida/Delhi NCR/Hyderabad/Mumbai/Bangalore.
    Penalize candidates outside India unless willing to relocate.
    """
    profile = candidate.get("profile", {})
    location = text_lower(profile.get("location", ""))
    country = text_lower(profile.get("country", ""))
    signals = candidate.get("redrob_signals", {})
    willing_to_relocate = signals.get("willing_to_relocate", False)
    notice_period = signals.get("notice_period_days", 90)

    # Top preferred locations
    TOP_LOCATIONS = {"pune", "noida", "delhi ncr", "delhi", "ncr"}
    GOOD_LOCATIONS = {"hyderabad", "mumbai", "bangalore", "bengaluru", "gurugram", "gurgaon"}

    if any(loc in location for loc in TOP_LOCATIONS):
        loc_score = 1.0
    elif any(loc in location for loc in GOOD_LOCATIONS):
        loc_score = 0.85
    elif country in {"india", "in"}:
        loc_score = 0.6 if willing_to_relocate else 0.4
    else:
        # Outside India — JD says case-by-case, no visa sponsorship
        loc_score = 0.2 if willing_to_relocate else 0.05

    # Notice period factor (JD wants <30 days, can buy out 30 days)
    if notice_period <= 30:
        notice_bonus = 1.0
    elif notice_period <= 60:
        notice_bonus = 0.85
    elif notice_period <= 90:
        notice_bonus = 0.7
    else:
        notice_bonus = 0.5

    final_loc_score = loc_score * 0.7 + notice_bonus * 0.3

    return final_loc_score, {
        "location": profile.get("location"),
        "country": profile.get("country"),
        "notice_period_days": notice_period,
        "willing_to_relocate": willing_to_relocate
    }


def score_education(candidate: dict) -> tuple[float, dict]:
    """
    Score education. JD doesn't specify a degree requirement, but tier matters.
    CS/related field gets a bonus. Tier 1/2 gets bonus.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3, {"note": "No education listed"}

    RELEVANT_FIELDS = {
        "computer science", "cs", "machine learning", "artificial intelligence",
        "data science", "information technology", "statistics", "mathematics",
        "electrical engineering", "electronics", "software engineering"
    }

    best_score = 0.0
    best_entry = None

    TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}
    DEGREE_SCORES = {"phd": 1.0, "ph.d": 1.0, "m.tech": 0.9, "m.e.": 0.9, "mtech": 0.9,
                     "mba": 0.7, "b.tech": 0.75, "b.e.": 0.75, "btech": 0.75, "b.e": 0.75,
                     "m.sc": 0.85, "msc": 0.85, "b.sc": 0.6, "bsc": 0.6}

    for edu in education:
        degree = text_lower(edu.get("degree", ""))
        field = text_lower(edu.get("field_of_study", ""))
        tier = edu.get("tier", "unknown")

        tier_score = TIER_SCORES.get(tier, 0.5)
        degree_score = 0.5
        for d, ds in DEGREE_SCORES.items():
            if d in degree:
                degree_score = ds
                break

        field_bonus = 0.15 if any(rf in field for rf in RELEVANT_FIELDS) else 0.0

        score = tier_score * 0.5 + degree_score * 0.4 + field_bonus
        if score > best_score:
            best_score = score
            best_entry = edu

    return min(1.0, best_score), {
        "best_institution": best_entry.get("institution") if best_entry else None,
        "best_degree": best_entry.get("degree") if best_entry else None,
        "field": best_entry.get("field_of_study") if best_entry else None
    }


def compute_behavioral_multiplier(candidate: dict) -> tuple[float, dict]:
    """
    Behavioral multiplier based on Redrob platform signals.
    A great-on-paper candidate who isn't reachable is useless.
    Multiplier range: 0.3 to 1.2
    """
    signals = candidate.get("redrob_signals", {})

    # 1. Availability signals
    open_to_work = signals.get("open_to_work_flag", False)
    last_active_months = months_since(signals.get("last_active_date"))
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)

    if last_active_months > 6:
        activity_score = max(0.2, 1.0 - (last_active_months - 6) / 12.0)
    elif last_active_months > 3:
        activity_score = 0.8
    else:
        activity_score = 1.0

    availability = (
        (0.3 if open_to_work else 0.0) +
        activity_score * 0.4 +
        (0.1 if verified_email else 0.0) +
        (0.1 if verified_phone else 0.0) +
        (0.1 if signals.get("linkedin_connected", False) else 0.0)
    )

    # 2. Recruiter engagement signals
    response_rate = signals.get("recruiter_response_rate", 0.5)
    avg_response_hrs = signals.get("avg_response_time_hours", 48)
    interview_rate = signals.get("interview_completion_rate", 0.5)

    response_time_score = max(0.0, 1.0 - avg_response_hrs / 96.0)
    engagement = (response_rate * 0.4 + response_time_score * 0.3 + interview_rate * 0.3)

    # 3. Platform credibility signals
    profile_completeness = signals.get("profile_completeness_score", 50) / 100.0
    saved_by_recruiters = min(1.0, signals.get("saved_by_recruiters_30d", 0) / 10.0)
    github_score = signals.get("github_activity_score", -1)
    github_factor = (github_score / 100.0 * 0.5) if github_score >= 0 else 0.25

    credibility = (profile_completeness * 0.4 + saved_by_recruiters * 0.3 + github_factor * 0.3)

    # Combine into multiplier
    combined = availability * 0.45 + engagement * 0.35 + credibility * 0.20
    # Map [0, 1] → [0.3, 1.2]
    multiplier = 0.3 + combined * 0.9

    evidence = {
        "open_to_work": open_to_work,
        "last_active_months_ago": round(last_active_months, 1),
        "response_rate": response_rate,
        "interview_completion_rate": interview_rate,
        "github_activity_score": github_score,
        "profile_completeness": signals.get("profile_completeness_score"),
    }
    return multiplier, evidence


def detect_honeypot(candidate: dict) -> tuple[float, list[str]]:
    """
    Detect honeypot candidates with impossible or fraudulent profiles.
    Returns a penalty multiplier (1.0 = clean, 0.01 = honeypot) and a list of flags.
    """
    flags = []
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})

    # Check 1: Expert skills with 0 duration_months (keyword stuffing)
    expert_zero_duration = [
        sk["name"] for sk in skills
        if sk.get("proficiency") == "expert" and sk.get("duration_months", -1) == 0
    ]
    if len(expert_zero_duration) >= EXPERT_SKILLS_0_MONTHS_THRESHOLD:
        flags.append(f"Expert proficiency in {len(expert_zero_duration)} skills with 0 months duration")

    # Check 2: Career duration inconsistency
    # If someone has 3 years of experience but their job durations sum to 10+ years
    yoe = profile.get("years_of_experience", 0)
    total_months_from_history = sum(j.get("duration_months", 0) for j in career)
    if total_months_from_history > (yoe + 2) * 12 + 24:
        flags.append(f"Career duration mismatch: claimed {yoe}y but history totals {total_months_from_history//12}y")

    # Check 3: Impossible date ranges (end before start)
    for job in career:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start and end and end < start:
            flags.append(f"Impossible date range at {job.get('company')}: end before start")

    # Check 4: Duration mismatch (listed duration_months vs actual date span)
    for job in career:
        start = parse_date(job.get("start_date"))
        end_raw = job.get("end_date")
        end = parse_date(end_raw) if end_raw else REFERENCE_DATE
        stated = job.get("duration_months", 0)
        if start and end:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(actual_months - stated) > 18:  # allow 18 month slop for rounding
                flags.append(
                    f"Duration mismatch at {job.get('company')}: stated {stated}mo, actual ~{actual_months}mo"
                )

    # Check 5: Overlapping job periods
    sorted_jobs = sorted(
        [(parse_date(j.get("start_date")), parse_date(j.get("end_date")) or REFERENCE_DATE, j.get("company"))
         for j in career if parse_date(j.get("start_date"))],
        key=lambda x: x[0]
    )
    for i in range(len(sorted_jobs) - 1):
        s1, e1, c1 = sorted_jobs[i]
        s2, e2, c2 = sorted_jobs[i + 1]
        overlap_days = (min(e1, e2) - max(s1, s2)).days
        if overlap_days > 365:  # >1 year concurrent is suspicious
            flags.append(f"Suspicious overlap of {overlap_days}d between {c1} and {c2}")

    # Check 6: Perfect-score on all assessments (bot-like)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if len(assessments) >= 5:
        perfect = sum(1 for v in assessments.values() if v >= 99)
        if perfect >= 4:
            flags.append(f"{perfect} out of {len(assessments)} assessments scored 99-100 (suspicious)")

    penalty = 1.0
    if len(flags) >= 2:
        penalty = 0.05  # clear honeypot
    elif len(flags) == 1:
        penalty = 0.6  # suspicious

    return penalty, flags


# ─── REASONING GENERATOR ─────────────────────────────────────────────────────

def generate_reasoning(cs: CandidateScore, candidate: dict) -> str:
    """
    Generate a 1-2 sentence evidence-based reasoning string.
    Uses actual data from the candidate profile — no hallucination.
    """
    profile = candidate.get("profile", {})
    ev = cs.evidence

    parts = []

    # Lead with the strongest signal
    top_jobs = ev.get("career", {}).get("top_jobs", [])
    if top_jobs:
        j = top_jobs[0]
        parts.append(f"{profile.get('current_title', 'Engineer')} with {profile.get('years_of_experience', '?')}y experience")
        if j.get("is_product"):
            parts.append(f"; prior product-company work at {j.get('company', 'a tech company')}")
    else:
        parts.append(f"{profile.get('current_title', 'Engineer')} with {profile.get('years_of_experience', '?')}y experience")

    # Skill highlights
    must_hits = ev.get("skills", {}).get("must_have_hits", [])
    if must_hits:
        skills_str = ", ".join(must_hits[:3])
        parts.append(f"; strong on {skills_str}")

    # Location / notice
    loc_ev = ev.get("location", {})
    notice = loc_ev.get("notice_period_days")
    location = loc_ev.get("location", "")
    if notice is not None and notice <= 30:
        parts.append(f"; immediate availability ({notice}d notice)")
    elif notice is not None and notice > 90:
        parts.append(f"; long notice period ({notice}d) is a concern")

    # Behavioral caveats
    beh = ev.get("behavioral", {})
    last_active = beh.get("last_active_months_ago", 0)
    resp_rate = beh.get("response_rate", 1.0)
    if last_active > 4:
        parts.append(f"; inactive for ~{int(last_active)}mo")
    if resp_rate < 0.3:
        parts.append(f"; low recruiter response rate ({int(resp_rate*100)}%)")

    # Honeypot flags
    if ev.get("honeypot_flags"):
        parts.append(f"; flagged: {ev['honeypot_flags'][0][:60]}")

    # Consulting penalty note
    if ev.get("career", {}).get("consulting_only"):
        parts.append("; entire career in consulting/services (JD disqualifies pure consulting)")

    sentence = "".join(parts).strip("; ").capitalize()
    if not sentence:
        sentence = f"Candidate with {profile.get('years_of_experience', '?')}y experience at {profile.get('current_company', 'unknown company')}"

    # Score context for rank calibration
    score_context = f" Score: {cs.final_score:.3f}."
    return sentence + "." + score_context


# ─── MAIN RANKER ─────────────────────────────────────────────────────────────

def score_candidate(candidate: dict) -> CandidateScore:
    cid = candidate.get("candidate_id", "UNKNOWN")
    cs = CandidateScore(candidate_id=cid)

    cs.skill_score, skill_ev = score_skills(candidate)
    cs.career_score, career_ev = score_career(candidate)
    cs.experience_score, exp_ev = score_experience(candidate)
    cs.location_score, loc_ev = score_location(candidate)
    cs.education_score, edu_ev = score_education(candidate)
    cs.behavioral_multiplier, beh_ev = compute_behavioral_multiplier(candidate)
    cs.honeypot_penalty, honeypot_flags = detect_honeypot(candidate)

    cs.evidence = {
        "skills": skill_ev,
        "career": career_ev,
        "experience": exp_ev,
        "location": loc_ev,
        "education": edu_ev,
        "behavioral": beh_ev,
        "honeypot_flags": honeypot_flags
    }

    cs.compute_final()
    return cs


def rank_candidates(input_path: str, output_path: str, top_n: int = 100):
    """Main entry point. Reads JSONL (plain or gzipped), scores all, writes top-N CSV."""
    t0 = time.time()

    # Open gzip or plain
    path = Path(input_path)
    opener = gzip.open if path.suffix == ".gz" else open

    print(f"Loading candidates from {input_path}...")
    scores = []
    count = 0

    with opener(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                cs = score_candidate(candidate)
                scores.append((cs, candidate))
                count += 1
                if count % 10000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Processed {count:,} candidates in {elapsed:.1f}s "
                          f"({count/elapsed:.0f}/s)...")
            except json.JSONDecodeError:
                continue

    print(f"Scored {count:,} candidates in {time.time()-t0:.1f}s. Sorting...")

    # Sort by final_score DESC, then candidate_id ASC for tie-break
    scores.sort(key=lambda x: (-x[0].final_score, x[0].candidate_id))

    top = scores[:top_n]

    # Normalize scores to [0.01, 0.999] range for clean output
    if top:
        max_score = top[0][0].final_score
        min_score = top[-1][0].final_score
        score_range = max_score - min_score or 1.0

    # Write CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cs, candidate) in enumerate(top, start=1):
            # Scale scores so rank 1 ≈ 0.999 and rank 100 ≈ 0.100
            normalized = 0.100 + (cs.final_score - min_score) / score_range * 0.899
            reasoning = generate_reasoning(cs, candidate)
            writer.writerow([cs.candidate_id, rank, f"{normalized:.4f}", reasoning])

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s. Written {top_n} candidates to {output_path}")
    print(f"   Top-5 candidates:")
    for rank, (cs, cand) in enumerate(top[:5], 1):
        p = cand.get("profile", {})
        print(f"   #{rank}: {cs.candidate_id} | {p.get('current_title')} | "
              f"score={cs.final_score:.4f} | {p.get('location')}, {p.get('country')}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python ranker.py <candidates.jsonl[.gz]> <output.csv>")
        sys.exit(1)
    rank_candidates(sys.argv[1], sys.argv[2])
