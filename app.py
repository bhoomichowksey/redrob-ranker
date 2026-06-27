import gradio as gr
import json
import numpy as np
from datetime import datetime, date
import re

# ── JD Constants (hardcoded for demo) ────────────────────────────────────────
JD_SKILLS = {
    "python": 0.95, "machine learning": 0.90, "deep learning": 0.85,
    "pytorch": 0.85, "tensorflow": 0.80, "nlp": 0.85,
    "retrieval": 0.90, "embeddings": 0.90, "vector search": 0.88,
    "rag": 0.88, "llm": 0.85, "sql": 0.70, "aws": 0.65,
    "docker": 0.60, "fastapi": 0.65, "transformers": 0.85,
    "fine-tuning": 0.80, "mlops": 0.75, "recommendation": 0.80,
}
PRODUCT_COMPANIES = {
    "google","meta","amazon","microsoft","apple","netflix","openai",
    "anthropic","flipkart","swiggy","zomato","meesho","cred","razorpay",
    "zepto","groww","phonepe","paytm","freshworks","zoho","atlassian",
    "adobe","nvidia","huggingface","cohere","mistral","ola","byju",
    "unacademy","sharechat","moj","slice","jupiter",
}
CONSULTING_FIRMS = {
    "accenture","tcs","infosys","wipro","cognizant","capgemini","hcl",
    "tech mahindra","deloitte","pwc","kpmg","ey","ibm global","mindtree",
    "mphasis","hexaware","niit","l&t infotech",
}
TOP_LOCATIONS = {"pune","noida","delhi","gurugram","gurgaon","bengaluru","bangalore","hyderabad","mumbai"}
TIER1_COLLEGES = {
    "iit","iim","bits pilani","nit trichy","nit warangal","dtu","iiit hyderabad",
    "vit","srm","manipal","pes university","nitk","iit bombay","iit delhi",
    "iit madras","iit kharagpur","iit kanpur","iit roorkee","iit guwahati",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_date(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    return None

def days_ago(d):
    if not d:
        return 9999
    try:
        return (date.today() - d).days
    except:
        return 9999

# ── Scoring Functions ─────────────────────────────────────────────────────────
def score_skills(skills):
    if not skills:
        return 0.0
    total, max_possible = 0.0, 0.0
    for skill in skills:
        name = skill.get("name", "").lower()
        prof = skill.get("proficiency_level", 3) / 5.0
        endorse = min(skill.get("endorsements", 0) / 20.0, 1.0)
        months = min(skill.get("duration_months", 0) / 36.0, 1.0)
        assess = skill.get("assessment_score", 0.5)
        jd_weight = 0.0
        for jd_skill, w in JD_SKILLS.items():
            if jd_skill in name or name in jd_skill:
                jd_weight = w
                break
        if jd_weight > 0:
            signal = (0.35 * prof + 0.25 * endorse + 0.25 * months + 0.15 * assess) * jd_weight
            total += signal
            max_possible += jd_weight
    return min(total / max_possible, 1.0) if max_possible > 0 else 0.0

def score_career(jobs):
    if not jobs:
        return 0.0
    score, total_w = 0.0, 0.0
    today = date.today()
    sorted_jobs = sorted(jobs, key=lambda j: parse_date(j.get("end_date")) or today, reverse=True)
    for i, job in enumerate(sorted_jobs[:5]):
        recency = 1.0 / (1 + i * 0.3)
        title = (job.get("title", "") + " " + job.get("role_category", "")).lower()
        company = job.get("company_name", "").lower()
        is_product = any(pc in company for pc in PRODUCT_COMPANIES)
        is_consulting = any(cf in company for cf in CONSULTING_FIRMS)
        company_score = 1.0 if is_product else (0.3 if is_consulting else 0.65)
        ml_keywords = ["ml","machine learning","ai","data scientist","nlp","deep learning",
                       "research","engineer","retrieval","recommendation","llm"]
        role_score = 0.9 if any(k in title for k in ml_keywords) else 0.3
        score += recency * (0.5 * company_score + 0.5 * role_score)
        total_w += recency
    return min(score / total_w, 1.0) if total_w > 0 else 0.0

def score_experience(yoe):
    if yoe is None:
        return 0.0
    if 6 <= yoe <= 8:
        return 1.0
    elif 5 <= yoe < 6:
        return 0.85
    elif 8 < yoe <= 10:
        return 0.80
    elif 4 <= yoe < 5:
        return 0.65
    elif 10 < yoe <= 14:
        return 0.55
    elif yoe < 4:
        return 0.35
    return 0.20

def score_location(loc, notice):
    loc = (loc or "").lower()
    loc_score = 1.0 if any(l in loc for l in TOP_LOCATIONS) else 0.4
    if notice is None:
        notice_score = 0.5
    elif notice == 0:
        notice_score = 1.0
    elif notice <= 15:
        notice_score = 0.90
    elif notice <= 30:
        notice_score = 0.75
    elif notice <= 60:
        notice_score = 0.50
    else:
        notice_score = 0.20
    return 0.6 * loc_score + 0.4 * notice_score

def score_education(edu):
    if not edu:
        return 0.4
    best = 0.0
    for e in edu:
        inst = (e.get("institution", "") + " " + e.get("institution_name", "")).lower()
        degree = e.get("degree", "").lower()
        field = (e.get("field_of_study", "") + " " + e.get("major", "")).lower()
        tier = 0.55
        if any(t in inst for t in TIER1_COLLEGES):
            tier = 1.0
        elif "nit" in inst or "iiit" in inst:
            tier = 0.85
        deg_bonus = 0.2 if "ph" in degree else (0.1 if "master" in degree or " m." in degree else 0.0)
        field_bonus = 0.1 if any(f in field for f in ["computer","data","machine","ai","statistics"]) else 0.0
        best = max(best, min(tier + deg_bonus + field_bonus, 1.0))
    return best

def behavioral_multiplier(beh):
    if not beh:
        return 0.7
    avail, engage, cred = 0.0, 0.0, 0.0
    # Availability
    otw = 1.0 if beh.get("open_to_work") else 0.3
    last = parse_date(beh.get("last_active_date"))
    active_score = 1.0 if days_ago(last) <= 7 else (0.8 if days_ago(last) <= 30 else (0.5 if days_ago(last) <= 90 else 0.2))
    email_v = 0.8 if beh.get("verified_email") else 0.3
    phone_v = 0.2 if beh.get("verified_phone") else 0.0
    avail = 0.4 * otw + 0.35 * active_score + 0.15 * email_v + 0.10 * phone_v
    # Engagement
    rr = min(beh.get("recruiter_response_rate", 0.5), 1.0)
    rt = beh.get("avg_response_time_hours", 48)
    rt_score = 1.0 if rt <= 2 else (0.8 if rt <= 6 else (0.6 if rt <= 24 else 0.3))
    ic = min(beh.get("interview_completion_rate", 0.7), 1.0)
    engage = 0.45 * rr + 0.30 * rt_score + 0.25 * ic
    # Credibility
    pc = min(beh.get("profile_completeness_score", 0.7), 1.0)
    saved = min(beh.get("saved_by_recruiters_30d", 0) / 10.0, 1.0)
    gh = min(beh.get("github_activity_score", 0.5), 1.0)
    cred = 0.35 * pc + 0.30 * saved + 0.35 * gh
    raw = 0.45 * avail + 0.35 * engage + 0.20 * cred
    return 0.3 + 0.9 * raw  # range 0.3–1.2

def honeypot_penalty(candidate):
    flags = 0
    skills = candidate.get("skills", [])
    # Flag 1: expert with 0 months
    zero_month_experts = sum(
        1 for s in skills
        if s.get("proficiency_level", 0) >= 5 and s.get("duration_months", 1) == 0
    )
    if zero_month_experts >= 5:
        flags += 1
    # Flag 2: overlapping jobs
    jobs = candidate.get("career_history", [])
    today = date.today()
    periods = []
    for j in jobs:
        s = parse_date(j.get("start_date"))
        e = parse_date(j.get("end_date")) or today
        if s and e and e > s:
            periods.append((s, e))
    periods.sort()
    overlap_days = 0
    for i in range(1, len(periods)):
        if periods[i][0] < periods[i-1][1]:
            overlap_days += (periods[i-1][1] - periods[i][0]).days
    if overlap_days > 365:
        flags += 1
    # Flag 3: near-perfect assessment across many skills
    high_assess = sum(1 for s in skills if s.get("assessment_score", 0) >= 0.97)
    if high_assess >= 8:
        flags += 1
    return 0.05 if flags >= 2 else 1.0

def rank_candidate(c):
    s_skills = score_skills(c.get("skills", []))
    s_career = score_career(c.get("career_history", []))
    s_exp = score_experience(c.get("years_of_experience"))
    s_loc = score_location(c.get("current_location"), c.get("notice_period_days"))
    s_edu = score_education(c.get("education", []))
    beh_mult = behavioral_multiplier(c.get("behavioral_signals", {}))
    hp = honeypot_penalty(c)
    raw = (0.35 * s_skills + 0.35 * s_career + 0.15 * s_exp + 0.10 * s_loc + 0.05 * s_edu)
    final = raw * beh_mult * hp
    return {
        "final_score": round(final, 4),
        "skills_score": round(s_skills, 3),
        "career_score": round(s_career, 3),
        "experience_score": round(s_exp, 3),
        "location_score": round(s_loc, 3),
        "education_score": round(s_edu, 3),
        "behavioral_multiplier": round(beh_mult, 3),
        "honeypot_penalty": hp,
    }

# ── Gradio UI ─────────────────────────────────────────────────────────────────
SAMPLE_JSON = json.dumps([
  {
    "candidate_id": "C001",
    "name": "Priya Sharma",
    "years_of_experience": 7,
    "current_location": "Pune, Maharashtra",
    "notice_period_days": 15,
    "skills": [
      {"name": "Python", "proficiency_level": 5, "endorsements": 28, "duration_months": 72, "assessment_score": 0.91},
      {"name": "Machine Learning", "proficiency_level": 5, "endorsements": 22, "duration_months": 60, "assessment_score": 0.88},
      {"name": "Retrieval", "proficiency_level": 4, "endorsements": 15, "duration_months": 36, "assessment_score": 0.85},
      {"name": "Embeddings", "proficiency_level": 4, "endorsements": 12, "duration_months": 30, "assessment_score": 0.82},
      {"name": "PyTorch", "proficiency_level": 4, "endorsements": 18, "duration_months": 48, "assessment_score": 0.87}
    ],
    "career_history": [
      {"title": "Senior ML Engineer", "company_name": "Swiggy", "role_category": "Machine Learning", "start_date": "2021-06", "end_date": None},
      {"title": "ML Engineer", "company_name": "Meesho", "role_category": "Machine Learning", "start_date": "2019-01", "end_date": "2021-05"}
    ],
    "education": [
      {"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "Computer Science"}
    ],
    "behavioral_signals": {
      "open_to_work": True,
      "last_active_date": "2025-06-20",
      "verified_email": True,
      "verified_phone": True,
      "recruiter_response_rate": 0.92,
      "avg_response_time_hours": 4,
      "interview_completion_rate": 0.95,
      "profile_completeness_score": 0.97,
      "saved_by_recruiters_30d": 8,
      "github_activity_score": 0.85
    }
  },
  {
    "candidate_id": "C002",
    "name": "Rahul Verma",
    "years_of_experience": 3,
    "current_location": "Jaipur, Rajasthan",
    "notice_period_days": 90,
    "skills": [
      {"name": "Python", "proficiency_level": 3, "endorsements": 5, "duration_months": 24, "assessment_score": 0.70},
      {"name": "Machine Learning", "proficiency_level": 3, "endorsements": 4, "duration_months": 20, "assessment_score": 0.65}
    ],
    "career_history": [
      {"title": "Data Analyst", "company_name": "TCS", "role_category": "Consulting", "start_date": "2022-01", "end_date": None}
    ],
    "education": [
      {"institution": "State University", "degree": "B.Tech", "field_of_study": "Electronics"}
    ],
    "behavioral_signals": {
      "open_to_work": False,
      "last_active_date": "2024-11-01",
      "verified_email": False,
      "verified_phone": False,
      "recruiter_response_rate": 0.30,
      "avg_response_time_hours": 72,
      "interview_completion_rate": 0.50,
      "profile_completeness_score": 0.55,
      "saved_by_recruiters_30d": 0,
      "github_activity_score": 0.20
    }
  }
], indent=2)

def run_ranker(json_input):
    try:
        candidates = json.loads(json_input)
        if not isinstance(candidates, list):
            candidates = [candidates]
    except Exception as e:
        return f"❌ Invalid JSON: {e}", ""

    results = []
    for c in candidates:
        scores = rank_candidate(c)
        results.append({
            "rank": 0,
            "candidate_id": c.get("candidate_id", "N/A"),
            "name": c.get("name", "Unknown"),
            "final_score": scores["final_score"],
            "skills_score": scores["skills_score"],
            "career_score": scores["career_score"],
            "experience_score": scores["experience_score"],
            "behavioral_multiplier": scores["behavioral_multiplier"],
            "honeypot": "🚩 FLAGGED" if scores["honeypot_penalty"] < 1.0 else "✅ Clean",
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    summary_lines = []
    for r in results:
        summary_lines.append(
            f"#{r['rank']} {r['name']} ({r['candidate_id']}) — Score: {r['final_score']:.4f}\n"
            f"   Skills: {r['skills_score']} | Career: {r['career_score']} | "
            f"Exp: {r['experience_score']} | Behavioral×: {r['behavioral_multiplier']} | {r['honeypot']}"
        )

    output = "## 🏆 Ranked Candidates\n\n" + "\n\n".join(summary_lines)
    csv_lines = ["rank,candidate_id,name,final_score,skills_score,career_score,experience_score,behavioral_multiplier,honeypot_flag"]
    for r in results:
        csv_lines.append(f"{r['rank']},{r['candidate_id']},{r['name']},{r['final_score']},{r['skills_score']},{r['career_score']},{r['experience_score']},{r['behavioral_multiplier']},{r['honeypot']}")
    csv_out = "\n".join(csv_lines)

    return output, csv_out

with gr.Blocks(title="Redrob Intelligent Candidate Ranker", theme=gr.themes.Base()) as demo:
    gr.Markdown("""
# 🎯 Redrob Intelligent Candidate Ranker
**Hackathon: Intelligent Candidate Discovery & Ranking Challenge**

Ranks candidates the way a great recruiter would — using skills, career trajectory, experience, location, education, and behavioral signals.

**Scoring Formula:**
`Final Score = (0.35 × Skills + 0.35 × Career + 0.15 × Experience + 0.10 × Location + 0.05 × Education) × Behavioral_Multiplier × Honeypot_Penalty`
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Candidates (JSON Array)")
            json_input = gr.Textbox(
                value=SAMPLE_JSON,
                lines=25,
                label="Paste candidate JSON here",
                placeholder="Paste a JSON array of candidates..."
            )
            run_btn = gr.Button("🚀 Rank Candidates", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Ranked Results")
            output_text = gr.Markdown(label="Ranking Results")
            gr.Markdown("### 📄 CSV Output")
            csv_output = gr.Textbox(lines=8, label="CSV (copy-paste ready)", show_copy_button=True)

    gr.Markdown("""
---
### 📐 How Scoring Works

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Skills | 35% | Proficiency × endorsements × duration × assessment |
| Career | 35% | Product co. vs consulting, ML role relevance, recency |
| Experience | 15% | Sweet spot: 5–9 years (ideal 6–8 per JD) |
| Location | 10% | Pune/Noida/Delhi NCR preferred + short notice period |
| Education | 5% | Institution tier + degree level + relevant field |
| Behavioral × | Multiplier | Availability, engagement, credibility (0.3×–1.2×) |

**Honeypot Detection** catches fraudulent profiles: expert with 0 months use, overlapping jobs >1yr, near-perfect scores → 0.05× penalty.
    """)

    run_btn.click(fn=run_ranker, inputs=[json_input], outputs=[output_text, csv_output])

if __name__ == "__main__":
    demo.launch()
