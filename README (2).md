# Redrob Intelligent Candidate Ranker

**Hackathon: Intelligent Candidate Discovery & Ranking Challenge**

> Ranking candidates the way a great recruiter would — not by matching keywords, but by understanding who genuinely fits the role.

---

## Quick Start

```bash
# 1. Install dependencies (all stdlib except numpy)
pip install numpy

# 2. Run the ranker
python src/ranker.py candidates.jsonl outputs/team_submission.csv

# 3. Validate
python validate_submission.py outputs/team_submission.csv
# → Submission is valid.
```

**Runtime:** ~53 seconds for 100,000 candidates on a standard CPU (no GPU, no API calls).

---

## Architecture

The ranker scores each candidate on 6 dimensions, then applies a behavioral multiplier:

```
Final Score = (0.35 × Skills + 0.35 × Career + 0.15 × Experience
             + 0.10 × Location + 0.05 × Education)
             × Behavioral_Multiplier
             × Honeypot_Penalty
```

### Scoring Dimensions

| Dimension | Weight | Key signals |
|-----------|--------|-------------|
| **Skills** | 35% | Proficiency × endorsements × duration_months × assessment score |
| **Career History** | 35% | Product company vs consulting, role relevance, recency weighting |
| **Experience Years** | 15% | Sweet spot: 5–9y (ideal 6–8y per JD) |
| **Location** | 10% | Pune/Noida/Delhi NCR top-tier; notice period scoring |
| **Education** | 5% | Institution tier + degree level + relevant field bonus |
| **Behavioral ×** | Multiplier (0.3–1.2×) | Availability, engagement, credibility signals |

### Honeypot Detection

Six checks catch impossible/fraudulent profiles:

1. Expert proficiency with 0 months used (≥5 skills → flag)
2. Career duration math mismatch vs profile YOE
3. End date before start date
4. Stated `duration_months` vs actual date span divergence (>18mo)
5. Job period overlaps >1 year
6. Near-perfect assessment scores across many skills

Two or more flags → 0.05× penalty (effective disqualification from top 100).

### Behavioral Multiplier (Redrob Signals)

| Category | Weight | Signals |
|----------|--------|---------|
| Availability | 45% | open_to_work, last_active_date, verified email/phone |
| Engagement | 35% | recruiter_response_rate, avg_response_time_hours, interview_completion_rate |
| Credibility | 20% | profile_completeness_score, saved_by_recruiters_30d, github_activity_score |

Multiplier range: **0.3× to 1.2×** — a ghost candidate gets penalized heavily.

---

## Key Design Decisions

**Why rule-based + signal weighting instead of LLM per candidate?**

The 5-minute CPU budget makes per-candidate LLM inference impossible at 100K scale. Instead, we encode the JD's *actual meaning* into explicit scoring rules — which also makes the system fully transparent, auditable, and non-hallucinating.

**Why is career history weighted the same as skills (35% each)?**

The JD is explicit: someone with all the AI keywords but a pure-consulting career or a "Marketing Manager" title is not a fit. Career trajectory is the primary signal for separating genuine ML engineers from keyword stuffers.

**Why the behavioral multiplier instead of a hard filter?**

Hard filters lose candidates who are "slightly inactive" but still reachable. A multiplier preserves ordering while gracefully down-weighting candidates who are statistically less likely to respond to recruiter outreach.

---

## File Structure

```
redrob-ranker/
├── src/
│   └── ranker.py          # Complete ranking logic (~400 lines)
├── outputs/
│   └── team_submission.csv  # The actual submission
├── validate_submission.py   # Validator (from challenge bundle)
├── README.md
└── submission_metadata.yaml
```

---

## Compute Compliance

| Constraint | Limit | Our result |
|-----------|-------|-----------|
| Runtime | ≤ 5 min | **53 seconds** |
| RAM | ≤ 16 GB | < 2 GB |
| Compute | CPU only | ✅ Pure Python + numpy |
| Network | Offline | ✅ Zero external calls |
| Disk | ≤ 5 GB | < 10 MB intermediate |
