# 🎯 HireScore — Intelligent Recruiter Ranking Portal

> **Hackathon: Intelligent Candidate Discovery & Ranking Challenge**
> Rank candidates the way a great recruiter would — not by matching keywords, but by understanding who genuinely fits the role.

🚀 **Live Demo**: [HireScore on HuggingFace](https://huggingface.co/spaces/bhoomichowksey/redrob-ranker-demo)

---

## 💡 The Problem

Recruiters go through hundreds of profiles and still often miss the right person. Not because the talent isn't there — but because keyword filters can't see what actually matters.

HireScore solves this by ranking candidates across **5 intelligent dimensions** with a **behavioral multiplier** — the way a seasoned recruiter actually thinks.

---

## 🏗️ What We Built

A **multi-tenant recruiter portal** where companies sign in with their credentials and get a department-specific candidate ranking system. Each department has its own scoring weights and JD focus — so the same candidate scores differently for a Machine Learning role vs a Data Analytics role.

### Departments Supported
| Department | Skills Weight | Career Weight | Key JD Focus |
|---|---|---|---|
| Machine Learning | 42% | 33% | PyTorch, LLM, RAG, Transformers |
| Software Engineering | 38% | 32% | React, Node, System Design, Docker |
| Data Science | 35% | 30% | Python, SQL, Spark, Airflow |
| Data Analytics | 32% | 28% | SQL, Tableau, A/B Testing, Cohort |
| Data Engineering | 38% | 30% | Spark, Kafka, Airflow, dbt |
| AI Research | 45% | 28% | PyTorch, RL, NLP, Fine-tuning |
| Product Analytics | 30% | 30% | SQL, Growth, Funnel, Retention |
| Engineering | 40% | 35% | Python, ML, FastAPI, MLOps |

---

## 🧠 Scoring Formula

```
Final Score = (Skills × W₁ + Career × W₂ + Experience × W₃ + Location × W₄ + Education × W₅) × Behavioral Multiplier
```

### Dimension Breakdown

**1. Skills Score (30–45% depending on dept)**
- Matches candidate skills against department-specific JD keywords
- Weights proficiency level (1–5), duration (months), and JD relevance
- Penalizes shallow or irrelevant skills

**2. Career Score (28–35%)**
- Product company vs consulting vs other (1.0 / 0.28 / 0.60)
- Role title relevance to department JD keywords
- Rewards candidates from high-signal companies (Swiggy, Razorpay, Google, etc.)

**3. Experience Score (14–20%)**
- Sweet spot: 6–8 years scores 1.0
- Penalizes both under-experience (<4 yrs) and over-experience (>14 yrs)

**4. Location Score (5–15%)**
- Top cities: Bengaluru, Mumbai, Hyderabad, Pune, Delhi NCR
- Notice period: 0 days = 1.0, 15 days = 0.90, 30 days = 0.75, 60+ days = 0.20

**5. Education Score (3–12%)**
- Tier-1 institutions: IIT, IIM, BITS Pilani, NIT, IIIT = 1.0
- Others = 0.55

**Behavioral Multiplier (0.3× – 1.2×)**
- Open to work status
- Recruiter response rate
- GitHub activity score

---

## 🔐 Multi-Tenant Auth

| Company | Recruiter ID | Dept | Theme |
|---|---|---|---|
| SWIGGY | REC001 | Engineering | Orange |
| RAZORPAY | REC002 | Data Science | Purple |
| MEESHO | REC003 | Product Analytics | Teal |

New companies can sign up and pick their department — scoring weights auto-configure.

---

## 📁 Repo Structure

```
├── app.py                    # Flask server / Gradio wrapper
├── portal.html               # Full recruiter portal UI
├── Dockerfile                # HuggingFace Docker deployment
├── ranked_candidates.csv     # Sample ranked output
├── approach.pdf              # Methodology deck
└── README.md
```

---

## 🚀 Run Locally

```bash
# Install dependencies
pip install flask

# Run
python app.py

# Open browser at http://localhost:7860
```

---

## 📊 Sample Output

See `ranked_candidates.csv` for a full ranked candidate output with all dimension scores.

---

## 🏆 Built for the Redrob × H2S Hackathon
