 import gradio as gr
import json
from datetime import datetime, date

JD_SKILLS = {
    "python": 0.95, "machine learning": 0.90, "deep learning": 0.85,
    "pytorch": 0.85, "tensorflow": 0.80, "nlp": 0.85,
    "retrieval": 0.90, "embeddings": 0.90, "vector search": 0.88,
    "rag": 0.88, "llm": 0.85, "sql": 0.70, "aws": 0.65,
    "docker": 0.60, "fastapi": 0.65, "transformers": 0.85,
    "fine-tuning": 0.80, "mlops": 0.75, "recommendation": 0.80,
}
PRODUCT_COS = ["google","meta","amazon","microsoft","apple","netflix","openai","anthropic",
                "flipkart","swiggy","zomato","meesho","cred","razorpay","zepto","groww",
                "phonepe","paytm","freshworks","zoho","atlassian","adobe","nvidia"]
CONSULT = ["accenture","tcs","infosys","wipro","cognizant","capgemini","hcl",
           "tech mahindra","deloitte","pwc","kpmg","ibm","mindtree","mphasis"]
TOP_LOCS = ["pune","noida","delhi","gurugram","gurgaon","bengaluru","bangalore","hyderabad","mumbai"]
TIER1 = ["iit","bits pilani","nit","iiit","dtu","vit","srm","manipal"]

def parse_skills(raw):
    skills = []
    for line in raw.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if not parts[0]: continue
        name = parts[0]
        try: prof = int(parts[1]) if len(parts) > 1 else 3
        except: prof = 3
        try: months = int(parts[2]) if len(parts) > 2 else 24
        except: months = 24
        skills.append({"name": name, "proficiency_level": prof, "duration_months": months, "endorsements": 10, "assessment_score": 0.78})
    return skills

def score_skills(skills):
    if not skills: return 0.0
    total, maxw = 0.0, 0.0
    for s in skills:
        n = s["name"].lower()
        w = next((v for k, v in JD_SKILLS.items() if k in n or n in k), 0)
        if w > 0:
            p = s["proficiency_level"] / 5.0
            m = min(s["duration_months"] / 36.0, 1.0)
            total += (0.35*p + 0.25*0.5 + 0.25*m + 0.15*s["assessment_score"]) * w
            maxw += w
    return min(total / maxw, 1.0) if maxw > 0 else 0.0

def score_career(company, role):
    co, ro = company.lower(), role.lower()
    is_prod = any(p in co for p in PRODUCT_COS)
    is_con = any(c in co for c in CONSULT)
    co_s = 1.0 if is_prod else (0.28 if is_con else 0.60)
    ml_kw = ["ml","machine learning","ai","data scientist","nlp","deep learning","retrieval","llm","engineer","research"]
    ro_s = 0.9 if any(k in ro for k in ml_kw) else 0.3
    return min(0.5*co_s + 0.5*ro_s, 1.0)

def score_exp(yoe):
    if yoe is None: return 0.0
    if 6 <= yoe <= 8: return 1.0
    elif yoe >= 5: return 0.85
    elif yoe > 8 and yoe <= 10: return 0.80
    elif yoe >= 4: return 0.65
    elif yoe > 10 and yoe <= 14: return 0.55
    return 0.35

def score_loc(loc, notice):
    loc_s = 1.0 if any(l in (loc or "").lower() for l in TOP_LOCS) else 0.4
    if notice is None: n_s = 0.5
    elif notice == 0: n_s = 1.0
    elif notice <= 15: n_s = 0.90
    elif notice <= 30: n_s = 0.75
    elif notice <= 60: n_s = 0.50
    else: n_s = 0.20
    return 0.6*loc_s + 0.4*n_s

def score_edu(edu):
    if not edu: return 0.4
    return 1.0 if any(t in edu.lower() for t in TIER1) else 0.55

def beh_mult(otw, rr, gh):
    avail = 0.4*(1.0 if otw else 0.3) + 0.35*1.0 + 0.15*0.8 + 0.10*0.5
    engage = 0.45*min(rr, 1.0) + 0.30*0.7 + 0.25*0.8
    cred = 0.35*0.85 + 0.30*0.4 + 0.35*min(gh, 1.0)
    return min(0.3 + 0.9*(0.45*avail + 0.35*engage + 0.20*cred), 1.2)

MEDAL = ["🥇", "🥈", "🥉"]
BAR_CHARS = "█"

def make_bar(v, width=20):
    filled = int(round(v * width))
    return "█" * filled + "░" * (width - filled)

def rank_candidates(
    n1, y1, l1, np1, co1, ro1, ed1, ow1, rr1, gh1, sk1,
    n2, y2, l2, np2, co2, ro2, ed2, ow2, rr2, gh2, sk2,
    n3, y3, l3, np3, co3, ro3, ed3, ow3, rr3, gh3, sk3,
    n4, y4, l4, np4, co4, ro4, ed4, ow4, rr4, gh4, sk4,
    n5, y5, l5, np5, co5, ro5, ed5, ow5, rr5, gh5, sk5,
):
    inputs = [
        (n1,y1,l1,np1,co1,ro1,ed1,ow1,rr1,gh1,sk1),
        (n2,y2,l2,np2,co2,ro2,ed2,ow2,rr2,gh2,sk2),
        (n3,y3,l3,np3,co3,ro3,ed3,ow3,rr3,gh3,sk3),
        (n4,y4,l4,np4,co4,ro4,ed4,ow4,rr4,gh4,sk4),
        (n5,y5,l5,np5,co5,ro5,ed5,ow5,rr5,gh5,sk5),
    ]
    candidates = []
    for (name, yoe, loc, notice, company, role, edu, otw, rr, gh, sk_raw) in inputs:
        if not name.strip(): continue
        skills = parse_skills(sk_raw or "")
        s_sk = score_skills(skills)
        s_ca = score_career(company or "", role or "")
        s_ex = score_exp(float(yoe) if yoe else None)
        s_lo = score_loc(loc, float(notice) if notice else None)
        s_ed = score_edu(edu or "")
        bm = beh_mult(otw == "Yes", float(rr) if rr else 0.65, float(gh) if gh else 0.55)
        raw = 0.35*s_sk + 0.35*s_ca + 0.15*s_ex + 0.10*s_lo + 0.05*s_ed
        final = raw * bm
        is_prod = any(p in (company or "").lower() for p in PRODUCT_COS)
        is_con = any(c in (company or "").lower() for c in CONSULT)
        co_tag = "🏢 Product co." if is_prod else ("🏗️ Consulting" if is_con else "🏬 Other")
        candidates.append({
            "name": name, "role": role or "—", "company": company or "—",
            "loc": loc or "—", "yoe": yoe, "otw": otw,
            "sk": s_sk, "ca": s_ca, "ex": s_ex, "lo": s_lo, "ed": s_ed,
            "bm": bm, "raw": raw, "final": final, "co_tag": co_tag,
        })

    if not candidates:
        return "⚠️ Add at least one candidate name to rank.", ""

    candidates.sort(key=lambda x: x["final"], reverse=True)

    lines = ["# 🏆 Redrob Candidate Rankings\n"]
    lines.append(f"**{len(candidates)} candidate{'s' if len(candidates)!=1 else ''} ranked** · Scoring: Skills 35% · Career 35% · Experience 15% · Location 10% · Education 5%\n")
    lines.append("---\n")

    for i, c in enumerate(candidates):
        medal = MEDAL[i] if i < 3 else f"#{i+1}"
        lines.append(f"## {medal} {c['name']}")
        lines.append(f"**{c['role']}** at **{c['company']}** · {c['loc']} · {c['yoe'] or '?'} yrs exp")
        lines.append(f"{c['co_tag']} · {'✅ Open to work' if c['otw']=='Yes' else '⏸️ Not open'}\n")

        score_pct = c['final'] * 100
        lines.append(f"### Final score: **{score_pct:.1f} / 100**")
        lines.append(f"`{make_bar(c['final'])}` {score_pct:.1f}%\n")

        lines.append("**Dimension breakdown:**")
        dims = [
            ("Skills      ", c['sk']),
            ("Career      ", c['ca']),
            ("Experience  ", c['ex']),
            ("Location    ", c['lo']),
            ("Education   ", c['ed']),
        ]
        for dname, dval in dims:
            pct = dval * 100
            lines.append(f"- `{dname}` {make_bar(dval, 15)} {pct:.0f}%")

        lines.append(f"\n_Behavioral multiplier: **{c['bm']:.2f}×** · Raw score: {c['raw']*100:.1f}_")
        lines.append("\n---\n")

    csv_lines = ["rank,name,role,company,location,yoe,final_score,skills,career,experience,location_score,education,behavioral_mult,open_to_work"]
    for i, c in enumerate(candidates):
        csv_lines.append(f"{i+1},{c['name']},{c['role']},{c['company']},{c['loc']},{c['yoe'] or ''},"
                         f"{c['final']:.4f},{c['sk']:.3f},{c['ca']:.3f},{c['ex']:.3f},{c['lo']:.3f},"
                         f"{c['ed']:.3f},{c['bm']:.3f},{c['otw']}")

    return "\n".join(lines), "\n".join(csv_lines)

CSS = """
.gradio-container { max-width: 1100px !important; }
.cand-block { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 8px; background: #fafafa; }
footer { display: none !important; }
"""

SKILL_PLACEHOLDER = "Python, 5, 72\nMachine Learning, 5, 60\nRetrieval, 4, 36\nEmbeddings, 4, 30\nPyTorch, 5, 48"

def make_cand_block(label, defaults):
    with gr.Group():
        gr.Markdown(f"### {label}")
        with gr.Row():
            name = gr.Textbox(label="Full name", placeholder="Priya Sharma", value=defaults.get("name",""), scale=2)
            yoe  = gr.Number(label="Years of experience", value=defaults.get("yoe",None), minimum=0, maximum=40, scale=1)
        with gr.Row():
            loc    = gr.Textbox(label="Current location", placeholder="Pune, Maharashtra", value=defaults.get("loc",""), scale=2)
            notice = gr.Number(label="Notice period (days)", value=defaults.get("notice",None), minimum=0, scale=1)
        with gr.Row():
            company = gr.Textbox(label="Current company", placeholder="Swiggy", value=defaults.get("company",""), scale=2)
            role    = gr.Textbox(label="Role title", placeholder="Senior ML Engineer", value=defaults.get("role",""), scale=2)
        with gr.Row():
            edu  = gr.Textbox(label="Institution", placeholder="IIT Bombay", value=defaults.get("edu",""), scale=2)
            otw  = gr.Radio(label="Open to work", choices=["Yes","No"], value=defaults.get("otw","Yes"), scale=1)
        with gr.Row():
            rr = gr.Slider(label="Recruiter response rate", minimum=0, maximum=1, step=0.05, value=defaults.get("rr",0.70), scale=1)
            gh = gr.Slider(label="GitHub activity score",   minimum=0, maximum=1, step=0.05, value=defaults.get("gh",0.60), scale=1)
        skills = gr.Textbox(
            label="Skills  (one per line: Name, Proficiency 1-5, Months used)",
            placeholder=SKILL_PLACEHOLDER,
            value=defaults.get("skills",""),
            lines=5,
        )
    return name, yoe, loc, notice, company, role, edu, otw, rr, gh, skills

DEFAULTS = [
    {"name":"Priya Sharma","yoe":7,"loc":"Pune, Maharashtra","notice":15,"company":"Swiggy","role":"Senior ML Engineer","edu":"IIT Bombay","otw":"Yes","rr":0.92,"gh":0.85,"skills":"Python, 5, 72\nMachine Learning, 5, 60\nRetrieval, 4, 36\nEmbeddings, 4, 30\nPyTorch, 5, 48"},
    {"name":"Rahul Verma","yoe":3,"loc":"Jaipur, Rajasthan","notice":90,"company":"TCS","role":"Data Analyst","edu":"State University","otw":"No","rr":0.30,"gh":0.20,"skills":"Python, 3, 24\nSQL, 3, 18"},
    {"name":"Anita Rao","yoe":8,"loc":"Bengaluru, Karnataka","notice":30,"company":"Razorpay","role":"ML Engineer","edu":"NIT Trichy","otw":"Yes","rr":0.75,"gh":0.80,"skills":"PyTorch, 5, 60\nNLP, 4, 48\nEmbeddings, 4, 36\nLLM, 4, 30\nTransformers, 4, 42"},
    {},{},
]

with gr.Blocks(css=CSS, title="Redrob Candidate Ranker") as demo:
    gr.Markdown("""
# 🎯 Redrob Intelligent Candidate Ranker
> **Hackathon: Intelligent Candidate Discovery & Ranking Challenge**  
> Ranks candidates the way a great recruiter would — not by keywords, but by understanding who genuinely fits the role.

**Scoring formula:** `Final = (0.35×Skills + 0.35×Career + 0.15×Experience + 0.10×Location + 0.05×Education) × Behavioral multiplier`
    """)

    all_inputs = []
    with gr.Tabs():
        for i, d in enumerate(DEFAULTS):
            with gr.TabItem(f"Candidate {i+1}"):
                fields = make_cand_block(f"Candidate {i+1}", d)
                all_inputs.extend(fields)

    rank_btn = gr.Button("🚀 Run Ranking", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=3):
            output_md = gr.Markdown(label="Results")
        with gr.Column(scale=1):
            output_csv = gr.Textbox(label="CSV output (copy to submit)", lines=20, show_copy_button=True)

    gr.Markdown("""
---
### How scoring works
| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Skills | 35% | Proficiency × duration × endorsements × JD match |
| Career | 35% | Product co. vs consulting, ML role relevance |
| Experience | 15% | Sweet spot 6–8 years (JD requirement) |
| Location | 10% | Pune/NCR/Bengaluru preferred + short notice |
| Education | 5% | Tier-1 institution + relevant field |
| Behavioral × | Multiplier 0.3–1.2× | Open to work, response rate, GitHub activity |

**Skill format:** one skill per line — `Skill Name, Proficiency (1-5), Months used`  
**Example:** `Machine Learning, 5, 60`
    """)

    rank_btn.click(fn=rank_candidates, inputs=all_inputs, outputs=[output_md, output_csv])

if __name__ == "__main__":
    demo.launch()
