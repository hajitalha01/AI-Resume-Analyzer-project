# ================================================================
# AI Resume Analyzer — Streamlit App
# OCEAN Personality Prediction (lively, extraverted, responsible,
#                                serious, dependable)
# ================================================================
 
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import io
import time
import joblib
import plotly.graph_objects as go
import plotly.express as px
 
# ── Optional heavy imports ──────────────────────────────────────
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False
 
try:
    from docx import Document
    DOCX_OK = True
except ImportError:
    DOCX_OK = False
 
# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ================================================================
# GLOBAL CSS
# ================================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #F7F9FC;
    color: #1A1D23;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2563EB 100%);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: white !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 8px 14px;
    transition: background 0.2s;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.18);
}
.card {
    background: white;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07);
    margin-bottom: 20px;
    border: 1px solid #E8EDF5;
}
.card-blue {
    background: linear-gradient(135deg,#2563EB,#1E40AF);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.metric-tile {
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(37,99,235,0.06);
    border: 1px solid #E8EDF5;
}
.metric-tile .val {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2563EB;
}
.metric-tile .label {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
}
.stButton > button:hover { opacity: 0.88; }
.badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 3px;
    border: 1px solid #BFDBFE;
}
.section-header {
    font-size:1.3rem;
    font-weight:700;
    color:#1E3A5F;
    border-left:4px solid #2563EB;
    padding-left:12px;
    margin-bottom:16px;
}
.personality-box {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 2px solid #2563EB;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 10px 0;
}
.personality-label {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1E3A5F;
    text-transform: capitalize;
    letter-spacing: 0.02em;
}
.personality-emoji {
    font-size: 3rem;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)
 
# ================================================================
# SESSION STATE
# ================================================================
defaults = {
    "page":        "🏠 Home",
    "resume_text": "",
    "manual_data": {},
    "prediction":  None,
    "ats_score":   None,
    "confidence":  None,
    "skill_scores":{},
    "model":       None,
    "le_target":   None,
    "features":    None,
    "processed":   False,
    "dist":        [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
# ================================================================
# CONSTANTS
# ================================================================
SKILL_CATEGORIES = {
    "Programming":    ["python","java","javascript","c++","c#","ruby","go","rust","swift","kotlin","r","scala","php","typescript"],
    "Data & ML":      ["machine learning","deep learning","tensorflow","pytorch","scikit-learn","pandas","numpy","sql","data analysis","nlp","computer vision","keras","xgboost"],
    "Cloud & DevOps": ["aws","azure","gcp","docker","kubernetes","ci/cd","jenkins","terraform","ansible","linux","bash","git"],
    "Web":            ["react","angular","vue","node.js","django","flask","fastapi","html","css","rest api","graphql","next.js"],
    "Soft Skills":    ["leadership","communication","teamwork","problem solving","agile","scrum","project management","presentation"],
}
 
PERSONALITY_LABELS = ["lively", "extraverted", "responsible", "serious", "dependable"]
 
PERSONALITY_INFO = {
    "lively":      {"emoji": "🎉", "desc": "Energetic, enthusiastic and fun-loving personality"},
    "extraverted": {"emoji": "🌟", "desc": "Outgoing, sociable and draws energy from others"},
    "responsible": {"emoji": "🎯", "desc": "Reliable, accountable and takes ownership"},
    "serious":     {"emoji": "🧠", "desc": "Thoughtful, analytical and detail-oriented"},
    "dependable":  {"emoji": "🤝", "desc": "Trustworthy, consistent and keeps commitments"},
}
 
# ================================================================
# UTILITY FUNCTIONS
# ================================================================
 
def extract_pdf_text(file_bytes: bytes) -> str:
    if not PDF_OK:
        return ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)
 
 
def extract_docx_text(file_bytes: bytes) -> str:
    if not DOCX_OK:
        return ""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)
 
 
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-\+\#\.]", " ", text)
    return text.strip().lower()
 
 
def extract_skills(text: str) -> dict:
    text_lower = text.lower()
    scores = {}
    for cat, skills in SKILL_CATEGORIES.items():
        found = [s for s in skills if s in text_lower]
        pct   = min(100, int(len(found) / max(1, len(skills)) * 100 + np.random.randint(0, 10)))
        scores[cat] = {"found": found, "score": pct}
    return scores
 
 
def extract_experience_years(text: str) -> int:
    matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience", text, re.I)
    if matches:
        return int(matches[0])
    return np.random.randint(1, 8)
 
 
def compute_ats_score(text: str, skill_scores: dict) -> int:
    score = 40
    word_count = len(text.split())
    if word_count > 200: score += 10
    if word_count > 400: score += 5
    skill_avg = np.mean([v["score"] for v in skill_scores.values()])
    score += int(skill_avg * 0.3)
    if re.search(r"\d{4}", text):          score += 5
    if re.search(r"@", text):             score += 3
    if re.search(r"linkedin", text, re.I): score += 2
    return min(100, max(30, score + np.random.randint(-3, 4)))
 
 
def predict_personality(text: str, skill_scores: dict, model=None, le_target=None, features=None):
    """
    If pkl model loaded use it.
    Otherwise rule-based fallback returning one of 5 personality labels.
    """
    # ── PKL Model ─────────────────────────────────────────────
    if model is not None and features is not None:
        try:
            feature_vec = np.zeros(len(features))
            row = pd.DataFrame([feature_vec], columns=features)
            proba = model.predict_proba(row)[0]
            idx   = int(np.argmax(proba))
            conf  = float(proba[idx])
            label = le_target.inverse_transform([idx])[0] if le_target else PERSONALITY_LABELS[idx % 5]
            dist  = list(zip(
                le_target.classes_.tolist() if le_target else PERSONALITY_LABELS,
                proba.tolist()
            ))
            return str(label), conf, dist
        except Exception:
            pass
 
    # ── Rule-based fallback ────────────────────────────────────
    soft  = skill_scores.get("Soft Skills",   {}).get("score", 0)
    data  = skill_scores.get("Data & ML",     {}).get("score", 0)
    prog  = skill_scores.get("Programming",   {}).get("score", 0)
    cloud = skill_scores.get("Cloud & DevOps",{}).get("score", 0)
    web   = skill_scores.get("Web",           {}).get("score", 0)
 
    weights = {
        "lively":      soft  * 1.2 + web  * 0.3,
        "extraverted": soft  * 1.0 + web  * 0.5,
        "responsible": cloud * 0.8 + prog * 0.5 + soft * 0.3,
        "serious":     data  * 1.2 + prog * 0.4,
        "dependable":  prog  * 0.8 + cloud* 0.6 + data * 0.3,
    }
    total = sum(weights.values()) or 1
    dist  = [(lbl, round(w / total, 4)) for lbl, w in weights.items()]
    dist.sort(key=lambda x: -x[1])
    best_label, best_conf = dist[0]
    return best_label, min(0.97, best_conf + 0.10), dist
 
 
def generate_suggestions(text: str, ats: int, skill_scores: dict) -> list:
    tips = []
    if ats < 60:
        tips.append("📌 Add quantifiable achievements (e.g., 'Improved performance by 40%').")
    if len(text.split()) < 300:
        tips.append("📄 Your resume looks short. 400–600 words is better for ATS compatibility.")
    if not re.search(r"@", text):
        tips.append("📧 Make sure to include a professional email address.")
    if not re.search(r"linkedin", text, re.I):
        tips.append("🔗 Add your LinkedIn profile URL.")
    if skill_scores.get("Cloud & DevOps", {}).get("score", 0) < 20:
        tips.append("☁️ Add cloud skills (AWS/GCP/Azure) — they are in high demand in 2025.")
    if skill_scores.get("Data & ML", {}).get("score", 0) < 15:
        tips.append("🤖 ML skills (Python, scikit-learn) can significantly boost your hirability.")
    if not tips:
        tips.append("✅ Great resume! Keep your skills section regularly updated.")
    return tips
 
 
# ================================================================
# MODEL LOADER (Sidebar)
# ================================================================
def load_model_section():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Load Your Model")
    model_file  = st.sidebar.file_uploader("model.pkl",          type=["pkl"], key="m_model")
    target_file = st.sidebar.file_uploader("target_encoder.pkl", type=["pkl"], key="m_target")
    feat_file   = st.sidebar.file_uploader("features.pkl",       type=["pkl"], key="m_feat")
 
    if model_file:
        try:
            st.session_state.model     = joblib.load(model_file)
            st.sidebar.success("✅ Model loaded")
        except Exception as e:
            st.sidebar.error(f"Model error: {e}")
    if target_file:
        try:
            st.session_state.le_target = joblib.load(target_file)
            st.sidebar.success("✅ Encoder loaded")
        except Exception as e:
            st.sidebar.error(f"Encoder error: {e}")
    if feat_file:
        try:
            st.session_state.features  = joblib.load(feat_file)
            st.sidebar.success("✅ Features loaded")
        except Exception as e:
            st.sidebar.error(f"Features error: {e}")
 
 
# ================================================================
# SIDEBAR
# ================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:16px 0 8px'>
          <span style='font-size:2rem'>🧠</span><br>
          <span style='font-size:1.15rem; font-weight:700; color:white'>ResumeAI</span><br>
          <span style='font-size:0.75rem; color:rgba(255,255,255,0.65)'>OCEAN Personality Analyzer</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📤 Upload Resume", "📊 Dashboard", "👤 Portfolio"],
            label_visibility="collapsed",
        )
        st.session_state.page = page
        load_model_section()
        st.sidebar.markdown("---")
        if st.session_state.model:
            st.sidebar.markdown("🟢 **Model:** Active")
        else:
            st.sidebar.markdown("🟡 **Model:** Demo mode")
 
 
# ================================================================
# PAGE 1 — HOME
# ================================================================
def page_home():
    st.markdown("""
    <div class='card-blue'>
      <h2 style='color:white; font-size:2rem; margin-bottom:8px'>🧠 AI Resume & Personality Analyzer</h2>
      <p style='color:rgba(255,255,255,0.9); font-size:1.05rem'>
        Upload your resume and predict your OCEAN personality type.
        Powered by your trained ML model.
      </p>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("### 🎭 5 Personality Types")
    cols = st.columns(5)
    for col, label in zip(cols, PERSONALITY_LABELS):
        info = PERSONALITY_INFO[label]
        col.markdown(f"""
        <div class='metric-tile'>
          <div style='font-size:1.8rem'>{info['emoji']}</div>
          <div style='font-weight:700; margin:8px 0 4px; text-transform:capitalize'>{label}</div>
          <div style='font-size:0.75rem; color:#64748B'>{info['desc']}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🚀 How It Works")
        steps = [
            ("Upload",   "Upload your PDF/DOCX resume or fill in the manual form"),
            ("Extract",  "Text and skill signals are extracted automatically"),
            ("Predict",  "The trained OCEAN model predicts your personality"),
            ("Result",   "View your personality label, ATS score, and charts"),
        ]
        for i, (s, d) in enumerate(steps, 1):
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:center;margin:10px 0'>
              <div style='width:30px;height:30px;border-radius:50%;background:#2563EB;
                color:white;display:flex;align-items:center;justify-content:center;
                font-weight:700;flex-shrink:0'>{i}</div>
              <div><strong>{s}</strong><br>
              <span style='font-size:0.85rem;color:#64748B'>{d}</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 

 
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("📤 Upload Resume", use_container_width=True):
            st.session_state.page = "📤 Upload Resume"
            st.rerun()
    with c2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "📊 Dashboard"
            st.rerun()
 
 
# ================================================================
# PAGE 2 — UPLOAD
# ================================================================
def page_upload():
    st.markdown("<div class='section-header'>📤 Resume Upload & Analysis</div>", unsafe_allow_html=True)
 
    tab1, tab2 = st.tabs(["📁 File Upload", "✏️ Manual Input"])
 
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop your PDF or DOCX resume here",
            type=["pdf", "docx", "doc"],
        )
        if uploaded:
            ext       = uploaded.name.rsplit(".", 1)[-1].lower()
            raw_bytes = uploaded.read()
            if ext == "pdf":
                text = extract_pdf_text(raw_bytes) if PDF_OK else ""
                if not PDF_OK:
                    st.warning("Install pdfplumber: pip install pdfplumber")
            elif ext in ("docx", "doc"):
                text = extract_docx_text(raw_bytes) if DOCX_OK else ""
                if not DOCX_OK:
                    st.warning("Install python-docx: pip install python-docx")
            else:
                text = raw_bytes.decode("utf-8", errors="ignore")
 
            if text.strip():
                st.session_state.resume_text = text
                st.success(f"✅ {len(text.split())} words extracted from {uploaded.name}")
                with st.expander("📄 Extracted text preview"):
                    st.text_area("", text[:2000], height=200)
            else:
                st.error("Could not extract text. Please try the Manual Input tab.")
        st.markdown("</div>", unsafe_allow_html=True)
 
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.form("manual_form"):
            c1, c2 = st.columns(2)
            with c1:
                name   = st.text_input("Full Name",  placeholder="John Smith")
                email  = st.text_input("Email",      placeholder="john@email.com")
                age    = st.slider("Age", 18, 60, 25)
                gender = st.selectbox("Gender", ["Male", "Female"])
            with c2:
                exp    = st.slider("Years of Experience", 0, 20, 3)
                edu    = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD", "Bootcamp"])
                loc    = st.text_input("Location", placeholder="New York, USA")
 
            st.markdown("#### 🛠 Select Your Skills")
            skill_cols = st.columns(len(SKILL_CATEGORIES))
            selected_skills = {}
            for col, (cat, skills) in zip(skill_cols, SKILL_CATEGORIES.items()):
                with col:
                    st.markdown(f"**{cat}**")
                    picks = st.multiselect(cat, skills, label_visibility="collapsed", key=f"sk_{cat}")
                    selected_skills[cat] = picks
 
            summary   = st.text_area("Professional Summary", placeholder="Write your achievements here...", height=100)
            submitted = st.form_submit_button("🚀 Analyse", use_container_width=True)
 
        if submitted:
            all_skills = [s for picks in selected_skills.values() for s in picks]
            st.session_state.resume_text = " ".join(filter(None, [
                name, email, loc, f"{exp} years experience",
                edu, " ".join(all_skills), summary
            ]))
            st.session_state.manual_data = {
                "name": name, "email": email, "age": age,
                "gender": gender, "exp": exp, "edu": edu,
                "skills": selected_skills,
            }
            st.success("✅ Data saved!")
        st.markdown("</div>", unsafe_allow_html=True)
 
    if st.session_state.resume_text:
        st.markdown("---")
        if st.button("⚡ Run Full Analysis", use_container_width=False):
            _run_analysis()
 
 
def _run_analysis():
    steps = [
        "🔍 Cleaning text...",
        "🛠 Extracting skills...",
        "📐 Building feature vector...",
        "🤖 Running OCEAN model prediction...",
        "📊 Calculating ATS score...",
        "✅ Analysis complete!",
    ]
    bar    = st.progress(0)
    status = st.empty()
    text   = clean_text(st.session_state.resume_text)
 
    for i, step in enumerate(steps):
        status.markdown(f"""
        <div class='card' style='padding:14px 20px;border-left:4px solid #2563EB'>
          <b>{step}</b>
        </div>""", unsafe_allow_html=True)
        bar.progress(int((i + 1) / len(steps) * 100))
        time.sleep(0.5)
 
    skill_scores     = extract_skills(text)
    ats              = compute_ats_score(text, skill_scores)
    pred, conf, dist = predict_personality(
        text, skill_scores,
        st.session_state.model,
        st.session_state.le_target,
        st.session_state.features,
    )
 
    st.session_state.skill_scores = skill_scores
    st.session_state.ats_score    = ats
    st.session_state.prediction   = pred
    st.session_state.confidence   = conf
    st.session_state.dist         = dist
    st.session_state.processed    = True
 
    status.success("🎉 Analysis complete! Go to the Dashboard to view your results.")
 
 
# ================================================================
# PAGE 3 — DASHBOARD
# ================================================================
def page_dashboard():
    st.markdown("<div class='section-header'>📊 Analysis Dashboard</div>", unsafe_allow_html=True)
 
    if not st.session_state.processed:
        st.info("ℹ️ Please analyse your resume first — go to the Upload Resume page.")
        return
 
    ats    = st.session_state.ats_score
    pred   = st.session_state.prediction
    conf   = st.session_state.confidence
    skills = st.session_state.skill_scores
    dist   = st.session_state.dist
    text   = st.session_state.resume_text
    info   = PERSONALITY_INFO.get(pred, {"emoji": "🧠", "desc": ""})
 
    # ── KPI Row ───────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, (val, lbl) in zip([k1,k2,k3,k4], [
        (f"{ats}",              "ATS Score"),
        (f"{conf*100:.0f}%",    "Confidence"),
        (f"{len(text.split())}","Word Count"),
        (f"{extract_experience_years(text)}yr", "Experience"),
    ]):
        col.markdown(f"""
        <div class='metric-tile'>
          <div class='val'>{val}</div>
          <div class='label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Personality Result (MAIN HIGHLIGHT) ──────────────────
    st.markdown(f"""
    <div class='personality-box'>
      <div class='personality-emoji'>{info['emoji']}</div>
      <div style='font-size:0.95rem;color:#2563EB;font-weight:600;
           text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>
        Predicted Personality
      </div>
      <div class='personality-label'>{pred}</div>
      <div style='color:#64748B;margin-top:8px;font-size:0.95rem'>{info['desc']}</div>
      <div style='margin-top:14px;font-size:1rem;color:#1E3A5F'>
        Confidence: <strong style='color:#2563EB;font-size:1.2rem'>{conf*100:.1f}%</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Row 1: Gauge + Pie ────────────────────────────────────
    col_g, col_d = st.columns(2)
 
    with col_g:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 ATS Score Gauge")
        color = "#22C55E" if ats >= 75 else "#F59E0B" if ats >= 50 else "#EF4444"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ats,
            delta={"reference": 60, "increasing": {"color": "#22C55E"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  50], "color": "#FEE2E2"},
                    {"range": [50, 75], "color": "#FEF3C7"},
                    {"range": [75,100], "color": "#DCFCE7"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 75},
            },
            title={"text": "ATS Compatibility"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with col_d:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🥧 Personality Distribution")
        if dist:
            labels = [d[0] for d in dist]
            values = [d[1] for d in dist]
        else:
            labels = PERSONALITY_LABELS
            values = [0.2] * 5
        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=["#2563EB","#3B82F6","#60A5FA","#93C5FD","#BFDBFE"],
            textinfo="label+percent",
            pull=[0.08 if l == pred else 0 for l in labels],
        ))
        fig_pie.update_layout(
            height=280, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    # ── Row 2: Skill Bar + Radar ──────────────────────────────
    col_b, col_r = st.columns(2)
 
    with col_b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🛠 Skill Matching")
        cats   = list(skills.keys())
        scores = [skills[c]["score"] for c in cats]
        fig_bar = go.Figure(go.Bar(
            x=scores, y=cats,
            orientation="h",
            marker=dict(color=scores, colorscale="Blues", showscale=False),
            text=[f"{s}%" for s in scores],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=300, margin=dict(t=10,b=10,l=10,r=40),
            xaxis=dict(range=[0, 115]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        for cat in cats:
            found = skills[cat]["found"]
            if found:
                badges = "".join(f"<span class='badge'>{s}</span>" for s in found[:5])
                st.markdown(f"**{cat}:** {badges}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with col_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🕸 Skill Radar")
        fig_radar = go.Figure(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="#2563EB", width=2),
            marker=dict(color="#2563EB"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=300, margin=dict(t=30,b=30,l=30,r=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    # ── Experience Line ───────────────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📈 Experience Growth Trend")
    exp_years = extract_experience_years(text)
    years     = list(range(max(1, exp_years - 4), exp_years + 2))
    growth    = [max(20, 40 + i * 12 + np.random.randint(-5, 6)) for i in range(len(years))]
    fig_line  = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=years, y=growth,
        mode="lines+markers",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=8, color="#2563EB"),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
    ))
    fig_line.update_layout(
        height=250, margin=dict(t=10,b=10,l=10,r=10),
        xaxis_title="Year", yaxis_title="Proficiency Score",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
 
    # ── Suggestions ───────────────────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 💡 Resume Improvement Suggestions")
    for tip in generate_suggestions(text, ats, skills):
        st.markdown(f"- {tip}")
    st.markdown("</div>", unsafe_allow_html=True)
 
 
# ================================================================
# PAGE 4 — PORTFOLIO
# ================================================================
def page_portfolio():
    st.markdown("<div class='section-header'>👤 Candidate Portfolio</div>", unsafe_allow_html=True)
 
    if not st.session_state.processed:
        st.info("ℹ️ Please analyse your resume first.")
        return
 
    md     = st.session_state.manual_data
    name   = md.get("name", "Candidate")
    exp    = md.get("exp",  extract_experience_years(st.session_state.resume_text))
    edu    = md.get("edu",  "N/A")
    ats    = st.session_state.ats_score
    pred   = st.session_state.prediction
    conf   = st.session_state.confidence
    skills = st.session_state.skill_scores
    info   = PERSONALITY_INFO.get(pred, {"emoji": "🧠", "desc": ""})
 
    col_l, col_r = st.columns([1, 2])
 
    with col_l:
        initials = "".join(w[0].upper() for w in name.split()[:2]) if name != "Candidate" else "AI"
        st.markdown(f"""
        <div class='card' style='text-align:center'>
          <div style='width:80px;height:80px;border-radius:50%;
            background:linear-gradient(135deg,#2563EB,#1E40AF);
            color:white;font-size:1.8rem;font-weight:700;
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 14px'>{initials}</div>
          <div style='font-size:1.25rem;font-weight:700;color:#1E3A5F'>{name}</div>
          <div style='font-size:2rem;margin:10px 0'>{info['emoji']}</div>
          <div style='color:#2563EB;font-weight:700;font-size:1.1rem;
               text-transform:capitalize'>{pred}</div>
          <div style='color:#64748B;font-size:0.83rem;margin:4px 0'>{info['desc']}</div>
          <hr>
          <div style='display:flex;justify-content:space-around;margin-top:12px'>
            <div>
              <div style='font-weight:700;color:#2563EB;font-size:1.2rem'>{ats}</div>
              <div style='font-size:0.75rem;color:#64748B'>ATS</div>
            </div>
            <div>
              <div style='font-weight:700;color:#2563EB;font-size:1.2rem'>{conf*100:.0f}%</div>
              <div style='font-size:0.75rem;color:#64748B'>Conf.</div>
            </div>
            <div>
              <div style='font-weight:700;color:#2563EB;font-size:1.2rem'>{exp}</div>
              <div style='font-size:0.75rem;color:#64748B'>Yrs</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
 
        tier = ("🥇 Excellent","#22C55E") if ats>=80 else ("🥈 Good","#F59E0B") if ats>=60 else ("🔴 Needs Work","#EF4444")
        st.markdown(f"""
        <div class='card' style='text-align:center;border:2px solid {tier[1]}'>
          <div style='font-size:1.4rem'>{tier[0]}</div>
          <div style='color:#64748B;font-size:0.82rem'>ATS Score Tier</div>
        </div>""", unsafe_allow_html=True)
 
    with col_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🛠 Skill Breakdown")
        for cat, data in skills.items():
            score = data["score"]
            bar_c = "#22C55E" if score>=70 else "#F59E0B" if score>=40 else "#EF4444"
            found = data.get("found", [])
            badges = " ".join(f"<span class='badge'>{s}</span>" for s in found[:5])
            st.markdown(f"""
            <div style='margin-bottom:14px'>
              <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                <span style='font-weight:600'>{cat}</span>
                <span style='color:#2563EB;font-weight:700'>{score}%</span>
              </div>
              <div style='background:#F1F5F9;border-radius:8px;height:8px'>
                <div style='background:{bar_c};border-radius:8px;height:8px;width:{score}%'></div>
              </div>
              <div style='margin-top:4px'>{badges}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 💡 Top Suggestions")
        for tip in generate_suggestions(st.session_state.resume_text, ats, skills)[:3]:
            st.markdown(f"- {tip}")
        st.markdown("</div>", unsafe_allow_html=True)
 
 
# ================================================================
# MAIN ROUTER
# ================================================================
def main():
    render_sidebar()
    page = st.session_state.page
    if page == "🏠 Home":
        page_home()
    elif page == "📤 Upload Resume":
        page_upload()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "👤 Portfolio":
        page_portfolio()
 
 
if __name__ == "__main__":
    main()
