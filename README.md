# 🧠 AI Resume Analyzer — Streamlit App

A full-stack, production-style Streamlit dashboard that analyses resumes using
your trained scikit-learn model from `good_ml_project.ipynb`.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
resume_analyzer/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file

# Place your model files alongside app.py (or upload via sidebar):
├── model.pkl         ← Trained sklearn model (from notebook)
├── gender_encoder.pkl
├── target_encoder.pkl
└── features.pkl
```

---

## 🔬 Loading Your Trained Model

Your notebook (`good_ml_project.ipynb`) saves these files with `joblib.dump()`:

| File | Description |
|------|-------------|
| `model.pkl` | Trained classifier (LogisticRegression / RandomForest / etc.) |
| `target_encoder.pkl` | LabelEncoder for personality/role labels |
| `gender_encoder.pkl` | LabelEncoder for gender column |
| `features.pkl` | Feature column order list |

**Two ways to load them in the app:**

1. **Sidebar uploader** — drag & drop each `.pkl` at runtime (easiest)
2. **Auto-load** — place all four files in the same folder as `app.py` and add:
   ```python
   # At the top of app.py, after imports:
   if os.path.exists("model.pkl"):
       st.session_state.model    = joblib.load("model.pkl")
       st.session_state.le_target = joblib.load("target_encoder.pkl")
       st.session_state.features  = joblib.load("features.pkl")
   ```

---

## 📊 Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Introduction, feature highlights, quick-start buttons |
| 📤 Upload Resume | PDF/DOCX upload **or** manual form input |
| 📊 Dashboard | ATS gauge, skill bar/radar/pie charts, experience line chart, suggestions |
| 👤 Portfolio | Profile card, skill progress bars, role recommendations |

---

## 🧠 Model Integration Notes

The notebook trains on a **personality dataset** (Openness, Neuroticism,
Conscientiousness, Agreeableness, Extraversion, Age, Gender).

The app bridges resume content → personality signals → job role prediction:
- If the `.pkl` model is loaded: passes feature vector to the model and maps
  the output personality class to a closest job role.
- If no model is loaded: falls back to **rule-based prediction** derived from
  detected skill scores.

You can customise the `personality_to_role` mapping in `predict_role()` to
match your exact label set.

---

## ⚙️ Customisation

| What | Where in app.py |
|------|-----------------|
| Skill taxonomy | `SKILL_CATEGORIES` dict |
| Job role list | `JOB_ROLES` list |
| ATS scoring weights | `compute_ats_score()` function |
| Suggestion rules | `generate_suggestions()` function |
| Personality → Role mapping | `predict_role()` function |
| Colour theme | CSS block at top of file |

---

## 📦 Dependencies

- **streamlit** — UI framework
- **plotly** — all charts (gauge, bar, radar, pie, line)
- **scikit-learn / joblib** — model loading
- **pdfplumber** — PDF text extraction
- **python-docx** — DOCX text extraction
- **pandas / numpy** — data handling
