import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalAI – Contract Risk Classification System",
    page_icon="⚖️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
}

/* Card */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
}

/* Result badge */
.result-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}

/* Confidence bar container */
.conf-label {
    color: #cbd5e1;
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
    font-weight: 500;
}

/* Top-N item */
.clause-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid;
}
.clause-name {
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 500;
}
.clause-pct {
    color: #a5f3fc;
    font-size: 0.9rem;
    font-weight: 600;
}

/* Example pill */
.example-pill {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.35);
    color: #c4b5fd;
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    cursor: pointer;
    margin: 0.2rem;
}

/* Stat chips */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    color: #94a3b8;
    font-size: 0.82rem;
    margin: 0.2rem;
}

/* Streamlit textarea override */
textarea {
    background: rgba(15,12,41,0.7) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}
textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.5rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.88;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load LegalAI Models ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = joblib.load("best_model.pkl")
    le        = joblib.load("label_encoder.pkl")
    tfidf_word = joblib.load("tfidf_word.pkl")
    return model, le, tfidf_word

model, le, tfidf_word = load_models()

# Clause type → color accent
COLORS = [
    "#a78bfa","#60a5fa","#34d399","#f472b6",
    "#fbbf24","#f87171","#38bdf8","#a3e635",
]

# ── Example clauses ───────────────────────────────────────────────────────────
EXAMPLES = {
    "Governing Law":       "This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to conflict of law principles.",
    "Termination":         "Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice to the other party.",
    "Liability Cap":       "In no event shall either party's total liability exceed the total fees paid in the twelve (12) months preceding the claim.",
    "Non-Compete":         "During the term and for two (2) years thereafter, Employee shall not engage in any business that directly competes with the Company within the United States.",
    "Renewal Term":        "This Agreement shall automatically renew for successive one-year terms unless either party provides sixty (60) days written notice of non-renewal.",
    "Confidentiality":     "Each party agrees to keep confidential all proprietary information received from the other party and not to disclose it to any third party.",
    "IP Ownership":        "All intellectual property developed by Employee in connection with their duties shall be the exclusive property of the Company.",
    "Audit Rights":        "Client shall have the right to audit Vendor's records relating to this Agreement upon thirty (30) days written notice, no more than once per calendar year.",
}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
 <h1>⚖️ LegalAI – Contract Risk Classification System</h1>
<p>Naive Bayes-powered contract risk analysis and clause classification platform</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; margin-bottom:1.5rem;'>
  <span class='chip'>🧠 Naive Bayes (ComplementNB)</span>
  <span class='chip'>📋 {len(le.classes_)} Clause Types</span>
  <span class='chip'>🔤 TF-IDF Features</span>
  <span class='chip'>📚 CUAD Dataset</span>
</div>
""", unsafe_allow_html=True)

# ── Examples ──────────────────────────────────────────────────────────────────
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:0.6rem;'>⚡ Try an example</p>", unsafe_allow_html=True)

cols = st.columns(4)
example_keys = list(EXAMPLES.keys())
selected_example = None
for i, key in enumerate(example_keys):
    with cols[i % 4]:
        if st.button(key, key=f"ex_{i}"):
            selected_example = EXAMPLES[key]

st.markdown("</div>", unsafe_allow_html=True)

# ── Text input ────────────────────────────────────────────────────────────────
if "clause_text" not in st.session_state:
    st.session_state.clause_text = ""

if selected_example:
    st.session_state.clause_text = selected_example

clause_text = st.text_area(  
    "Paste Contract Clause for Risk Analysis",
    value=st.session_state.clause_text,
    height=160,
    placeholder="e.g. This Agreement shall be governed by the laws of California...",
    label_visibility="visible",
)

col_btn, col_clear = st.columns([4, 1])
with col_btn:
    analyze = st.button("🔍 Analyze Contract Risk", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.clause_text = ""
        st.rerun()

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze and clause_text.strip():
    with st.spinner("Analyzing…"):
        X = tfidf_word.transform([clause_text])[:, :8000]
        proba = model.predict_proba(X)[0]
        top5_idx = np.argsort(proba)[::-1][:5]

        top_label = le.classes_[top5_idx[0]]
        top_conf  = proba[top5_idx[0]] * 100

    st.markdown("---")
    st.markdown("<h3 style='color:#e2e8f0; margin-bottom:0.8rem;'>📊 LegalAI Risk Analysis</h3>", unsafe_allow_html=True)

    # Primary result
    st.markdown(f"""
    <div class='glass-card' style='border-color:rgba(124,58,237,0.4);'>
      <p style='color:#94a3b8; font-size:0.82rem; margin-bottom:0.5rem;'>DETECTED CONTRACT CLAUSE TYPE</p>
      <div class='result-badge'>{top_label}</div>
      <div class='conf-label'>Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar for top prediction
    st.progress(int(top_conf))
    st.markdown(f"<p style='color:#a5f3fc; font-size:0.9rem; text-align:right; margin-top:-0.5rem;'>{top_conf:.1f}%</p>", unsafe_allow_html=True)

    # Top-5 breakdown
    st.markdown("<div class='glass-card' style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:0.8rem;'>🏆 Top 5 Predictions</p>", unsafe_allow_html=True)

    for rank, idx in enumerate(top5_idx):
        label = le.classes_[idx]
        pct   = proba[idx] * 100
        color = COLORS[rank % len(COLORS)]
        bar_w = int(pct)
        st.markdown(f"""
        <div class='clause-row' style='border-color:{color};'>
          <div>
            <span style='color:#64748b; font-size:0.75rem; margin-right:0.5rem;'>#{rank+1}</span>
            <span class='clause-name'>{label}</span>
          </div>
          <span class='clause-pct'>{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif analyze and not clause_text.strip():
    st.warning("Please enter a clause to classify.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(255,255,255,0.08); margin-top:2rem;'>
<p style='text-align:center; color:#475569; font-size:0.78rem;'>
  Built with Scikit-learn · TF-IDF + Naive Bayes (ComplementNB) · CUAD Legal Dataset
</p>
""", unsafe_allow_html=True)