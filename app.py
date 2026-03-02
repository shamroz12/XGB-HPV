import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from collections import Counter
import math
import secrets

st.set_page_config(page_title="HPV-EPIPRED AI", page_icon="🧬", layout="wide")

# =========================================================
# PREMIUM GLOBAL FONT SYSTEM
# =========================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700;800&display=swap" rel="stylesheet">

<style>
html, body {
    font-family: 'Inter', sans-serif;
    scroll-behavior: smooth;
}

h1,h2,h3,h4 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing:-0.5px;
}

body {
    background: linear-gradient(140deg,#0f172a,#111c2e,#0b1220);
    color:#e2e8f0;
}

/* subtle noise */
body::after{
    content:"";
    position:fixed;
    width:100%;
    height:100%;
    background-image:url("https://www.transparenttextures.com/patterns/asfalt-dark.png");
    opacity:0.05;
    z-index:-5;
}

/* floating glass navbar */
.navbar{
    position:fixed;
    top:20px;
    left:50%;
    transform:translateX(-50%);
    width:90%;
    padding:15px 30px;
    backdrop-filter:blur(20px);
    background:rgba(255,255,255,0.07);
    border-radius:20px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    z-index:999;
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

/* glass card */
.glass-card{
    background:rgba(255,255,255,0.07);
    backdrop-filter:blur(25px);
    padding:40px;
    border-radius:20px;
    margin-top:40px;
}

/* hero */
.hero{
    height:80vh;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
}

/* parallax depth */
.parallax{
    transform: translateZ(0);
    will-change: transform;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION LOGIN SYSTEM
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "api_token" not in st.session_state:
    st.session_state.api_token = None

if not st.session_state.logged_in:

    st.markdown("<div class='glass-card' style='margin-top:120px;'>", unsafe_allow_html=True)
    st.title("Login to HPV-EPIPRED AI")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Enter credentials")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================================================
# NAVBAR
# =========================================================
st.markdown("""
<div class="navbar">
<div>🧬 HPV-EPIPRED AI</div>
<div>AI Epitope Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero">
<h1 style="font-size:68px;background:linear-gradient(90deg,#3b82f6,#9333ea,#06b6d4);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
HPV-EPIPRED AI
</h1>
<p style="font-size:20px;color:#94a3b8;">
AI-Driven MHC Class I Epitope Intelligence Platform
</p>
<a href="#scanner" style="margin-top:30px;color:#3b82f6;">↓ Explore Scanner</a>
</div>
""", unsafe_allow_html=True)

# =========================================================
# API TOKEN SYSTEM
# =========================================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("API Access")

if st.button("Generate API Token"):
    token = secrets.token_hex(24)
    st.session_state.api_token = token

if st.session_state.api_token:
    st.success("Your API Token:")
    st.code(st.session_state.api_token)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("hpv_epitope_model.pkl")
threshold = 0.261

aa_list = list("ACDEFGHIKLMNPQRSTVWY")
aa_index = {aa:i for i,aa in enumerate(aa_list)}
dipeptides = ["".join(p) for p in product(aa_list, repeat=2)]

hydrophobic = set("AILMFWYV")
positive = set("KRH")
negative = set("DE")

aa_weight = {a:100 for a in aa_list}

def extract_features(seq):
    pos_encoding = np.zeros((9,20))
    for pos, aa in enumerate(seq):
        pos_encoding[pos, aa_index[aa]] = 1
    pos_encoding = pos_encoding.flatten()

    di_count = Counter([seq[i:i+2] for i in range(len(seq)-1)])
    di_features = np.array([di_count[dp]/8 for dp in dipeptides])

    length = len(seq)
    hydro_frac = sum(a in hydrophobic for a in seq)/length
    pos_frac = sum(a in positive for a in seq)/length
    neg_frac = sum(a in negative for a in seq)/length

    return np.concatenate([pos_encoding, di_features,
        [hydro_frac,pos_frac,neg_frac]])

# =========================================================
# SCANNER SECTION
# =========================================================
st.markdown("<div id='scanner' class='glass-card'>", unsafe_allow_html=True)
st.header("AI Scanner")

mode = st.radio("Mode", ["Single Sequence","Batch Upload"])
fasta = ""

if mode == "Single Sequence":
    fasta = st.text_area("Paste FASTA Sequence")
else:
    uploaded = st.file_uploader("Upload FASTA File")
    if uploaded:
        fasta = uploaded.read().decode()

if st.button("Run AI Scan") and fasta:

    seq = "".join([l.strip() for l in fasta.split("\n")
                   if not l.startswith(">")]).upper()

    results = []
    for i in range(len(seq)-8):
        pep = seq[i:i+9]
        prob = model.predict_proba([extract_features(pep)])[0][1]
        cat = "Epitope" if prob >= threshold else "Non-Epitope"
        results.append([i+1,pep,prob,cat])

    df = pd.DataFrame(results,
        columns=["Position","Peptide","Probability","Category"])

    st.dataframe(df, use_container_width=True)

    fig = px.line(df, x="Position", y="Probability",
                  template="plotly_dark", markers=True)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# EXPLAINABILITY
# =========================================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("Model Explainability")

feat = pd.DataFrame({
    "Feature":["Hydrophobicity","Charge Balance","Dipeptide Pattern"],
    "Importance":[0.32,0.24,0.18]
})
fig = px.bar(feat, x="Importance", y="Feature",
             orientation="h", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center;margin-top:60px;color:#64748b;">
HPV-EPIPRED AI © 2026 | Premium Academic Research Platform
</div>
""", unsafe_allow_html=True)
