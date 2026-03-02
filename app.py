import streamlit as st
st.set_page_config(page_title="HPV-EPIPRED AI™", page_icon="🧬", layout="wide")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from collections import Counter
import math
import shap
import matplotlib.pyplot as plt

# =========================================================
# PROFESSIONAL DESIGN SYSTEM (HYBRID + BRANDING)
# =========================================================

theme = st.toggle("🌙 Dark Mode", value=True)

if theme:
    bg_color = "#0f172a"
    text_color = "#f1f5f9"
    card_color = "rgba(30,41,59,0.65)"
else:
    bg_color = "#f8fafc"
    text_color = "#0f172a"
    card_color = "rgba(255,255,255,0.75)"

st.markdown(f"""
<style>
:root {{
    --primary: #4f46e5;
    --accent: #14b8a6;
    --warning: #f59e0b;
    --danger: #f43f5e;
}}

.stApp {{
    background: {bg_color} !important;
    color: {text_color};
}}

.glass-card {{
    background: {card_color};
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 35px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    margin-bottom: 40px;
}}

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&display=swap');

html, body, [class*="css"] {{
    font-family: "Times New Roman", Times, serif !important;
}}

h1 {{
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 72px !important;
    letter-spacing: -2px;
}}

h2 {{
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 30px !important;
    margin-top: 2.5rem !important;
}}

p, label, span {{
    font-size: 17px;
    line-height: 1.7;
}}

[data-testid="stMetricValue"] {{
    font-family: 'Sora', sans-serif !important;
    font-size: 40px !important;
    font-weight: 700 !important;
}}

.stButton > button {{
    background-color: var(--primary);
    color: white;
    border-radius: 8px;
    padding: 8px 18px;
    border: none;
    font-weight: 600;
}}

[data-testid="stDataFrame"] thead tr th {{
    background-color: #f1f5f9 !important;
    font-weight: bold !important;
}}

.section-divider {{
    height: 1px;
    background: linear-gradient(to right, transparent, #4f46e5, transparent);
    margin: 3rem 0;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="glass-card">
<h1 style="color:#4f46e5;">HPV–EPIPRED AI™</h1>
<p>Precision Immunoinformatics Platform for High-Risk HPV Epitope Mapping</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================

model = joblib.load("hpv_epitope_model.pkl")
threshold = 0.261

aa_list = list("ACDEFGHIKLMNPQRSTVWY")
aa_index = {aa:i for i,aa in enumerate(aa_list)}
dipeptides = ["".join(p) for p in product(aa_list, repeat=2)]

hydrophobic = set("AILMFWYV")
aromatic = set("FWY")
positive = set("KRH")
negative = set("DE")

aa_weight = {
"A":89,"C":121,"D":133,"E":147,"F":165,"G":75,"H":155,
"I":131,"K":146,"L":131,"M":149,"N":132,"P":115,
"Q":146,"R":174,"S":105,"T":119,"V":117,"W":204,"Y":181
}

def extract_features(seq):
    pos_encoding = np.zeros((9,20))
    for pos, aa in enumerate(seq):
        pos_encoding[pos, aa_index[aa]] = 1
    pos_encoding = pos_encoding.flatten()

    di_count = Counter([seq[i:i+2] for i in range(len(seq)-1)])
    di_features = np.array([di_count[dp]/8 for dp in dipeptides])

    length = len(seq)
    aa_count = Counter(seq)

    hydro_frac = sum(a in hydrophobic for a in seq)/length
    arom_frac = sum(a in aromatic for a in seq)/length
    pos_frac = sum(a in positive for a in seq)/length
    neg_frac = sum(a in negative for a in seq)/length
    net_charge = pos_frac - neg_frac
    entropy = -sum((aa_count[a]/length)*math.log2(aa_count[a]/length)
                   for a in aa_count)
    avg_weight = sum(aa_weight[a] for a in seq)/length

    return np.concatenate([pos_encoding, di_features,
        [hydro_frac,arom_frac,pos_frac,neg_frac,net_charge,entropy,avg_weight]])

# =========================================================
# SCANNER
# =========================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("## AI Epitope Scanner")

mode = st.radio("Mode", ["Single Sequence","Batch Upload"])
fasta = ""

if mode == "Single Sequence":
    fasta = st.text_area("Paste FASTA Sequence")
else:
    uploaded = st.file_uploader("Upload FASTA File")
    if uploaded:
        fasta = uploaded.read().decode()

if st.button("Run AI Scan") and fasta:

    seq = "".join([l.strip() for l in fasta.split("\n") if not l.startswith(">")]).upper()

    results = []
    for i in range(len(seq) - 8):
        pep = seq[i:i+9]
        prob = model.predict_proba([extract_features(pep)])[0][1]
        cat = "Epitope" if prob >= threshold else "Non-Epitope"
        results.append([i+1, pep, prob, cat])

    df = pd.DataFrame(results, columns=["Position", "Peptide", "Probability", "Category"])
    epitope_df = df[df["Category"]=="Epitope"]
    non_df = df[df["Category"]=="Non-Epitope"]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Executive Immunogenic Summary")

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total 9-mers", len(df))
    col2.metric("Predicted Epitopes", len(epitope_df))
    col3.metric("Non-Epitopes", len(non_df))
    col4.metric("Mean Score", round(df["Probability"].mean(),3))

    st.markdown("### Predicted Epitopes")
    st.dataframe(epitope_df)

    st.markdown("### Predicted Non-Epitopes")
    st.dataframe(non_df)

    fig = px.line(df, x="Position", y="Probability")
    fig.add_hline(y=threshold, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    heatmap_fig = px.imshow(
        [df["Probability"].values],
        aspect="auto",
        color_continuous_scale=[
            [0, "#0ea5e9"],
            [0.5, "#14b8a6"],
            [0.75, "#f59e0b"],
            [1, "#f43f5e"]
        ]
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown("## SHAP Explainability")

    sample_peptide = df.iloc[0]["Peptide"]
    features = extract_features(sample_peptide)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values([features])
    fig2, ax = plt.subplots()
    shap.summary_plot(shap_values, [features], show=False)
    st.pyplot(fig2)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
© 2026 HPV–EPIPRED AI™  
Precision Immunoinformatics Platform  
Research Edition v1.0  
Developed by Shamroz Ahmad  
All Rights Reserved
""")
