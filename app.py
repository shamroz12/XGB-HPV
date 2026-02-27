import streamlit as st
st.set_page_config(page_title="HPV-EPIPRED", page_icon="🧬", layout="wide")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from collections import Counter
import math
import time

# =========================================================
# THEME
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

toggle = st.toggle("🌗 Dark Mode", value=True)
theme = "dark" if toggle else "light"

# =========================================================
# GLOBAL AI STARTUP CSS
# =========================================================
st.markdown(f"""
<style>

.stApp {{
    background: {"#0a0f1c" if theme=="dark" else "#f8fafc"};
    font-family: 'Inter', sans-serif;
}}

.navbar {{
    display:flex;
    justify-content:space-between;
    padding:20px 40px;
    font-weight:600;
    color:white;
}}

.nav-title {{
    font-size:22px;
}}

.hero {{
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    padding:150px 20px;
}}

.gradient-text {{
    font-size:80px;
    font-weight:800;
    background: linear-gradient(90deg,#3b82f6,#9333ea,#06b6d4);
    background-size:200% auto;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:shine 4s linear infinite;
}}

@keyframes shine {{
    to {{ background-position:200% center; }}
}}

.typing {{
    font-size:22px;
    color:#94a3b8;
    margin-top:20px;
}}

.glass {{
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border-radius:30px;
    padding:60px;
    box-shadow:0 20px 60px rgba(0,0,0,0.4);
    animation:pulse 4s infinite alternate;
}}

@keyframes pulse {{
    from {{ box-shadow:0 0 40px rgba(59,130,246,0.4); }}
    to {{ box-shadow:0 0 80px rgba(147,51,234,0.6); }}
}}

.footer {{
    text-align:center;
    padding:40px;
    color:gray;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FLOATING GRADIENT BLOBS
# =========================================================
st.components.v1.html("""
<style>
.blob {
  position: fixed;
  border-radius: 50%;
  filter: blur(120px);
  opacity:0.6;
  z-index:-1;
}
#blob1 { width:400px; height:400px; background:#3b82f6; top:10%; left:10%; }
#blob2 { width:350px; height:350px; background:#9333ea; bottom:15%; right:15%; }
</style>
<div id="blob1" class="blob"></div>
<div id="blob2" class="blob"></div>
""", height=0)

# =========================================================
# NAVBAR
# =========================================================
st.markdown("""
<div class="navbar">
    <div class="nav-title">🧬 HPV-EPIPRED</div>
    <div>AI Epitope Intelligence</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO WITH AI TYPING
# =========================================================
st.markdown("""
<div class="hero">
    <div class="gradient-text">HPV-EPIPRED</div>
    <div class="typing" id="typing"></div>
</div>
<script>
var text = "Predicting epitopes in real time using AI...";
var i = 0;
function typeWriter() {
  if (i < text.length) {
    document.getElementById("typing").innerHTML += text.charAt(i);
    i++;
    setTimeout(typeWriter, 40);
  }
}
typeWriter();
</script>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
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

    global_features = np.array([
        hydro_frac, arom_frac, pos_frac,
        neg_frac, net_charge, entropy, avg_weight
    ])

    return np.concatenate([pos_encoding, di_features, global_features])

# =========================================================
# LOGIN-STYLE SCANNER PANEL
# =========================================================
st.markdown('<div class="glass">', unsafe_allow_html=True)

fasta = st.text_area("Paste HPV Protein FASTA Sequence")
run = st.button("Run AI Scan")

if run:
    progress = st.progress(0)
    for percent in range(100):
        time.sleep(0.01)
        progress.progress(percent + 1)

    seq = "".join([l.strip() for l in fasta.split("\n") if not l.startswith(">")]).upper()

    results = []
    for i in range(len(seq)-8):
        pep = seq[i:i+9]
        prob = model.predict_proba([extract_features(pep)])[0][1]
        results.append(prob)

    df = pd.DataFrame({
        "Position": range(1,len(results)+1),
        "Probability": results
    })

    # Animated graph
    fig = px.line(df, x="Position", y="Probability")
    fig.update_traces(line=dict(width=3))
    fig.add_hline(y=threshold, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    # Gauge
    mean_prob = np.mean(results)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mean_prob,
        gauge={'axis': {'range': [0,1]}},
        title={'text': "Model Confidence"}
    ))
    st.plotly_chart(gauge, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
HPV-EPIPRED © 2026 | Developed by Shamroz
</div>
""", unsafe_allow_html=True)
