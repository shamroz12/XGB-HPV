import streamlit as st

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="HPV-EPIPRED",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from itertools import product
from collections import Counter
import math

# ===============================
# THEME STATE
# ===============================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_toggle = st.sidebar.toggle(
    "🌗 Dark / Light Mode",
    value=True if st.session_state.theme=="dark" else False
)

st.session_state.theme = "dark" if theme_toggle else "light"
theme = st.session_state.theme

# ===============================
# GLOBAL CSS (REAL WORKING VERSION)
# ===============================
st.markdown(f"""
<style>

html, body, [class*="css"]  {{
    font-family: 'Segoe UI', sans-serif;
}}

.stApp {{
    background: {"#0b1120" if theme=="dark" else "#f1f5f9"};
}}

section.main > div {{
    padding-top: 2rem;
}}

.glass {{
    background: {"rgba(255,255,255,0.07)" if theme=="dark" else "rgba(255,255,255,0.8)"};
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 60px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    transition: 0.4s ease;
}}

.glass:hover {{
    transform: translateY(-6px);
}}

.hero {{
    text-align: center;
    padding: 120px 20px;
    color: {"white" if theme=="dark" else "#0f172a"};
    animation: fadeUp 1.4s ease forwards;
}}

@keyframes fadeUp {{
    from {{opacity:0; transform:translateY(40px);}}
    to {{opacity:1; transform:translateY(0);}}
}}

.footer {{
    text-align:center;
    padding:40px;
    color:gray;
    font-size:14px;
}}

</style>
""", unsafe_allow_html=True)

# ===============================
# PARTICLE BACKGROUND (FIXED VERSION)
# ===============================
st.components.v1.html(f"""
<style>
#particles-js {{
  position: fixed;
  width: 100%;
  height: 100%;
  z-index: -1;
  top: 0;
  left: 0;
  background: {"#0b1120" if theme=="dark" else "#f1f5f9"};
}}
</style>

<div id="particles-js"></div>

<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {{
  particles: {{
    number: {{ value: 60 }},
    color: {{ value: "#3b82f6" }},
    shape: {{ type: "circle" }},
    opacity: {{ value: 0.5 }},
    size: {{ value: 3 }},
    line_linked: {{
      enable: true,
      distance: 140,
      color: "#3b82f6",
      opacity: 0.15,
      width: 1
    }},
    move: {{
      enable: true,
      speed: 1.5
    }}
  }}
}});
</script>
""", height=0)

# ===============================
# LOAD MODEL
# ===============================
try:
    model = joblib.load("hpv_epitope_model.pkl")
except:
    st.error("Model file not found.")
    st.stop()

threshold = 0.261

# ===============================
# FEATURE ENGINEERING
# ===============================
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

# ===============================
# NAVIGATION
# ===============================
page = st.sidebar.radio(
    "Navigation",
    ["Home","Epitope Scanner","Methods"]
)

# ===============================
# HOME
# ===============================
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="glass">
            <h1 style="font-size:72px;">HPV-EPIPRED</h1>
            <h3>AI-Driven MHC Class I Epitope Prediction</h3>
            <p>Machine Learning-Based Immunogenic Hotspot Identification</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# EPITOPE SCANNER
# ===============================
elif page == "Epitope Scanner":

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    fasta = st.text_area("Paste HPV Protein FASTA Sequence")
    run = st.button("🔬 Run Epitope Scan")
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        seq = "".join(
            [l.strip() for l in fasta.split("\n") if not l.startswith(">")]
        ).upper()

        if len(seq) < 9:
            st.error("Sequence must be ≥ 9 amino acids.")
            st.stop()

        results = []
        for i in range(len(seq)-8):
            pep = seq[i:i+9]
            prob = model.predict_proba([extract_features(pep)])[0][1]

            results.append({
                "Start": i+1,
                "Peptide": pep,
                "Probability": round(prob,3)
            })

        df = pd.DataFrame(results)

        fig = px.line(
            df,
            x="Start",
            y="Probability",
            template="plotly_dark" if theme=="dark" else "simple_white",
            title="Epitope Probability Landscape"
        )
        fig.add_hline(y=threshold, line_dash="dash")

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer">
HPV-EPIPRED © 2026 | HPV Immunoinformatics Research
</div>
""", unsafe_allow_html=True)
