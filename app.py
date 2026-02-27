import streamlit as st
st.set_page_config(page_title="HPV-EPIPRED AI", page_icon="🧬", layout="wide")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
from collections import Counter
import math
import time
import io

# =====================================================
# PREMIUM FONTS + GLOBAL STYLE
# =====================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1,h2,h3 {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #050a18;
}

.navbar {
    position: fixed;
    top: 0;
    width: 100%;
    padding: 20px 80px;
    display: flex;
    justify-content: space-between;
    background: rgba(5,10,25,0.9);
    backdrop-filter: blur(12px);
    z-index: 999;
    color: white;
}

.main-container {
    margin-top: 100px;
}

.hero {
    text-align: center;
    padding: 60px 20px;
}

.gradient-title {
    font-size: 70px;
    font-weight: 800;
    background: linear-gradient(90deg,#22d3ee,#6366f1,#a855f7);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 30px;
}

.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    text-align:center;
}

</style>

<div class="navbar">
    <div>🧬 HPV-EPIPRED AI</div>
    <div>AI Immunoinformatics Platform</div>
</div>

<div class="main-container">
""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero">
    <div class="gradient-title">HPV-EPIPRED AI</div>
    <h3 style="color:#94a3b8;">Next-Generation Epitope Intelligence Platform</h3>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("hpv_epitope_model.pkl")
threshold = 0.261

# =====================================================
# FEATURE ENGINEERING
# =====================================================
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

# =====================================================
# HELIX LOADER
# =====================================================
def render_helix(height=250):
    st.components.v1.html(f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <div id="dna-container"></div>
    <script>
    const container = document.getElementById("dna-container");
    container.innerHTML = "";
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth/{height}, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({alpha:true});
    renderer.setSize(window.innerWidth, {height});
    container.appendChild(renderer.domElement);

    const geometry = new THREE.TorusKnotGeometry(1,0.3,120,16);
    const material = new THREE.MeshBasicMaterial({color:0x22d3ee, wireframe:true});
    const dna = new THREE.Mesh(geometry, material);
    scene.add(dna);

    camera.position.z = 5;

    function animate() {{
      requestAnimationFrame(animate);
      dna.rotation.x += 0.015;
      dna.rotation.y += 0.015;
      renderer.render(scene, camera);
    }}
    animate();
    </script>
    """, height=height)

# =====================================================
# SCANNER SECTION
# =====================================================
st.header("AI Epitope Scanner")

mode = st.radio("Mode", ["Single Sequence","Batch Upload"])

fasta = ""
if mode == "Single Sequence":
    fasta = st.text_area("Paste FASTA Sequence")
else:
    uploaded = st.file_uploader("Upload FASTA File")
    if uploaded:
        fasta = uploaded.read().decode()

# =====================================================
# RUN SCAN
# =====================================================
if st.button("Run AI Scan") and fasta:

    loader = st.empty()
    with loader:
        render_helix(250)
        st.markdown("<h4 style='text-align:center;color:#94a3b8;'>Running AI inference engine...</h4>", unsafe_allow_html=True)

    time.sleep(2)
    loader.empty()

    seq = "".join([l.strip() for l in fasta.split("\n")
                   if not l.startswith(">")]).upper()

    results = []
    for i in range(len(seq)-8):
        pep = seq[i:i+9]
        prob = model.predict_proba([extract_features(pep)])[0][1]

        if prob >= 0.6:
            cat = "High Epitope"
        elif prob >= threshold:
            cat = "Moderate Epitope"
        else:
            cat = "Non-Epitope"

        results.append([i+1,pep,prob,cat])

    df = pd.DataFrame(results,
        columns=["Position","Peptide","Probability","Category"])

    # ================= SUMMARY =================
    col1,col2,col3 = st.columns(3)
    col1.metric("Total Peptides", len(df))
    col2.metric("High Epitopes",
                len(df[df["Category"]=="High Epitope"]))
    col3.metric("Mean Probability",
                round(df["Probability"].mean(),3))

    # ================= PLOT =================
    fig = px.line(df, x="Position", y="Probability",
                  template="plotly_dark", markers=True)
    fig.add_hline(y=threshold, line_dash="dash")
    fig.update_layout(height=450)

    col1,col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df["Probability"].mean(),
            gauge={'axis':{'range':[0,1]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

    # ================= TABLE =================
    def style_row(row):
        if row["Category"]=="High Epitope":
            return ["background-color:#16a34a;color:white"]*len(row)
        elif row["Category"]=="Moderate Epitope":
            return ["background-color:#f59e0b;color:white"]*len(row)
        else:
            return ["background-color:#dc2626;color:white"]*len(row)

    st.dataframe(df.style.apply(style_row, axis=1),
                 use_container_width=True)

    # ================= DOWNLOAD =================
    st.subheader("Download Results")

    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV",
                       csv,
                       "epitope_results.csv")

    fig.write_html("plot.html")
    with open("plot.html","rb") as f:
        st.download_button("Download Interactive Plot",
                           f,
                           "epitope_plot.html")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr style="border:1px solid rgba(255,255,255,0.1);">
<p style="text-align:center;color:gray;">
HPV-EPIPRED AI © 2026 | Developed by Shamroz
</p>
""", unsafe_allow_html=True)
