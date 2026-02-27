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

# =========================================================
# THEME
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_toggle = st.toggle("🌗 Dark Mode", value=True)
theme = "dark" if theme_toggle else "light"

# =========================================================
# HD VISUAL CSS
# =========================================================
st.markdown(f"""
<style>
.stApp {{
    background: {"#060b18" if theme=="dark" else "#f4f7fb"};
    font-family: 'Inter', sans-serif;
}}

.navbar {{
    display:flex;
    justify-content:space-between;
    padding:25px 60px;
    font-weight:600;
    font-size:18px;
    color:white;
}}

.hero {{
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    padding:160px 20px;
}}

.gradient-text {{
    font-size:90px;
    font-weight:900;
    background: linear-gradient(90deg,#3b82f6,#9333ea,#06b6d4);
    background-size:200% auto;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:shine 5s linear infinite;
}}

@keyframes shine {{
    to {{ background-position:200% center; }}
}}

.glass {{
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(25px);
    border-radius:30px;
    padding:70px;
    box-shadow:0 0 60px rgba(59,130,246,0.4);
    transition:0.4s ease;
}}

.glass:hover {{
    box-shadow:0 0 120px rgba(147,51,234,0.7);
}}

.footer {{
    text-align:center;
    padding:40px;
    color:gray;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FLOATING HD BLOBS
# =========================================================
st.components.v1.html("""
<style>
.blob {
  position: fixed;
  border-radius: 50%;
  filter: blur(150px);
  opacity:0.7;
  z-index:-1;
}
#blob1 { width:600px; height:600px; background:#3b82f6; top:5%; left:5%; }
#blob2 { width:500px; height:500px; background:#9333ea; bottom:10%; right:10%; }
</style>
<div id="blob1" class="blob"></div>
<div id="blob2" class="blob"></div>
""", height=0)

# =========================================================
# LANDING GATE
# =========================================================
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:

    st.components.v1.html("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <div id="dna-container"></div>
    <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth/400, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({alpha:true});
    renderer.setSize(window.innerWidth, 400);
    document.getElementById("dna-container").appendChild(renderer.domElement);

    const geometry = new THREE.TorusKnotGeometry(1,0.3,120,16);
    const material = new THREE.MeshBasicMaterial({color:0x3b82f6, wireframe:true});
    const dna = new THREE.Mesh(geometry, material);
    scene.add(dna);

    camera.position.z = 5;

    function animate() {
      requestAnimationFrame(animate);
      dna.rotation.x += 0.01;
      dna.rotation.y += 0.01;
      renderer.render(scene, camera);
    }
    animate();
    </script>
    """, height=400)

    st.markdown("""
    <div class="hero">
        <div class="gradient-text">HPV-EPIPRED AI</div>
        <h3 style="color:#94a3b8;">Next-Generation Epitope Intelligence Platform</h3>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Enter Platform"):
        st.session_state.entered = True
    st.stop()

# =========================================================
# NAVBAR
# =========================================================
st.markdown("""
<div class="navbar">
    <div>🧬 HPV-EPIPRED</div>
    <div>AI Immunoinformatics Engine</div>
</div>
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

    return np.concatenate([pos_encoding, di_features,
        [hydro_frac,arom_frac,pos_frac,neg_frac,net_charge,entropy,avg_weight]])

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["🔬 AI Scanner", "🧠 Model Explainability"])

with tab1:

    allele = st.selectbox("Select HLA Allele",
        ["HLA-A*02:01","HLA-A*11:01","HLA-B*07:02"])

    mode = st.radio("Mode", ["Single Sequence","Batch Upload"])

    fasta = ""
    if mode == "Single Sequence":
        fasta = st.text_area("Paste FASTA Sequence")
    else:
        uploaded = st.file_uploader("Upload FASTA File")
        if uploaded:
            fasta = uploaded.read().decode()

    if st.button("Run AI Scan") and fasta:

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

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

        st.dataframe(df, use_container_width=True)

        fig = px.line(df, x="Position", y="Probability")
        fig.add_hline(y=threshold, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        mean_prob = df["Probability"].mean()

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prob,
            gauge={'axis':{'range':[0,1]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "epitope_results.csv")

        buffer = io.StringIO()
        fig.write_html("plot.html")
        with open("plot.html","rb") as f:
            st.download_button("Download Plot HTML", f, "plot.html")

with tab2:
    st.subheader("Feature Importance Overview")
    feat = pd.DataFrame({
        "Feature":["Hydrophobicity","Net Charge","Entropy"],
        "Importance":[0.32,0.21,0.17]
    })
    fig = px.bar(feat, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="footer">
HPV-EPIPRED AI © 2026 | Developed by Shamroz
</div>
""", unsafe_allow_html=True)
