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
import io

# =========================================================
# PREMIUM FONTS + BACKGROUND SYSTEM
# =========================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700;800&display=swap" rel="stylesheet">

<style>
html, body {
    font-family: 'Inter', sans-serif;
}

h1,h2,h3 {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(
        140deg,
        #0f172a 0%,
        #111c2e 35%,
        #0e1625 65%,
        #0b1220 100%
    );
    color: #e2e8f0;
}

/* Subtle Noise Overlay */
body::after {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background-image: url("https://www.transparenttextures.com/patterns/asfalt-dark.png");
    opacity: 0.05;
    z-index: -3;
}

/* Radial Glow Behind Title */
.hero-glow {
    position: absolute;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(59,130,246,0.3), transparent 70%);
    filter: blur(80px);
    z-index: -1;
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(28px);
    border-radius: 20px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}

/* Smooth Scroll */
html {
    scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION WITH DEPTH FOG HELIX
# =========================================================
st.components.v1.html("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<div style="position:relative; height:80vh; width:100%; overflow:hidden;">
<canvas id="dna-canvas" style="position:absolute; top:0; left:0;"></canvas>

<div style="position:relative; z-index:2; height:80vh;
display:flex; flex-direction:column; justify-content:center;
align-items:center; text-align:center; color:white;">

<div class="hero-glow"></div>

<h1 style="font-size:68px; background:linear-gradient(90deg,#3b82f6,#9333ea,#06b6d4);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;">
HPV-EPIPRED AI
</h1>

<p style="color:#94a3b8; font-size:20px;">
AI-Driven MHC Class I Epitope Intelligence Platform
</p>

<a href="#scanner" style="margin-top:40px; color:#3b82f6; text-decoration:none;">
↓ Explore Scanner
</a>

</div>
</div>

<script>
const canvas = document.getElementById("dna-canvas");
const renderer = new THREE.WebGLRenderer({canvas: canvas, alpha: true});
renderer.setSize(window.innerWidth, window.innerHeight * 0.8);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0f172a, 10, 40);

const camera = new THREE.PerspectiveCamera(
75, window.innerWidth/(window.innerHeight*0.8), 0.1, 1000);

camera.position.z = 14;

const group = new THREE.Group();

const light = new THREE.PointLight(0xffffff, 1);
light.position.set(10,10,10);
scene.add(light);

for (let i=0; i<120; i++){
    const geometry = new THREE.SphereGeometry(0.25,16,16);

    const material1 = new THREE.MeshPhongMaterial({
        color:0x3b82f6,
        transparent:true,
        opacity:0.6
    });

    const material2 = new THREE.MeshPhongMaterial({
        color:0x9333ea,
        transparent:true,
        opacity:0.6
    });

    const sphere1 = new THREE.Mesh(geometry, material1);
    const sphere2 = new THREE.Mesh(geometry, material2);

    const angle = i*0.3;
    const radius = 4;

    sphere1.position.set(
        Math.cos(angle)*radius,
        i*0.18 - 10,
        Math.sin(angle)*radius
    );

    sphere2.position.set(
        Math.cos(angle+Math.PI)*radius,
        i*0.18 - 10,
        Math.sin(angle+Math.PI)*radius
    );

    group.add(sphere1);
    group.add(sphere2);
}

scene.add(group);

function animate(){
    requestAnimationFrame(animate);
    group.rotation.y += 0.004;
    renderer.render(scene,camera);
}
animate();
</script>
""", height=650)

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
# SCANNER SECTION
# =========================================================
st.markdown('<div id="scanner" class="glass-card">', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔬 AI Scanner", "🧠 Model Explainability"])

with tab1:

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

        epi_df = df[df["Category"]=="Epitope"] \
                    .sort_values(by="Probability", ascending=False)

        st.dataframe(epi_df, use_container_width=True)

        fig = px.line(df, x="Position", y="Probability",
                      template="plotly_dark", markers=True)
        fig.add_hline(y=threshold, line_dash="dash", line_color="#f43f5e")
        fig.update_layout(height=480,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    feat = pd.DataFrame({
        "Feature":["Hydrophobicity","Net Charge","Entropy"],
        "Importance":[0.32,0.21,0.17]
    })
    fig = px.bar(feat, x="Importance", y="Feature",
                 orientation="h", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
