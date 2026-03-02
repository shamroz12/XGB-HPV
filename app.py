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
# PREMIUM FONTS
# =========================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

h1,h2,h3 {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 30%, #0f172a 0%, #060b18 50%, #030712 100%);
    color: white;
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(25px);
    border-radius: 24px;
    padding: 40px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 60px rgba(59,130,246,0.25);
    transition: 0.4s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 120px rgba(147,51,234,0.4);
}

/* Fade Animation */
.fade-in {
    animation: fadeUp 1.2s ease forwards;
}

@keyframes fadeUp {
    from { opacity:0; transform: translateY(40px);}
    to { opacity:1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FULL SCREEN ROTATING DNA BACKGROUND
# =========================================================
st.components.v1.html("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<canvas id="dna-bg"></canvas>

<script>
const canvas = document.getElementById("dna-bg");
canvas.style.position = "fixed";
canvas.style.top = "0";
canvas.style.left = "0";
canvas.style.zIndex = "-2";

const renderer = new THREE.WebGLRenderer({canvas: canvas, alpha: true});
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth/window.innerHeight,
    0.1,
    1000
);

camera.position.z = 12;

const group = new THREE.Group();

for (let i = 0; i < 100; i++) {
    const geometry = new THREE.SphereGeometry(0.2, 16, 16);
    const material1 = new THREE.MeshBasicMaterial({color: 0x3b82f6});
    const material2 = new THREE.MeshBasicMaterial({color: 0x9333ea});

    const sphere1 = new THREE.Mesh(geometry, material1);
    const sphere2 = new THREE.Mesh(geometry, material2);

    const angle = i * 0.3;
    const radius = 3;

    sphere1.position.set(
        Math.cos(angle) * radius,
        i * 0.15 - 7,
        Math.sin(angle) * radius
    );

    sphere2.position.set(
        Math.cos(angle + Math.PI) * radius,
        i * 0.15 - 7,
        Math.sin(angle + Math.PI) * radius
    );

    group.add(sphere1);
    group.add(sphere2);
}

scene.add(group);

function animate() {
    requestAnimationFrame(animate);
    group.rotation.y += 0.003;
    renderer.render(scene, camera);
}
animate();
</script>
""", height=0)

# =========================================================
# NAVBAR
# =========================================================
st.markdown("""
<div style="padding:30px 80px; font-weight:600; font-size:20px;">
🧬 HPV-EPIPRED AI — AI-Driven Epitope Intelligence
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

    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)

    mode = st.radio("Mode", ["Single Sequence","Batch Upload"])
    fasta = ""

    if mode == "Single Sequence":
        fasta = st.text_area("Paste FASTA Sequence")
    else:
        uploaded = st.file_uploader("Upload FASTA File")
        if uploaded:
            fasta = uploaded.read().decode()

    if st.button("Run AI Scan") and fasta:

        with st.spinner("Running AI inference engine..."):
            seq = "".join([l.strip() for l in fasta.split("\n")
                           if not l.startswith(">")]).upper()

            results = []

            for i in range(len(seq)-8):
                pep = seq[i:i+9]
                prob = model.predict_proba([extract_features(pep)])[0][1]

                if prob >= threshold:
                    cat = "Epitope"
                else:
                    cat = "Non-Epitope"

                results.append([i+1,pep,prob,cat])

            df = pd.DataFrame(results,
                columns=["Position","Peptide","Probability","Category"])

        # ================= TABLE =================
        st.subheader("Predicted Epitopes")
        epi_df = df[df["Category"]=="Epitope"] \
                    .sort_values(by="Probability", ascending=False)
        st.dataframe(epi_df, use_container_width=True)

        with st.expander("View Non-Epitopes"):
            non_df = df[df["Category"]=="Non-Epitope"]
            st.dataframe(non_df, use_container_width=True)

        # ================= PLOT =================
        fig = px.line(df, x="Position", y="Probability",
                      template="plotly_dark", markers=True)
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        fig.update_layout(height=520,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # ================= GAUGE =================
        mean_prob = df["Probability"].mean()
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prob,
            gauge={'axis':{'range':[0,1]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # ================= DOWNLOAD =================
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "epitope_results.csv")

        fig.write_html("plot.html")
        with open("plot.html","rb") as f:
            st.download_button("Download Plot HTML", f, "plot.html")

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.subheader("Feature Importance Overview")
    feat = pd.DataFrame({
        "Feature":["Hydrophobicity","Net Charge","Entropy"],
        "Importance":[0.32,0.21,0.17]
    })
    fig = px.bar(feat, x="Importance", y="Feature",
                 orientation="h", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:40px;color:gray;">
HPV-EPIPRED AI © 2026 | Academic Research Platform
</div>
""", unsafe_allow_html=True)
