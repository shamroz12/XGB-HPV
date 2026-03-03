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
import streamlit.components.v1 as components

# =========================================================
# REMOVE STREAMLIT DEFAULT PADDING
# =========================================================
st.markdown("""
<style>
.block-container { padding-top: 0rem !important; }
header {visibility: hidden;}
html { scroll-behavior: smooth; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# PREMIUM FONT SYSTEM
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0f172a;
    color: white;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CINEMATIC HERO (CELLULAR + AI NETWORK)
# =========================================================
components.html("""
<style>
.hero {
    position: relative;
    width: 100%;
    height: 100vh;
    overflow: hidden;
    background: radial-gradient(circle at center, #1e293b 0%, #0f172a 80%);
}

.bg-layer {
    position: absolute;
    width: 100%;
    height: 100%;
    background-image: url('https://images.unsplash.com/photo-1581090700227-4c4f50f0e9e1?auto=format&fit=crop&w=2000&q=80');
    background-size: cover;
    background-position: center;
    opacity: 0.25;
    z-index: 0;
}

canvas {
    position: absolute;
    top: 0;
    left: 0;
}

.hero-content {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    z-index: 3;
}

.hero-title {
    font-size: 72px;
    background: linear-gradient(90deg,#3b82f6,#9333ea,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 22px;
    opacity: 0.85;
}

.cta {
    margin-top: 25px;
    font-size: 18px;
    color: #38bdf8;
    text-decoration: none;
}
</style>

<div class="hero">

    <div class="bg-layer"></div>

    <canvas id="immune"></canvas>
    <canvas id="network"></canvas>

    <div class="hero-content">
        <div class="hero-title">HPV–EPIPRED AI</div>
        <div class="hero-sub">Cinematic Immunoinformatics Intelligence Platform</div>
        <a href="#scanner" class="cta">↓ Launch Scanner</a>
    </div>

</div>

<script>

const immune = document.getElementById("immune");
const network = document.getElementById("network");

const ictx = immune.getContext("2d");
const nctx = network.getContext("2d");

function resize(){
    immune.width = window.innerWidth;
    immune.height = window.innerHeight;
    network.width = window.innerWidth;
    network.height = window.innerHeight;
}
resize();
window.addEventListener("resize", resize);

// IMMUNE CELLS
let cells = [];
for(let i=0;i<10;i++){
    cells.push({
        x:Math.random()*window.innerWidth,
        y:Math.random()*window.innerHeight,
        r:60+Math.random()*40,
        pulse:Math.random()*Math.PI
    });
}

function drawImmune(){
    ictx.clearRect(0,0,immune.width,immune.height);

    cells.forEach(c=>{
        c.pulse+=0.02;

        let g = ictx.createRadialGradient(c.x,c.y,c.r*0.2,c.x,c.y,c.r);
        g.addColorStop(0,"rgba(139,92,246,0.7)");
        g.addColorStop(1,"rgba(139,92,246,0.05)");

        ictx.beginPath();
        ictx.arc(c.x,c.y,c.r,0,Math.PI*2);
        ictx.fillStyle=g;
        ictx.fill();
    });

    requestAnimationFrame(drawImmune);
}
drawImmune();

// AI NETWORK
let nodes=[];
for(let i=0;i<60;i++){
    nodes.push({
        x:Math.random()*window.innerWidth,
        y:Math.random()*window.innerHeight,
        vx:(Math.random()-0.5)*0.5,
        vy:(Math.random()-0.5)*0.5
    });
}

function drawNetwork(){
    nctx.clearRect(0,0,network.width,network.height);

    nodes.forEach(n=>{
        n.x+=n.vx;
        n.y+=n.vy;
        if(n.x<0||n.x>network.width) n.vx*=-1;
        if(n.y<0||n.y>network.height) n.vy*=-1;

        nctx.beginPath();
        nctx.arc(n.x,n.y,2,0,Math.PI*2);
        nctx.fillStyle="rgba(56,189,248,0.7)";
        nctx.fill();
    });

    for(let i=0;i<nodes.length;i++){
        for(let j=i+1;j<nodes.length;j++){
            let dx=nodes[i].x-nodes[j].x;
            let dy=nodes[i].y-nodes[j].y;
            let dist=Math.sqrt(dx*dx+dy*dy);
            if(dist<120){
                nctx.beginPath();
                nctx.moveTo(nodes[i].x,nodes[i].y);
                nctx.lineTo(nodes[j].x,nodes[j].y);
                nctx.strokeStyle="rgba(56,189,248,0.15)";
                nctx.stroke();
            }
        }
    }

    requestAnimationFrame(drawNetwork);
}
drawNetwork();

</script>
""", height=900)

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
st.markdown('<div id="scanner"></div>', unsafe_allow_html=True)

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

        seq = "".join([l.strip() for l in fasta.split("\n") if not l.startswith(">")]).upper()

        results = []
        for i in range(len(seq) - 8):
            pep = seq[i:i+9]
            prob = model.predict_proba([extract_features(pep)])[0][1]
            cat = "Epitope" if prob >= threshold else "Non-Epitope"
            results.append([i+1, pep, prob, cat])

        df = pd.DataFrame(results, columns=["Position","Peptide","Probability","Category"])
        st.dataframe(df, use_container_width=True)

        mean_prob = df["Probability"].mean()

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prob,
            number={'valueformat': ".2f"},
            title={'text': "Global Immunogenic Score"},
            gauge={'axis': {'range': [0, 1]}}
        ))

        st.plotly_chart(gauge, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "epitope_results.csv")

with tab2:
    st.write("Explainability module coming soon...")

st.markdown("---")
st.markdown("© 2026 HPV–EPIPRED AI | Developed by Shamroz")
