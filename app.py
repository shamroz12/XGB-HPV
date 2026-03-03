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
# REMOVE STREAMLIT DEFAULT GAP
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
}

header {visibility: hidden;}

html {
    scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# PREMIUM GLOBAL FONT SYSTEM
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&family=Sora:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    letter-spacing: -1px;
    font-weight: 700 !important;
}

p, label, span, div {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# NEURAL NETWORK BACKGROUND (FIXED)
# =========================================================
components.html("""
<canvas id="network-canvas" style="
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
z-index:-3;
pointer-events:none;"></canvas>

<script>
const netCanvas = document.getElementById("network-canvas");
const netCtx = netCanvas.getContext("2d");

function resizeCanvas(){
    netCanvas.width = window.innerWidth;
    netCanvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

let nodes = [];
for(let i=0;i<65;i++){
    nodes.push({
        x:Math.random()*window.innerWidth,
        y:Math.random()*window.innerHeight,
        vx:(Math.random()-0.5)*0.6,
        vy:(Math.random()-0.5)*0.6
    });
}

function animateNetwork(){
    netCtx.clearRect(0,0,netCanvas.width,netCanvas.height);

    nodes.forEach(n=>{
        n.x+=n.vx;
        n.y+=n.vy;

        if(n.x<0||n.x>netCanvas.width) n.vx*=-1;
        if(n.y<0||n.y>netCanvas.height) n.vy*=-1;

        netCtx.beginPath();
        netCtx.arc(n.x,n.y,2.5,0,Math.PI*2);
        netCtx.fillStyle="rgba(99,102,241,0.6)";
        netCtx.fill();
    });

    requestAnimationFrame(animateNetwork);
}
animateNetwork();
</script>
""", height=0)

# =========================================================
# INTERACTIVE IMMUNE BACKGROUND SECTION (700px)
# =========================================================
components.html("""
<canvas id="immune-bg" style="width:100%; height:700px;"></canvas>

<script>
const canvas1 = document.getElementById("immune-bg");
const ctx1 = canvas1.getContext("2d");

canvas1.width = window.innerWidth;
canvas1.height = 700;

const cells1 = [];

for(let i=0;i<12;i++){
    cells1.push({
        x:Math.random()*canvas1.width,
        y:Math.random()*canvas1.height,
        r:50+Math.random()*20,
        pulse:Math.random()*Math.PI
    });
}

function draw1(){
    ctx1.clearRect(0,0,canvas1.width,canvas1.height);

    cells1.forEach(c=>{
        c.pulse+=0.02;

        let gradient = ctx1.createRadialGradient(
            c.x,c.y,c.r*0.3,
            c.x,c.y,c.r
        );

        gradient.addColorStop(0,"rgba(99,102,241,0.6)");
        gradient.addColorStop(1,"rgba(147,51,234,0.05)");

        ctx1.beginPath();
        ctx1.arc(c.x,c.y,c.r,0,Math.PI*2);
        ctx1.fillStyle=gradient;
        ctx1.fill();
    });

    requestAnimationFrame(draw1);
}
draw1();
</script>
""", height=700)

# =========================================================
# HERO SECTION (NO DUPLICATE CANVAS ID)
# =========================================================
components.html("""
<div style="
position:relative;
width:100%;
height:720px;
overflow:hidden;
">

<canvas id="immune-hero" style="
position:absolute;
top:0;
left:0;
width:100%;
height:100%;
z-index:-1;"></canvas>

<div style="
position:absolute;
top:50%;
left:50%;
transform:translate(-50%,-50%);
text-align:center;
z-index:2;">

<h1 style="
font-size:72px;
background:linear-gradient(90deg,#3b82f6,#9333ea,#06b6d4);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;">
HPV–EPIPRED AI
</h1>

<p style="font-size:20px;color:#334155;">
AI-Driven MHC Class I Epitope Intelligence Platform
</p>

<a href="#scanner" style="color:#2563eb;font-weight:600;">
↓ Launch Scanner
</a>

</div>
</div>

<script>
const canvas2 = document.getElementById("immune-hero");
const ctx2 = canvas2.getContext("2d");

canvas2.width = window.innerWidth;
canvas2.height = 720;

const cells2 = [];

for(let i=0;i<10;i++){
    cells2.push({
        x:Math.random()*canvas2.width,
        y:Math.random()*canvas2.height,
        r:60+Math.random()*20,
        pulse:Math.random()*Math.PI
    });
}

function draw2(){
    ctx2.clearRect(0,0,canvas2.width,canvas2.height);

    cells2.forEach(c=>{
        c.pulse+=0.02;

        let gradient = ctx2.createRadialGradient(
            c.x,c.y,c.r*0.2,
            c.x,c.y,c.r
        );

        gradient.addColorStop(0,"rgba(139,92,246,0.6)");
        gradient.addColorStop(1,"rgba(139,92,246,0.05)");

        ctx2.beginPath();
        ctx2.arc(c.x,c.y,c.r,0,Math.PI*2);
        ctx2.fillStyle=gradient;
        ctx2.fill();
    });

    requestAnimationFrame(draw2);
}
draw2();
</script>
""", height=720)

# =========================================================
# MODEL + SCANNER (UNCHANGED LOGIC)
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

st.markdown("""
---
© 2026 HPV–EPIPRED AI  
Developed by Shamroz  
""")
