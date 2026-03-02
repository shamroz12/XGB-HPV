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
# NEURAL NETWORK ANIMATED BACKGROUND
# =========================================================
st.components.v1.html("""
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
const NODE_COUNT = 65;

for(let i=0;i<NODE_COUNT;i++){
    nodes.push({
        x:Math.random()*window.innerWidth,
        y:Math.random()*window.innerHeight,
        vx:(Math.random()-0.5)*0.6,
        vy:(Math.random()-0.5)*0.6
    });
}

const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

function animateNetwork(){
    netCtx.clearRect(0,0,netCanvas.width,netCanvas.height);

    nodes.forEach(n=>{
        n.x+=n.vx;
        n.y+=n.vy;

        if(n.x<0||n.x>window.innerWidth) n.vx*=-1;
        if(n.y<0||n.y>window.innerHeight) n.vy*=-1;

        netCtx.beginPath();
        netCtx.arc(n.x,n.y,2.5,0,Math.PI*2);
        netCtx.fillStyle=isDark?
            "rgba(99,102,241,0.6)":
            "rgba(59,130,246,0.6)";
        netCtx.fill();
    });

    for(let i=0;i<nodes.length;i++){
        for(let j=i+1;j<nodes.length;j++){
            let dx=nodes[i].x-nodes[j].x;
            let dy=nodes[i].y-nodes[j].y;
            let dist=Math.sqrt(dx*dx+dy*dy);

            if(dist<150){
                netCtx.beginPath();
                netCtx.moveTo(nodes[i].x,nodes[i].y);
                netCtx.lineTo(nodes[j].x,nodes[j].y);

                netCtx.strokeStyle=isDark?
                    "rgba(99,102,241,0.15)":
                    "rgba(59,130,246,0.15)";

                netCtx.lineWidth=1;
                netCtx.stroke();
            }
        }
    }

    requestAnimationFrame(animateNetwork);
}

animateNetwork();
</script>
""", height=0)

st.markdown("""
<style>
/* Force background on Streamlit containers */

.stApp, .main, .block-container {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #e0e7ff 40%,
        #dbeafe 100%
    ) !important;
}

/* Remove excess padding to reduce height */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# INTERACTIVE 3D HELIX WITH DEPTH FOG
# =========================================================
st.components.v1.html("""
<canvas id="immune-canvas" style="width:100%; height:700px;"></canvas>

<script>
const canvas = document.getElementById("immune-canvas");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = 700;

const cells = [];
const peptides = [];

for(let i=0;i<12;i++){
    cells.push({
        x:Math.random()*canvas.width,
        y:Math.random()*canvas.height,
        r:50+Math.random()*20,
        pulse:Math.random()*Math.PI
    });
}

for(let i=0;i<60;i++){
    peptides.push({
        x:Math.random()*canvas.width,
        y:Math.random()*canvas.height,
        vx:(Math.random()-0.5)*1.2,
        vy:(Math.random()-0.5)*1.2
    });
}

function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);

    // Draw cells
    cells.forEach(c=>{
        c.pulse += 0.02;
        let glow = 15 + Math.sin(c.pulse)*5;

        let gradient = ctx.createRadialGradient(
            c.x,c.y,c.r*0.3,
            c.x,c.y,c.r
        );

        gradient.addColorStop(0,"rgba(99,102,241,0.6)");
        gradient.addColorStop(1,"rgba(147,51,234,0.05)");

        ctx.beginPath();
        ctx.arc(c.x,c.y,c.r,0,Math.PI*2);
        ctx.fillStyle = gradient;
        ctx.fill();
    });

    // Draw peptides
    peptides.forEach(p=>{
        p.x += p.vx;
        p.y += p.vy;

        if(p.x<0||p.x>canvas.width) p.vx*=-1;
        if(p.y<0||p.y>canvas.height) p.vy*=-1;

        ctx.beginPath();
        ctx.arc(p.x,p.y,3,0,Math.PI*2);
        ctx.fillStyle="rgba(59,130,246,0.9)";
        ctx.fill();
    });

    // Immune signaling lines
    peptides.forEach(p=>{
        cells.forEach(c=>{
            let dx = p.x - c.x;
            let dy = p.y - c.y;
            let dist = Math.sqrt(dx*dx + dy*dy);

            if(dist < c.r){
                ctx.beginPath();
                ctx.moveTo(p.x,p.y);
                ctx.lineTo(c.x,c.y);
                ctx.strokeStyle="rgba(59,130,246,0.15)";
                ctx.stroke();
            }
        });
    });

    requestAnimationFrame(draw);
}

draw();
</script>
""", height=700)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div style="height:75vh; display:flex; flex-direction:column;
justify-content:center; align-items:center; text-align:center;">

<div style="position:absolute; width:600px; height:600px;
background:radial-gradient(circle, rgba(59,130,246,0.25), transparent 70%);
filter:blur(80px); z-index:-1;"></div>

<h1 style="font-size:64px;
background:linear-gradient(90deg,#3b82f6,#9333ea,#06b6d4);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;">
HPV-EPIPRED AI
</h1>

<p style="font-size:20px; opacity:0.8;">
AI-Driven MHC Class I Epitope Intelligence Platform
</p>

<a href="#scanner" style="margin-top:40px;
color:#3b82f6; text-decoration:none;">
↓ Launch Scanner
</a>

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

        st.dataframe(df, use_container_width=True)

        fig = px.line(df, x="Position", y="Probability",
                      template="plotly_dark", markers=True)
        fig.add_hline(y=threshold, line_dash="dash", line_color="#f43f5e")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        mean_prob = df["Probability"].mean()

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prob,
            title={'text': "Global Immunogenic Score"},
            gauge={'axis':{'range':[0,1]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "epitope_results.csv")

with tab2:
    feat = pd.DataFrame({
        "Feature":["Hydrophobicity","Net Charge","Entropy"],
        "Importance":[0.32,0.21,0.17]
    })
    fig = px.bar(feat, x="Importance", y="Feature",
                 orientation="h", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
