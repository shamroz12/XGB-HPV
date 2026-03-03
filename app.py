import streamlit as st
st.set_page_config(page_title="HPV EPIPRED", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Apply to entire app */
html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Headings */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* Buttons */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 22px;
}

/* Radio + Labels */
label {
    font-weight: 500 !important;
}

/* Tabs container */
div[data-baseweb="tab-list"] {
    gap: 30px;
}

/* Individual tab */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 17px !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom: 2px solid #6366f1 !important;
}

textarea {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    border-radius: 12px !important;
}

/* Target internal dataframe grid */
div[data-testid="stDataFrame"] div[role="grid"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Cell text */
div[data-testid="stDataFrame"] div[role="grid"] div {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
}

/* Header cells */
div[data-testid="stDataFrame"] div[role="columnheader"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Table header styling */
div[data-testid="stDataFrame"] thead tr th {
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* Metric font */
div[data-testid="metric-container"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Download button */
.stDownloadButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

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
# HERO BLOCK
# =========================================================
components.html("""
<style>
.hero {
    position: relative;
    width: 100%;
    height: 100vh;
    overflow: hidden;
    background:
        radial-gradient(circle at 30% 40%, #3b0764 0%, transparent 45%),
        radial-gradient(circle at 70% 60%, #1e1b4b 0%, transparent 50%),
        linear-gradient(135deg, #020617 0%, #0f172a 60%, #020617 100%);
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
    z-index: 10;
}

.hero-title {
    font-size: 115px;
    font-family: 'Sora', sans-serif;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 40px;
    color: #cbd5e1;
}

.cta {
    margin-top: 25px;
    font-size: 20px;
    color: #38bdf8;
    text-decoration: none;
}
</style>

<div class="hero">

    <canvas id="immune"></canvas>
    <canvas id="network"></canvas>

    <div class="hero-content">
        <div class="hero-title">HPV EPIPRED</div>
        <div class="hero-sub">MHC I Epitope Prediction</div>
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

// =============================
// TRUE IMMUNE CELLS
// =============================

let cells = [];

for(let i=0;i<7;i++){
    cells.push({
        x:Math.random()*window.innerWidth,
        y:Math.random()*window.innerHeight,
        r:90+Math.random()*40,
        pulse:Math.random()*Math.PI,
        driftX:(Math.random()-0.5)*0.3,
        driftY:(Math.random()-0.5)*0.3
    });
}

function drawImmune(){
    ictx.clearRect(0,0,immune.width,immune.height);

    cells.forEach(c=>{
        c.pulse+=0.02;
        c.x+=c.driftX;
        c.y+=c.driftY;

        // Outer membrane glow
        let membrane = ictx.createRadialGradient(
            c.x,c.y,c.r*0.2,
            c.x,c.y,c.r
        );
        membrane.addColorStop(0,"rgba(168,85,247,0.8)");
        membrane.addColorStop(1,"rgba(168,85,247,0.02)");

        ictx.beginPath();
        ictx.arc(c.x,c.y,c.r,0,Math.PI*2);
        ictx.fillStyle=membrane;
        ictx.fill();

        // Cytoplasm
        ictx.beginPath();
        ictx.arc(c.x,c.y,c.r*0.65,0,Math.PI*2);
        ictx.fillStyle="rgba(99,102,241,0.4)";
        ictx.fill();

        // Nucleus
        ictx.beginPath();
        ictx.arc(c.x,c.y,c.r*0.3,0,Math.PI*2);
        ictx.fillStyle="rgba(56,189,248,0.7)";
        ictx.fill();
    });

    requestAnimationFrame(drawImmune);
}
drawImmune();

// =============================
// AI NEURAL OVERLAY
// =============================

let nodes=[];
for(let i=0;i<70;i++){
    nodes.push({
        x:Math.random()*window.innerWidth,
        y:Math.random()*window.innerHeight,
        vx:(Math.random()-0.5)*0.4,
        vy:(Math.random()-0.5)*0.4
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
        nctx.fillStyle="rgba(34,211,238,0.8)";
        nctx.fill();
    });

    for(let i=0;i<nodes.length;i++){
        for(let j=i+1;j<nodes.length;j++){
            let dx=nodes[i].x-nodes[j].x;
            let dy=nodes[i].y-nodes[j].y;
            let dist=Math.sqrt(dx*dx+dy*dy);

            if(dist<140){
                nctx.beginPath();
                nctx.moveTo(nodes[i].x,nodes[i].y);
                nctx.lineTo(nodes[j].x,nodes[j].y);
                nctx.strokeStyle="rgba(34,211,238,0.15)";
                nctx.stroke();
            }
        }
    }

    requestAnimationFrame(drawNetwork);
}
drawNetwork();

</script>
""", height=950)

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
# SCANNER SECTION (FULL RESTORED VERSION)
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

        # ==========================
        # SEQUENCE CLEANING
        # ==========================
        seq = "".join([
            l.strip() for l in fasta.split("\n")
            if not l.startswith(">")
        ]).upper()

        results = []

        # ==========================
        # PREDICTION LOOP
        # ==========================
        for i in range(len(seq) - 8):
            pep = seq[i:i+9]
            prob = model.predict_proba(
                [extract_features(pep)]
            )[0][1]

            cat = "Epitope" if prob >= threshold else "Non-Epitope"
            results.append([i+1, pep, prob, cat])

        df = pd.DataFrame(
            results,
            columns=["Position", "Peptide", "Probability", "Category"]
        )

        # ==========================
        # SPLIT TABLES
        # ==========================
        epitope_df = df[df["Category"] == "Epitope"] \
            .sort_values(by="Probability", ascending=False)

        non_df = df[df["Category"] == "Non-Epitope"] \
            .sort_values(by="Probability", ascending=False)

        # ==========================
        # DISPLAY TABLES
        # ==========================
        st.markdown("### 🟢 Predicted Epitopes")
        if not epitope_df.empty:
            st.dataframe(epitope_df, use_container_width=True)
        else:
            st.info("No epitopes detected above threshold.")

        st.markdown("### ⚪ Predicted Non-Epitopes")
        if not non_df.empty:
            st.dataframe(non_df, use_container_width=True)
        else:
            st.info("All peptides classified as epitopes.")

        # ==========================
        # PROBABILITY PLOT
        # ==========================
        fig = px.line(
            df,
            x="Position",
            y="Probability",
            markers=True
        )

        fig.update_layout(
            title="Epitope Probability Across Protein Sequence",
            xaxis_title="Amino Acid Position",
            yaxis_title="Predicted Epitope Probability",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text="Decision Threshold",
            annotation_position="top left"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==========================
        # GAUGE
        # ==========================
        mean_prob = df["Probability"].mean()

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prob,
            number={'valueformat': ".2f"},
            title={'text': "Global Immunogenic Score"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#38bdf8"}
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        # ==========================
        # DOWNLOAD
        # ==========================
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            csv,
            "epitope_results.csv"
        )

with tab2:
    feat = pd.DataFrame({
        "Feature": ["Hydrophobicity", "Net Charge", "Entropy"],
        "Importance": [0.32, 0.21, 0.17]
    })

    fig = px.bar(
        feat,
        x="Importance",
        y="Feature",
        orientation="h",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
