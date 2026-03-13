import streamlit as st
st.set_page_config(page_title="HPV EPIPRED", page_icon="🧬", layout="wide")

st.markdown("""
<style>

/* INPUT BOX */

textarea{
    background-color: {"#1e293b" if theme=="dark" else "#ffffff"};
    color: {"#e2e8f0" if theme=="dark" else "#1e293b"};
}

/* DATA TABLE */

div[data-testid="stDataFrame"]{
    background: {"rgba(255,255,255,0.04)" if theme=="dark" else "#f1f5f9"};
}

/* PLOT CONTAINER */

.stPlotlyChart{
    background: {"rgba(255,255,255,0.04)" if theme=="dark" else "#ffffff"};
}

/* ======================================================
IMPORT PROFESSIONAL FONTS
====================================================== */

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');


/* ======================================================
GLOBAL STYLE
====================================================== */

html, body, [data-testid="stAppViewContainer"]{
    font-family: 'Inter', sans-serif;
    background-color: {"#0f172a" if theme=="dark" else "#ffffff"};
    color: {"#e2e8f0" if theme=="dark" else "#1e293b"};
    line-height:1.6;
}


/* ======================================================
CENTERED RESEARCH LAYOUT
====================================================== */

.main .block-container{
    max-width:1200px;
    padding-top:2rem;
    padding-bottom:2rem;
}

[data-testid="stVerticalBlock"]{
    gap:16px;
}


/* ======================================================
HEADINGS
====================================================== */

h1,h2,h3{
    font-family:'Sora', sans-serif;
    font-weight:700;
}

h1{font-size:44px;}
h2{font-size:32px;}
h3{font-size:22px;}


/* ======================================================
BUTTONS
====================================================== */

.stButton > button{
    font-family:'Inter', sans-serif;
    font-weight:600;
    border-radius:12px;
    padding:8px 22px;
    background:#6366f1;
    color:white;
    border:none;
    transition:all .2s ease;
}

.stButton > button:hover{
    transform:translateY(-2px);
    box-shadow:0 6px 20px rgba(99,102,241,0.35);
}


/* ======================================================
DOWNLOAD BUTTON
====================================================== */

.stDownloadButton > button{
    border-radius:12px;
}


/* ======================================================
TABS
====================================================== */

div[data-baseweb="tab-list"]{
    gap:22px;
}

button[data-baseweb="tab"]{
    font-family:'Inter', sans-serif;
    font-size:16px;
    font-weight:600;
}

/* ======================================================
INPUTS & FASTA SEQUENCES
====================================================== */

textarea, input{
    font-family:'JetBrains Mono', monospace;
    font-size:15px;
    letter-spacing:1px;
    border-radius:12px;
}


/* ======================================================
DATA TABLES
====================================================== */

div[data-testid="stDataFrame"]{
    font-family:'IBM Plex Sans', sans-serif;
    border-radius:14px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.05);
}


/* Table header */

div[data-testid="stDataFrame"] thead th{
    font-weight:700;
}


/* Row hover */

div[data-testid="stDataFrame"] tbody tr:hover{
    background:rgba(255,255,255,0.05);
}


/* Highlight probability column */

div[data-testid="stDataFrame"] div[role="gridcell"]:nth-child(3){
    color:#818cf8;
    font-weight:700;
}


/* ======================================================
METRICS
====================================================== */

[data-testid="metric-container"]{
    background:rgba(255,255,255,0.03);
    border-radius:14px;
    padding:14px;
}


/* ======================================================
PLOT CONTAINERS
====================================================== */

.stPlotlyChart{
    background:rgba(255,255,255,0.03);
    border-radius:14px;
    padding:10px;
}


/* ======================================================
SCROLLBAR
====================================================== */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#6366f1;
    border-radius:10px;
}

/* ======================================================
REMOVE STREAMLIT HEADER
====================================================== */

header{
    visibility:hidden;
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
from sklearn.cluster import KMeans
import networkx as nx
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
.hero-footer{
    position:absolute;
    bottom:20px;
    width:100%;
    text-align:center;
    font-size:20px;
    color:#94a3b8;
    letter-spacing:1px;
}

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
    font-size: clamp(64px,8vw,115px);
    font-family: 'Sora', sans-serif;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size:40px;
    color:#cbd5e1;
}
.cta {
    margin-top: 60px;          /* pushes it lower */
    font-size: 20px;
    color: #38bdf8;
    text-decoration: none;
    display: inline-block;
    font-weight: 500;
}

/* Arrow animation */
.arrow {
    display: inline-block;
    animation: bounce 1.6s infinite;
}

/* Bounce effect */
@keyframes bounce {
    0%,100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(8px);
    }
}
</style>

<div class="hero">

    <canvas id="immune"></canvas>
    <canvas id="network"></canvas>

    <div class="hero-content">
        <div class="hero-title">HPV EPIPRED</div>
        <div class="hero-sub">MHC I Epitope Prediction</div>

        <a href="#scanner" class="cta">
            <span class="arrow">↓</span> Launch Scanner
        </a>

    </div>
    
    <div class="hero-footer">
    © 2026 HPV EPIPRED • Developed by <b>Shamroz Abrar</b>
    
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
@st.cache_resource
def load_model():
    return joblib.load("hpv_epitope_model.pkl")
model = load_model()
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

st.markdown("""
<style>

/* RESULT SECTION BACKGROUND */
.result-bg {
    background:
        radial-gradient(circle at 20% 30%, rgba(56,189,248,0.15), transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(168,85,247,0.15), transparent 40%),
        linear-gradient(135deg, #020617 0%, #0f172a 60%, #020617 100%);
    padding: 30px;
    border-radius: 18px;
}

/* TABLE CONTAINER */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    backdrop-filter: blur(6px);
}

/* PLOT CONTAINER */
.stPlotlyChart {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.legend-box {
    background: linear-gradient(135deg,#f8fafc,#eef2ff);
    border-left: 6px solid #6366f1;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 15px;
    font-size: 14px;
}

.legend-title {
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 8px;
}

.legend-item {
    margin-left: 6px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SCANNER SECTION (FULL RESTORED VERSION)
# =========================================================
st.markdown('<div id="scanner"></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔬 AI Scanner", "🧠 Model Explainability"])

# ==========================
# AI SCANNER TAB
# ==========================
with tab1:

    mode = st.radio("Mode", ["Single Sequence","Batch Upload"])
    fasta = ""

    if mode == "Single Sequence":
        fasta = st.text_area("Paste FASTA Sequence")
    else:
        uploaded = st.file_uploader("Upload FASTA File")
        if uploaded:
            fasta = uploaded.read().decode()

    # BUTTON MUST BE INSIDE TAB
    run_scan = st.button("Run AI Scan")

    if run_scan and fasta:

        # SEQUENCE CLEANING
        seq = "".join([
            l.strip() for l in fasta.split("\n")
            if not l.startswith(">")
        ]).upper()

        peptides = []
        positions = []

        for i in range(len(seq)-8):
            peptides.append(seq[i:i+9])
            positions.append(i+1)

        X = np.array([extract_features(p) for p in peptides])
        probs = model.predict_proba(X)[:,1]

        results = []
        for pos, pep, prob in zip(positions, peptides, probs):
            cat = "Epitope" if prob >= threshold else "Non-Epitope"
            results.append([pos, pep, prob, cat])

        df = pd.DataFrame(
            results,
            columns=["Position","Peptide","Probability","Category"]
        )

        st.session_state["df"] = df

       
        # ==========================
        # SPLIT TABLES
        # ==========================
        epitope_df = df[df["Category"] == "Epitope"].sort_values(
            by="Probability", ascending=False
        )

        non_df = df[df["Category"] == "Non-Epitope"].sort_values(
            by="Probability", ascending=False
        )

        # ==========================
        # RESULT TABS
        # ==========================
        tab_table, tab_prob, tab_landscape, tab_density, tab_fingerprint, tab_score, tab_atlas, tab_competition = st.tabs([
                "📊 Tables",
                "📈 Probability Plot",
                "🌍 Epitope Landscape",
                "🧬 Epitope Density Map",
                "🌐 Immunogenicity Fingerprint",
                "🧬 Immunogenic Score",
                "🧭 Epitope Atlas",
                "🔥 Epitope Competition Map"
        ])

        # ==========================
        # TABLE TAB
        # ==========================
        with tab_table:

                st.markdown("### 🟢 Predicted Epitopes")

                st.markdown("""
        <div class="legend-box">

        <div class="legend-title">📊 Table Description</div>

        <div class="legend-item">📍 <b>Position</b> – Starting index of the peptide in the protein sequence</div>

        <div class="legend-item">🧬 <b>Peptide</b> – Extracted 9-mer amino acid window</div>

        <div class="legend-item">📈 <b>Probability</b> – Machine learning predicted epitope likelihood</div>

        <div class="legend-item">🏷 <b>Category</b> – Epitope classification based on prediction threshold</div>

        </div>
        """, unsafe_allow_html=True)
            

                if not epitope_df.empty:

                        epi_show = epitope_df.copy()
                        epi_show["Probability"] = epi_show["Probability"].round(3)

                        st.dataframe(
                                epi_show,
                                use_container_width=True,
                                hide_index=True,
                                height=350
                        )

                else:
                        st.info("No epitopes detected above threshold.")

                st.markdown("---")

                st.markdown("### ⚪ Predicted Non-Epitopes")

                if not non_df.empty:

                        non_show = non_df.copy()
                        non_show["Probability"] = non_show["Probability"].round(3)

                        st.dataframe(
                                non_show,
                                use_container_width=True,
                                height=350,
                                hide_index=True
                        )

                else:
                        st.info("All peptides classified as epitopes.")

                st.markdown("---")

                st.markdown("### 🏆 Top 10 High-Confidence Epitopes")

                top10 = epitope_df.head(10).copy()
                top10.insert(0, "Rank", range(1, len(top10)+1))
                top10["Length"] = top10["Peptide"].apply(len)

                st.dataframe(
                        top10[["Rank","Position","Peptide","Length","Probability"]],
                        use_container_width=True,
                        hide_index=True
                )


        # ==========================
        # PROBABILITY PLOT TAB
        # ==========================
        with tab_prob:

                st.markdown("### 📈 Epitope Probability Across Protein Sequence")
                
                st.markdown("""
        <div class="legend-box">

        <div class="legend-title">📈 Plot Interpretation</div>

        <div class="legend-item">🔵 <b>Light Blue Line</b> – Raw prediction probability for each peptide</div>

        <div class="legend-item">🔷 <b>Dark Blue Line</b> – Smoothed immunogenic signal across sequence</div>

        <div class="legend-item">🚨 <b>Red Dashed Line</b> – Threshold used to classify epitopes</div>

        <div class="legend-item">⛰ <b>Probability Peaks</b> – Regions with strong predicted immune recognition</div>

        </div>
        """, unsafe_allow_html=True)

                window = 12
                smooth_prob = np.convolve(
                        df["Probability"],
                        np.ones(window)/window,
                        mode="same"
                )

                fig = go.Figure()

                fig.add_trace(
                        go.Scatter(
                                x=df["Position"],
                                y=df["Probability"],
                                mode="lines",
                                line=dict(color="rgba(59,130,246,0.3)", width=1),
                                name="Raw Prediction"
                        )
                )

                fig.add_trace(
                        go.Scatter(
                                x=df["Position"],
                                y=smooth_prob,
                                mode="lines",
                                line=dict(color="#1d4ed8", width=3),
                                name="Smoothed Signal"
                        )
                )

                fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Epitope Threshold"
                )

                fig.update_layout(
                        height=500,
                        xaxis_title="Protein Position",
                        yaxis_title="Epitope Probability",
                        hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)


        # ==========================
        # EPITOPE LANDSCAPE TAB
        # ==========================

        with tab_landscape:

                st.markdown("### 🌍 Epitope Immunogenic Landscape")

                st.markdown("""
                <div class="legend-box">

                <div class="legend-title">🌍 Landscape Interpretation</div>

                <div class="legend-item">📏 <b>X-axis</b> – Peptide position along the protein</div>

                <div class="legend-item">📊 <b>Y-axis</b> – Predicted epitope probability</div>

                <div class="legend-item">🎨 <b>Point Colors</b> – Cluster assignment of peptides</div>

                <div class="legend-item">✖ <b>Black Cross</b> – Cluster center</div>

                </div>
                """, unsafe_allow_html=True)

                X_cluster = df[["Position","Probability"]]

                kmeans = KMeans(n_clusters=4, random_state=42)

                df["Cluster"] = kmeans.fit_predict(X_cluster)

                centers = kmeans.cluster_centers_

                fig_land = px.scatter(
                        df,
                        x="Position",
                        y="Probability",
                        color="Cluster",
                        hover_data=["Peptide"],
                        color_continuous_scale="viridis"
                )

                fig_land.add_trace(
                        go.Scatter(
                                x=centers[:,0],
                                y=centers[:,1],
                                mode="markers",
                                marker=dict(color="black",size=12,symbol="x"),
                                name="Cluster Center"
                        )
                )

                st.plotly_chart(fig_land, use_container_width=True)

        # ==========================
        # EPITOPE DENSITY MAP TAB
        # ==========================
        with tab_density:

                st.markdown("### 🧬 Epitope Density Map")
                    
                st.markdown("""
        <div class="legend-box">

        <div class="legend-title">🧬 Density Map Guide</div>

        <div class="legend-item">📦 <b>Bars</b> – Sliding window regions across the protein</div>

        <div class="legend-item">📊 <b>Bar Height</b> – Fraction of predicted epitopes within that region</div>

        <div class="legend-item">🔥 <b>High Density</b> – Indicates potential immunogenic hotspot regions</div>

        </div>
        """, unsafe_allow_html=True)

                window = 15
                density = []

                for i in range(len(df)):

                        start = max(0, i-window)
                        end = min(len(df), i+window)

                        region = df.iloc[start:end]
                        ep_count = (region["Probability"] >= threshold).sum()

                        density.append(ep_count/len(region))

                density_df = pd.DataFrame({
                        "Position":df["Position"],
                        "Density":density
                })

                fig_density = go.Figure()

                fig_density.add_trace(
                        go.Bar(
                                x=density_df["Position"],
                                y=density_df["Density"],
                                marker_color="#6366f1"
                        )
                )

                fig_density.update_layout(
                        height=350,
                        xaxis_title="Protein Position",
                        yaxis_title="Epitope Density"
                )

                st.plotly_chart(fig_density,use_container_width=True)

       
        # IMMUNOGENICITY FINGERPRINT
        # ==========================
        with tab_fingerprint:

                st.markdown("### 🧬 Protein Immunogenicity Fingerprint")

                st.markdown("""
        <div class="legend-box">

        <div class="legend-title">🧬 Fingerprint Metrics</div>

        <div class="legend-item">🤖 <b>ML Immunogenicity</b> – Average predicted epitope probability</div>

        <div class="legend-item">📍 <b>Epitope Density</b> – Fraction of peptides classified as epitopes</div>

        <div class="legend-item">🧪 <b>Hydrophobicity</b> – Fraction of hydrophobic amino acids</div>

        <div class="legend-item">🧠 <b>Entropy</b> – Sequence diversity across peptides</div>

        <div class="legend-item">⚡ <b>Net Charge</b> – Balance of positive and negative residues</div>

        </div>
        """, unsafe_allow_html=True)

                # Global metrics
                mean_prob = df["Probability"].mean()

                epitope_density = len(df[df["Category"]=="Epitope"]) / len(df)

                hydro_score = np.mean([
                        extract_features(p)[-7]
                        for p in df["Peptide"]
                ])

                entropy_score = np.mean([
                        extract_features(p)[-2]
                        for p in df["Peptide"]
                ])

                charge_score = np.mean([
                        extract_features(p)[-3]
                        for p in df["Peptide"]
                ])

                metrics = {
                        "ML Immunogenicity": mean_prob,
                        "Epitope Density": epitope_density,
                        "Hydrophobicity": hydro_score,
                        "Entropy": entropy_score,
                        "Net Charge": abs(charge_score)
                }

                categories = list(metrics.keys())
                values = list(metrics.values())

                fig_radar = go.Figure()

                fig_radar.add_trace(
                        go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill="toself",
                                line=dict(color="#6366f1", width=3),
                                name="Protein Profile"
                        )
                )

                fig_radar.update_layout(
                        polar=dict(
                                radialaxis=dict(
                                        visible=True,
                                        range=[0,1]
                                )
                        ),
                        height=500
                )

                st.plotly_chart(fig_radar, use_container_width=True)
            
        # ==========================
        # IMMUNOGENIC SCORE TAB
        # ==========================
        with tab_score:

                st.markdown("### 🧬 Global Immunogenic Score")

                st.markdown("""
        <div class="legend-box">

        <div class="legend-title">🧬 Score Explanation</div>

        <div class="legend-item">🎯 <b>Gauge Value</b> – Mean predicted epitope probability</div>

        <div class="legend-item">🧬 <b>Total Peptides</b> – Number of peptide windows analyzed</div>

        <div class="legend-item">🔥 <b>Predicted Epitopes</b> – Peptides above classification threshold</div>

        <div class="legend-item">📊 <b>Epitope Density</b> – Percentage of peptides predicted as epitopes</div>

        </div>
        """, unsafe_allow_html=True)
                mean_prob = df["Probability"].mean()
                total_pep = len(df)
                epi_count = len(df[df["Category"]=="Epitope"])

                density_score = epi_count/total_pep

                gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=mean_prob,
                        number={'valueformat':".2f"},
                        title={'text':"Global Immunogenic Score"},
                        gauge={
                                'axis':{'range':[0,1]},
                                'bar':{'color':"#38bdf8"}
                        }
                ))

                st.plotly_chart(gauge,use_container_width=True)

                st.metric("Peptides scanned",total_pep)
                st.metric("Predicted epitopes",epi_count)
                st.metric("Epitope density",f"{density_score:.2%}")

        # ==========================
        # STACKED PROTEIN EPITOPE ATLAS
        # ==========================

        with tab_atlas:

                st.markdown("### 🧭 Stacked Protein Epitope Atlas")

                st.markdown("""
        <div class="legend-box">

        <div class="legend-title">🧭 Atlas Visualization Guide</div>

        <div class="legend-item">🔴 <b>Red Circles</b> – Predicted epitope peptides</div>

        <div class="legend-item">⚪ <b>Grey Circles</b> – Non-epitope peptide windows</div>

        <div class="legend-item">⭐ <b>Gold Stars</b> – Top high-confidence epitopes</div>

        <div class="legend-item">📏 <b>X-axis</b> – Protein sequence position</div>

        <div class="legend-item">📚 <b>Y-axis Tracks</b> – Stacked peptide windows to show overlaps</div>

        <div class="legend-item">📦 <b>Bubble Size</b> – Epitope prediction probability</div>

        </div>
        """, unsafe_allow_html=True)

                atlas_df = df.copy()

                # assign vertical tracks so overlapping peptides stack
                tracks = []
                current_track = 0

                for i in range(len(atlas_df)):
                        tracks.append(current_track)
                        current_track = (current_track + 1) % 6

                atlas_df["Track"] = tracks

                atlas_df["Color"] = atlas_df["Category"].map({
                        "Epitope": "red",
                        "Non-Epitope": "lightgray"
                })

                atlas_df["Size"] = atlas_df["Probability"] * 20 + 4

                fig_atlas = go.Figure()

                fig_atlas.add_trace(
                        go.Scatter(
                                x=atlas_df["Position"],
                                y=atlas_df["Track"],
                                mode="markers",
                                marker=dict(
                                        size=atlas_df["Size"],
                                        color=atlas_df["Color"],
                                        opacity=0.85
                                ),
                                hovertext=atlas_df["Peptide"],
                                name="Peptide Windows"
                        )
                )

                # highlight top epitopes
                top = atlas_df.nlargest(10, "Probability")

                fig_atlas.add_trace(
                        go.Scatter(
                                x=top["Position"],
                                y=top["Track"],
                                mode="markers",
                                marker=dict(
                                        size=18,
                                        color="gold",
                                        symbol="star"
                                ),
                                hovertext=top["Peptide"],
                                name="Top Epitopes"
                        )
                )

                fig_atlas.update_layout(
                        height=420,
                        xaxis_title="Protein Position",
                        yaxis_title="Peptide Track",
                        title="Stacked Epitope Atlas"
                )

                st.plotly_chart(fig_atlas, use_container_width=True)
        
        # ==========================
        # EPITOPE COMPETITION HEATMAP
        # ==========================

        with tab_competition:

                st.markdown("### ⚔️ Epitope Competition Heatmap")

                st.markdown("""
        <div class="legend-box">

        <div class="legend-title">⚔️ Epitope Competition Map Guide</div>

        <div class="legend-item">🔴 <b>Nodes</b> – Predicted epitope peptides</div>

        <div class="legend-item">📏 <b>Node Size</b> – Epitope prediction probability</div>

        <div class="legend-item">🔗 <b>Edges</b> – Overlapping peptides competing for MHC-I presentation</div>

        <div class="legend-item">🔥 <b>Dense Networks</b> – Regions where multiple epitopes overlap and may compete</div>

        <div class="legend-item">📍 <b>Isolated Nodes</b> – Unique epitopes with minimal overlap</div>

        <div class="legend-item">🧬 <b>Biological Meaning</b> – Highly connected peptides may dominate antigen presentation</div>

        </div>
        """, unsafe_allow_html=True)

                window = 10
                competition_scores = []

                for i in range(len(df)):

                        start = max(0, i-window)
                        end = min(len(df), i+window)

                        region = df.iloc[start:end]

                        competition_scores.append(region["Probability"].mean())

                comp_df = pd.DataFrame({
                        "Position": df["Position"],
                        "Competition_Score": competition_scores
                })

                # reshape for heatmap
                heat = comp_df["Competition_Score"].values.reshape(1,-1)

                fig = px.imshow(
                        heat,
                        aspect="auto",
                        color_continuous_scale="RdYlBu_r",
                        labels=dict(color="Competition Score")
                )

                fig.update_layout(
                        height=200,
                        xaxis_title="Protein Position",
                        yaxis=dict(showticklabels=False),
                        title="Epitope Competition Across Protein Sequence"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                st.markdown("### 🏆 Dominant Immunogenic Regions")

                region_df = comp_df.sort_values(
                        by="Competition_Score",
                        ascending=False
                ).head(10)

                region_df.insert(0,"Rank",range(1,len(region_df)+1))

                st.dataframe(
                        region_df,
                        use_container_width=True,
                        hide_index=True
                )
            
# ==========================
# FEATURE NAMES FOR EXPLAINABILITY
# ==========================

pos_features = []

for pos in range(1,10):
    for aa in aa_list:
        pos_features.append(f"Position{pos}_{aa}")

di_features = [f"Dipeptide_{dp}" for dp in dipeptides]

bio_features = [
    "Hydrophobicity",
    "Aromaticity",
    "Positive_Charge",
    "Negative_Charge",
    "Net_Charge",
    "Entropy",
    "Avg_Molecular_Weight"
]

feature_names = pos_features + di_features + bio_features


with tab2:

        st.markdown("### 📊 Model Feature Importance")

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(20)

        fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Sequence Features Influencing MHC-I Epitope Prediction"
        )

        st.plotly_chart(fig, use_container_width=True)
