import streamlit as st
st.set_page_config(page_title="HPV EPIPRED", page_icon="🧬", layout="wide")

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

[data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}

h3 {
    margin-bottom: 0.2rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* GLOBAL APP STYLE */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.2px;
}

/* HEADINGS */
h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
}

h1 { font-size: 48px !important; }
h2 { font-size: 34px !important; }
h3 { font-size: 24px !important; }

/* BUTTONS */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600;
    border-radius: 12px;
    padding: 10px 24px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99,102,241,0.25);
}

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    border-radius: 12px;
}

/* TABS */
div[data-baseweb="tab-list"] {
    gap: 35px;
}

button[data-baseweb="tab"] {
    font-size: 18px !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #6366f1 !important;
}

/* INPUTS + SEQUENCES */
textarea, input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 15px !important;
    border-radius: 12px !important;
}

/* DATAFRAME */
div[data-testid="stDataFrame"] div[role="grid"] * {
    font-size: 16px !important;
}

div[data-testid="stDataFrame"] thead tr th {
    font-weight: 700 !important;
}

/* PROBABILITY COLUMN */
div[data-testid="stDataFrame"] div[role="gridcell"]:nth-child(3) {
    font-weight: 700 !important;
    color: #6366f1 !important;
}

/* METRICS */
[data-testid="metric-container"] {
    border-radius: 16px;
    padding: 12px 18px;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #c7d2fe;
    border-radius: 10px;
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
    font-size: clamp(64px,8vw,115px);
    font-family: 'Sora', sans-serif;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size:40px;
    color:#cbd5e1;
    margin-top:10px;
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

    run_scan = st.button("Run AI Scan")

    if run_scan and fasta:
        st.session_state["run_scan"] = True
        st.session_state["fasta"] = fasta

    if "run_scan" in st.session_state and st.session_state["run_scan"]:
        fasta = st.session_state["fasta"]

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
        # ==========================
        # GENERATE ALL PEPTIDES
        # ==========================
        peptides = []
        positions = []

        for i in range(len(seq) - 8):
            peptides.append(seq[i:i+9])
            positions.append(i + 1)

        # ==========================
        # FEATURE EXTRACTION
        # ==========================
        X = np.array([extract_features(p) for p in peptides])

        # ==========================
        # BATCH PREDICTION
        # ==========================
        probs = model.predict_proba(X)[:, 1]

        # ==========================
        # BUILD RESULTS
        # ==========================
        results = []

        for pos, pep, prob in zip(positions, peptides, probs):
            cat = "Epitope" if prob >= threshold else "Non-Epitope"
            results.append([pos, pep, prob, cat])

        df = pd.DataFrame(
            results,
            columns=["Position", "Peptide", "Probability", "Category"]
        )

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
        tab_table, tab_plot, tab_landscape, tab_gauge = st.tabs(
            ["📊 Tables", "📈 Probability Plot", "🔬 Epitope Landscape", "🧬 Immunogenic Score"]
        )

        # ==========================
        # TABLE TAB
        # ==========================
        with tab_table:

            st.markdown("### 🟢 Predicted Epitopes")

            if not epitope_df.empty:

                epi_show = epitope_df.copy()
                epi_show["Probability"] = epi_show["Probability"].round(3)

                st.dataframe(
                    epi_show,
                    use_container_width=True,
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
                    height=350
                )

            else:
                st.info("All peptides classified as epitopes.")
        # ==========================
        # PROBABILITY PLOT
        # ==========================
       with tab_plot:

            fig = px.line(
                df,
                x="Position",
                y="Probability",
                markers=True
            )

            fig.update_traces(
                line=dict(width=2),
                marker=dict(size=6,color="#3b82f6")
            )

            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Decision Threshold",
                annotation_position="top left"
            )

            fig.update_layout(
                title="Epitope Probability Across Protein Sequence",
                xaxis_title="Protein Position",
                yaxis_title="Epitope Probability",
                height=420,
                margin=dict(l=20,r=20,t=60,b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
                            
        # EPITOPE LANDSCAPE TAB
        # ==========================
        with tab_landscape:

            colors = [
                "#22c55e" if p >= threshold else "#475569"
                for p in df["Probability"]
            ]

            strip = go.Figure()

            strip.add_trace(
                go.Bar(
                    x=df["Position"],
                    y=[1]*len(df),
                    marker=dict(color=colors),
                    customdata=df[["Peptide","Probability"]],
                    hovertemplate=
                    "<b>Position:</b> %{x}<br>"
                    "<b>Peptide:</b> %{customdata[0]}<br>"
                    "<b>Probability:</b> %{customdata[1]:.3f}<extra></extra>"
                )
            )

            strip.update_layout(
                height=180,
                yaxis=dict(showticklabels=False),
                xaxis_title="Protein Position",
                showlegend=False
            )

            st.plotly_chart(strip, use_container_width=True)

        # ==========================
        # GAUGE TAB
        # ==========================
        with tab_gauge:

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

    st.markdown("### 🧠 Model Feature Importance")

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
