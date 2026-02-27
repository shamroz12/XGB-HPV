import streamlit as st

# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =====================================================
st.set_page_config(
    page_title="HPV-EPIPRED",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from itertools import product
from collections import Counter
import math

# =====================================================
# THEME STATE (Stable Toggle)
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = (
        "light" if st.session_state.theme == "dark" else "dark"
    )

st.sidebar.button("🌗 Toggle Dark / Light Mode", on_click=toggle_theme)

theme = st.session_state.theme

# =====================================================
# GLOBAL CSS (Glassmorphism + Layout)
# =====================================================
st.markdown(f"""
<style>

.stApp {{
    background-color: {"#0f172a" if theme=="dark" else "#f4f7fb"};
}}

.glass {{
    background: {"rgba(255,255,255,0.08)" if theme=="dark" else "rgba(255,255,255,0.7)"};
    backdrop-filter: blur(18px);
    border-radius: 25px;
    padding: 50px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    margin-bottom: 40px;
}}

.hero {{
    text-align:center;
    padding:120px 20px;
    color: {"white" if theme=="dark" else "#0f172a"};
}}

.metric-card {{
    text-align:center;
    padding:20px;
}}

.footer {{
    text-align:center;
    padding:30px;
    font-size:14px;
    color:gray;
}}

</style>
""", unsafe_allow_html=True)

# =====================================================
# PARTICLE NETWORK BACKGROUND
# =====================================================
st.components.v1.html(f"""
<canvas id="particles"></canvas>
<style>
#particles {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: {"#0f172a" if theme=="dark" else "#f4f7fb"};
}}
</style>

<script>
const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');

function resize() {{
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}}
resize();
window.addEventListener("resize", resize);

let particles = [];
for (let i = 0; i < 70; i++) {{
    particles.push({{
        x: Math.random()*canvas.width,
        y: Math.random()*canvas.height,
        vx: (Math.random()-0.5)*0.4,
        vy: (Math.random()-0.5)*0.4
    }});
}}

function animate() {{
    ctx.clearRect(0,0,canvas.width,canvas.height);

    particles.forEach(p => {{
        p.x += p.vx;
        p.y += p.vy;

        if(p.x<0||p.x>canvas.width) p.vx*=-1;
        if(p.y<0||p.y>canvas.height) p.vy*=-1;

        ctx.beginPath();
        ctx.arc(p.x,p.y,2,0,Math.PI*2);
        ctx.fillStyle = "rgba(59,130,246,0.6)";
        ctx.fill();
    }});

    for(let i=0;i<particles.length;i++){{
        for(let j=i+1;j<particles.length;j++){{
            let dx = particles[i].x - particles[j].x;
            let dy = particles[i].y - particles[j].y;
            let dist = Math.sqrt(dx*dx + dy*dy);
            if(dist<120){{
                ctx.beginPath();
                ctx.moveTo(particles[i].x,particles[i].y);
                ctx.lineTo(particles[j].x,particles[j].y);
                ctx.strokeStyle="rgba(59,130,246,0.08)";
                ctx.stroke();
            }}
        }}
    }}

    requestAnimationFrame(animate);
}}
animate();
</script>
""", height=0)

# =====================================================
# LOAD MODEL
# =====================================================
try:
    model = joblib.load("hpv_epitope_model.pkl")
except:
    st.error("Model file not found. Upload hpv_epitope_model.pkl")
    st.stop()

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

    global_features = np.array([
        hydro_frac, arom_frac, pos_frac,
        neg_frac, net_charge, entropy, avg_weight
    ])

    return np.concatenate([pos_encoding, di_features, global_features])

# =====================================================
# NAVIGATION
# =====================================================
page = st.sidebar.radio(
    "Navigation",
    ["Home","Epitope Scanner","Methods"]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="glass">
            <h1 style="font-size:70px;">HPV-EPIPRED</h1>
            <h3>AI-Driven MHC Class I Epitope Prediction</h3>
            <p>Machine Learning–Based Immunogenic Hotspot Identification</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# EPITOPE SCANNER
# =====================================================
elif page == "Epitope Scanner":

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    fasta = st.text_area("Paste HPV Protein FASTA Sequence")
    run = st.button("🔬 Run Epitope Scan")
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        seq = "".join(
            [l.strip() for l in fasta.split("\n") if not l.startswith(">")]
        ).upper()

        if len(seq) < 9:
            st.error("Sequence must be ≥ 9 amino acids.")
            st.stop()

        if not all(c in aa_list for c in seq):
            st.error("Invalid amino acid detected.")
            st.stop()

        results = []
        for i in range(len(seq)-8):
            pep = seq[i:i+9]
            prob = model.predict_proba(
                [extract_features(pep)]
            )[0][1]

            results.append({
                "Start": i+1,
                "Peptide": pep,
                "Probability": round(prob,3)
            })

        df = pd.DataFrame(results)

        col1, col2 = st.columns(2)
        col1.metric("Protein Length", len(seq))
        col2.metric("Total 9-mers", len(df))

        fig = px.line(
            df,
            x="Start",
            y="Probability",
            template="plotly_dark" if theme=="dark" else "simple_white",
            title="Epitope Probability Landscape"
        )
        fig.add_hline(y=threshold, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

# =====================================================
# METHODS
# =====================================================
elif page == "Methods":

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("""
    ### Model Architecture
    - Model: XGBoost  
    - Epitope Length: 9-mer  
    - Threshold: 0.261  
    - Features:
        - Position-specific encoding
        - Dipeptide composition
        - Physicochemical descriptors
    - Validation:
        - Repeated stratified 70/30 splits
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
HPV-EPIPRED © 2026 | Developed for HPV Immunoinformatics Research
</div>
""", unsafe_allow_html=True)
