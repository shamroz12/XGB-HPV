import streamlit as st

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_toggle = st.toggle("🌗 Toggle Dark / Light Mode")

if theme_toggle:
    st.session_state.theme = "light"
else:
    st.session_state.theme = "dark"

theme = st.session_state.theme

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from itertools import product
from collections import Counter
import math
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import pagesizes
from reportlab.platypus import TableStyle

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="HPV-EPIPRED",
    page_icon="🧬",
    layout="wide"
)

# ==========================================
# PREMIUM BIOTECH STYLING
# ==========================================
st.markdown(f"""
<style>

/* THEME COLORS */
:root {{
    --bg-dark: #0f172a;
    --bg-light: #f4f7fb;
    --card-dark: rgba(255,255,255,0.08);
    --card-light: rgba(255,255,255,0.6);
    --text-dark: white;
    --text-light: #111827;
}}

body {{
    background-color: { "#0f172a" if theme=="dark" else "#f4f7fb" };
}}

/* PARTICLE CANVAS */
#particles {{
    position: fixed;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    z-index: -1;
}}

/* HERO */
.hero {{
    min-height: 75vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: { "white" if theme=="dark" else "#111827" };
    animation: fadeIn 1.5s ease-out;
}}

/* Glassmorphism Panels */
.glass {{
    background: { "rgba(255,255,255,0.08)" if theme=="dark" else "rgba(255,255,255,0.6)" };
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.15);
    animation: fadeUp 1.5s ease-out;
}}

/* Animations */
@keyframes fadeIn {{
    from {{opacity:0;}}
    to {{opacity:1;}}
}}

@keyframes fadeUp {{
    from {{opacity:0; transform: translateY(40px);}}
    to {{opacity:1; transform: translateY(0);}}
}}

</style>
""", unsafe_allow_html=True)

st.components.v1.html("""
<canvas id="particles"></canvas>
<script>
const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let particles = [];

for (let i = 0; i < 60; i++) {
    particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5
    });
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < particles.length; i++) {
        let p = particles[i];
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(59,130,246,0.6)";
        ctx.fill();

        for (let j = i + 1; j < particles.length; j++) {
            let p2 = particles[j];
            let dx = p.x - p2.x;
            let dy = p.y - p2.y;
            let dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < 120) {
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.strokeStyle = "rgba(59,130,246,0.1)";
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(animate);
}
animate();
</script>
""", height=0)

# ==========================================
# LOAD MODEL
# ==========================================
try:
    model = joblib.load("hpv_epitope_model.pkl")
except:
    st.error("Model file not found. Please upload hpv_epitope_model.pkl.")
    st.stop()

threshold = 0.261

# ==========================================
# FEATURE DEFINITIONS
# ==========================================
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

def extract_features_full(seq):
    pos_encoding = np.zeros((9,20))
    for pos, aa in enumerate(seq):
        if aa in aa_index:
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

# ==========================================
# HOTSPOT DETECTION
# ==========================================
def detect_hotspots(df):
    df_high = df[df["Probability"] >= threshold].sort_values("Start")
    hotspots = []

    if df_high.empty:
        return pd.DataFrame()

    start = df_high.iloc[0]["Start"]
    end = start + 8
    probs = [df_high.iloc[0]["Probability"]]

    for i in range(1, len(df_high)):
        cs = df_high.iloc[i]["Start"]
        ce = cs + 8
        if cs <= end:
            end = max(end, ce)
            probs.append(df_high.iloc[i]["Probability"])
        else:
            hotspots.append({
                "Region_Start": start,
                "Region_End": end,
                "Mean_Probability": round(np.mean(probs),3),
                "Peak_Probability": round(max(probs),3)
            })
            start = cs
            end = ce
            probs = [df_high.iloc[i]["Probability"]]

    hotspots.append({
        "Region_Start": start,
        "Region_End": end,
        "Mean_Probability": round(np.mean(probs),3),
        "Peak_Probability": round(max(probs),3)
    })

    return pd.DataFrame(hotspots)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module:",
    ["Home","Epitope Scanner","Hotspot Analysis","Methods"])

# ==========================================
# HOME PAGE (Premium Landing)
# ==========================================
if page == "Home":
st.markdown("""
<div class="hero">
    <div class="glass">
        <h1 style="font-size:60px;">HPV-EPIPRED</h1>
        <h3>HPV-Specific MHC Class I Epitope Prediction Platform</h3>
        <p style="max-width:700px;">
        Machine Learning–Driven Immunogenic Hotspot Identification
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# EPITOPE SCANNER
# ==========================================
elif page == "Epitope Scanner":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    fasta = st.text_area("Paste HPV Protein FASTA Sequence")
    run = st.button("Run Epitope Scan")
    st.markdown('</div>', unsafe_allow_html=True)

    if run:

        lines = fasta.split("\n")
        seq = "".join([l.strip() for l in lines if not l.startswith(">")]).upper()

        if len(seq) < 9:
            st.error("Sequence must be ≥ 9 amino acids.")
            st.stop()

        if not all(c in aa_list for c in seq):
            st.error("Invalid amino acid detected.")
            st.stop()

        results=[]
        for i in range(len(seq)-8):
            pep = seq[i:i+9]
            prob = model.predict_proba([extract_features_full(pep)])[0][1]

            if prob >= 0.60:
                conf="High"
            elif prob >= threshold:
                conf="Moderate"
            else:
                conf="Low"

            results.append({
                "Start":i+1,
                "Peptide":pep,
                "Probability":round(prob,3),
                "Confidence":conf
            })

        df=pd.DataFrame(results)
        df=df.reset_index(drop=True)
        st.session_state["results"]=df

        col1,col2,col3=st.columns(3)
        col1.metric("Protein Length",len(seq))
        col2.metric("Total 9-mers",len(df))
        col3.metric("High Confidence Hits",len(df[df["Confidence"]=="High"]))

        fig = px.line(df,x="Start",y="Probability",
                      title="Epitope Probability Landscape",
                      template="simple_white")
        fig.add_hline(y=threshold,line_dash="dash")
        st.plotly_chart(fig,use_container_width=True)

        st.dataframe(df)

# ==========================================
# HOTSPOT PAGE
# ==========================================
elif page == "Hotspot Analysis":

    if "results" in st.session_state:
        df=st.session_state["results"]
        hotspot_df=detect_hotspots(df)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("Predicted Immunogenic Hotspots")
        st.dataframe(hotspot_df)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Run Epitope Scanner first.")

# ==========================================
# METHODS
# ==========================================
elif page == "Methods":
    st.markdown("""
    <div class="section-card">
    <h3>Model Architecture</h3>
    • Model: XGBoost  
    • Epitope Length: 9-mer core  
    • Threshold: 0.261  
    • Features: Position-specific encoding, dipeptide composition, physicochemical descriptors  
    • Evaluation: Repeated stratified 70/30 splits  
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<footer>
HPV-EPIPRED © 2026 | Developed for HPV Immunoinformatics Research
</footer>
""", unsafe_allow_html=True)
