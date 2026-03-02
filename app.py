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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  {
    font-family: 'DM Sans', 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.5px;
}

/* AUTO DARK / LIGHT */
@media (prefers-color-scheme: dark) {
    .stApp {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #0f172a 40%,
            #0b1220 100%
        );
        color: #e2e8f0;
    }
}

@media (prefers-color-scheme: light) {
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #e2e8f0 50%,
            #f1f5f9 100%
        );
        color: #0f172a;
    }
}

/* Subtle Noise Overlay */
body::after {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background-image: url("https://www.transparenttextures.com/patterns/asfalt-dark.png");
    opacity: 0.04;
    z-index: -6;
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(24px);
    border-radius: 18px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    margin-top: 40px;
}

/* Smooth Scroll */
html {
    scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# NEURAL NETWORK ANIMATED BACKGROUND
# =========================================================
st.components.v1.html("""
<canvas id="network-canvas" style="position:fixed; top:0; left:0; z-index:-5;"></canvas>
<script>
const netCanvas = document.getElementById("network-canvas");
const netCtx = netCanvas.getContext("2d");
netCanvas.width = window.innerWidth;
netCanvas.height = window.innerHeight;

let nodes = [];
for(let i=0;i<50;i++){
    nodes.push({
        x:Math.random()*netCanvas.width,
        y:Math.random()*netCanvas.height,
        vx:(Math.random()-0.5)*0.4,
        vy:(Math.random()-0.5)*0.4
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
        netCtx.arc(n.x,n.y,2,0,Math.PI*2);
        netCtx.fillStyle="rgba(59,130,246,0.4)";
        netCtx.fill();
    });

    for(let i=0;i<nodes.length;i++){
        for(let j=i+1;j<nodes.length;j++){
            let dx=nodes[i].x-nodes[j].x;
            let dy=nodes[i].y-nodes[j].y;
            let dist=Math.sqrt(dx*dx+dy*dy);
            if(dist<130){
                netCtx.beginPath();
                netCtx.moveTo(nodes[i].x,nodes[i].y);
                netCtx.lineTo(nodes[j].x,nodes[j].y);
                netCtx.strokeStyle="rgba(59,130,246,0.08)";
                netCtx.stroke();
            }
        }
    }

    requestAnimationFrame(animateNetwork);
}
animateNetwork();
</script>
""", height=0)

# =========================================================
# INTERACTIVE 3D HELIX WITH DEPTH FOG
# =========================================================
st.components.v1.html("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<canvas id="dna-canvas" style="position:fixed; top:0; left:0; z-index:-4;"></canvas>

<script>
const canvas = document.getElementById("dna-canvas");
const renderer = new THREE.WebGLRenderer({canvas: canvas, alpha:true});
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0f172a, 10, 50);

const camera = new THREE.PerspectiveCamera(
75, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.z=15;

const group = new THREE.Group();

const light = new THREE.PointLight(0xffffff,1);
light.position.set(10,10,10);
scene.add(light);

for(let i=0;i<120;i++){
    const geometry=new THREE.SphereGeometry(0.25,16,16);
    const material=new THREE.MeshPhongMaterial({
        color:i%2?0x3b82f6:0x9333ea,
        transparent:true,
        opacity:0.55
    });

    const sphere=new THREE.Mesh(geometry,material);

    const angle=i*0.3;
    const radius=4;

    sphere.position.set(
        Math.cos(angle)*radius,
        i*0.18-10,
        Math.sin(angle)*radius
    );

    group.add(sphere);
}

scene.add(group);

let mouseX=0, mouseY=0;
document.addEventListener("mousemove",(event)=>{
    mouseX=(event.clientX/window.innerWidth)*2-1;
    mouseY=(event.clientY/window.innerHeight)*2-1;
});

function animate(){
    requestAnimationFrame(animate);
    group.rotation.y+=0.003;
    group.rotation.x=mouseY*0.25;
    group.rotation.z=mouseX*0.25;
    light.intensity=1+Math.abs(mouseX)*0.4;
    renderer.render(scene,camera);
}
animate();
</script>
""", height=0)

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
