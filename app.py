import streamlit as st
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

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="HPV-EPIPRED", layout="wide")

# ===============================
# PREMIUM STYLING
# ===============================
st.markdown("""
<style>
.main {background-color: #f8fafc;}
h1,h2,h3 {color:#1f2937;}
.stButton>button {
    background-color:#2563eb;
    color:white;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL SAFELY
# ===============================
try:
    model = joblib.load("hpv_epitope_model.pkl")
except:
    st.error("Model file not found. Please ensure hpv_epitope_model.pkl is in repository.")
    st.stop()

threshold = 0.261

# ===============================
# AMINO ACIDS
# ===============================
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

# ===============================
# FEATURE EXTRACTION
# ===============================
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

# ===============================
# HOTSPOT DETECTION
# ===============================
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
                "Mean_Prob": round(np.mean(probs),3),
                "Peak_Prob": round(max(probs),3)
            })
            start = cs
            end = ce
            probs = [df_high.iloc[i]["Probability"]]

    hotspots.append({
        "Region_Start": start,
        "Region_End": end,
        "Mean_Prob": round(np.mean(probs),3),
        "Peak_Prob": round(max(probs),3)
    })

    return pd.DataFrame(hotspots)

# ===============================
# PDF REPORT
# ===============================
def generate_pdf(df, protein_len):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=pagesizes.A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("HPV-EPIPRED Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Protein Length: {protein_len}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    top_df = df.sort_values("Probability", ascending=False).head(10)
    data = [["Rank","Start","Peptide","Prob","Conf"]]

    for i, row in enumerate(top_df.itertuples(),1):
        data.append([i,row.Start,row.Peptide,row.Probability,row.Confidence])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('GRID',(0,0),(-1,-1),0.5,colors.black)
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:",
    ["Home","Epitope Scanner","Hotspot Analysis","Methods"])

# ===============================
# HOME
# ===============================
if page=="Home":
    st.title("HPV-EPIPRED")
    st.write("HPV-specific MHC-I (9-mer core) epitope prediction server.")

# ===============================
# EPITOPE SCANNER
# ===============================
elif page=="Epitope Scanner":

    fasta = st.text_area("Paste FASTA Sequence")

    if st.button("Run Scan"):

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

            if prob>=0.60:
                conf="High"
            elif prob>=threshold:
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

        if df.empty:
            st.error("No peptides generated.")
            st.stop()

        df=df.reset_index(drop=True)
        st.session_state["results"]=df

        st.metric("Protein Length",len(seq))
        st.metric("Total 9-mers",len(df))

        # SAFE PLOTTING
        if "Start" in df.columns and "Probability" in df.columns:
            fig=px.line(df,x="Start",y="Probability",
                        title="Epitope Probability Landscape")
            fig.add_hline(y=threshold,line_dash="dash")
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.warning("Plotting skipped due to missing columns.")

        st.dataframe(df)

        pdf=generate_pdf(df,len(seq))
        st.download_button("Download PDF Report",pdf,
                           file_name="HPV_EPIPRED_Report.pdf")

# ===============================
# HOTSPOTS
# ===============================
elif page=="Hotspot Analysis":
    if "results" in st.session_state:
        df=st.session_state["results"]
        hotspot_df=detect_hotspots(df)
        if hotspot_df.empty:
            st.warning("No hotspots detected.")
        else:
            st.dataframe(hotspot_df)
    else:
        st.warning("Run scan first.")

# ===============================
# METHODS
# ===============================
elif page=="Methods":
    st.write("""
    Model: XGBoost  
    Epitope Length: 9-mer core  
    Threshold: 0.261  
    Features: Positional encoding + Dipeptide + Physicochemical  
    """)
