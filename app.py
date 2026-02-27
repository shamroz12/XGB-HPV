import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product
from collections import Counter
import math
import plotly.express as px

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="HPV-EPIPRED",
    layout="wide"
)

# -----------------------------------
# Load Model
# -----------------------------------
model = joblib.load("hpv_epitope_model.pkl")
threshold = 0.261

# -----------------------------------
# Amino Acid Setup
# -----------------------------------
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

# -----------------------------------
# Feature Extraction
# -----------------------------------
def extract_features_full(seq):
    seq = seq.upper()
    length = len(seq)

    pos_encoding = np.zeros((9,20))
    for pos, aa in enumerate(seq):
        if aa in aa_index:
            pos_encoding[pos, aa_index[aa]] = 1
    pos_encoding = pos_encoding.flatten()

    di_count = Counter([seq[i:i+2] for i in range(len(seq)-1)])
    di_features = np.array([di_count[dp]/8 for dp in dipeptides])

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

# -----------------------------------
# Hotspot Detection
# -----------------------------------
def detect_hotspots(df):
    df_high = df[df["Probability"] >= threshold].sort_values("Start_Position")
    hotspots = []

    if df_high.empty:
        return pd.DataFrame()

    start = df_high.iloc[0]["Start_Position"]
    end = start + 8
    probs = [df_high.iloc[0]["Probability"]]

    for i in range(1, len(df_high)):
        current_start = df_high.iloc[i]["Start_Position"]
        current_end = current_start + 8

        if current_start <= end:
            end = max(end, current_end)
            probs.append(df_high.iloc[i]["Probability"])
        else:
            hotspots.append({
                "Region_Start": start,
                "Region_End": end,
                "Mean_Probability": round(np.mean(probs),3),
                "Peak_Probability": round(max(probs),3)
            })
            start = current_start
            end = current_end
            probs = [df_high.iloc[i]["Probability"]]

    hotspots.append({
        "Region_Start": start,
        "Region_End": end,
        "Mean_Probability": round(np.mean(probs),3),
        "Peak_Probability": round(max(probs),3)
    })

    return pd.DataFrame(hotspots)

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "🔬 Epitope Scanner",
    "📊 Hotspot Analysis",
    "🏆 Top Candidates",
    "ℹ️ Methods & Model Info"
])

# -----------------------------------
# HOME PAGE
# -----------------------------------
if page == "🏠 Home":
    st.title("🧬 HPV-EPIPRED")
    st.markdown("""
    **HPV-Specific MHC Class I (9-mer Core) Epitope Prediction Server**

    This tool scans HPV protein sequences using a machine learning model 
    trained on experimentally validated 9-mer MHC-I epitopes.

    Features:
    - Sliding window epitope scanning
    - Interactive probability visualization
    - Hotspot region detection
    - Ranked candidate selection
    - Confidence stratification
    """)

# -----------------------------------
# EPITOPE SCANNER
# -----------------------------------
elif page == "🔬 Epitope Scanner":

    st.title("Epitope Scanner")

    fasta_input = st.text_area("Paste HPV Protein FASTA Sequence:")

    if st.button("Scan Protein"):

        lines = fasta_input.strip().split("\n")
        sequence = "".join([l.strip() for l in lines if not l.startswith(">")])
        sequence = sequence.upper()

        if len(sequence) < 9:
            st.error("Protein must be at least 9 amino acids long.")
        elif not all(c in aa_list for c in sequence):
            st.error("Invalid amino acid detected.")
        else:

            results = []

            for i in range(len(sequence)-8):
                peptide = sequence[i:i+9]
                features = extract_features_full(peptide)
                prob = model.predict_proba([features])[0][1]

                if prob >= 0.60:
                    confidence = "High"
                elif prob >= threshold:
                    confidence = "Moderate"
                else:
                    confidence = "Low"

                results.append({
                    "Start_Position": i+1,
                    "Peptide": peptide,
                    "Probability": round(prob,3),
                    "Confidence": confidence
                })

            df = pd.DataFrame(results)
            st.session_state["results"] = df

            st.success("Scanning Complete.")

            fig = px.line(df, x="Start_Position", y="Probability",
                          title="Epitope Probability Landscape")
            fig.add_hline(y=threshold, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df)

# -----------------------------------
# HOTSPOT ANALYSIS
# -----------------------------------
elif page == "📊 Hotspot Analysis":

    st.title("Predicted Immunogenic Hotspots")

    if "results" in st.session_state:
        df = st.session_state["results"]
        hotspot_df = detect_hotspots(df)

        if hotspot_df.empty:
            st.warning("No hotspot regions detected.")
        else:
            st.dataframe(hotspot_df)
    else:
        st.warning("Please run Epitope Scanner first.")

# -----------------------------------
# TOP CANDIDATES
# -----------------------------------
elif page == "🏆 Top Candidates":

    st.title("Top 10 Ranked Epitopes")

    if "results" in st.session_state:
        df = st.session_state["results"]
        top_df = df.sort_values("Probability", ascending=False).head(10)
        top_df.insert(0, "Rank", range(1, len(top_df)+1))
        st.dataframe(top_df)
    else:
        st.warning("Please run Epitope Scanner first.")

# -----------------------------------
# METHODS PAGE
# -----------------------------------
elif page == "ℹ️ Methods & Model Info":

    st.title("Model Information")

    st.markdown(f"""
    **Model Type:** XGBoost  
    **Epitope Length:** 9-mer (MHC-I core)  
    **Decision Threshold:** {threshold}  
    **Feature Types:**
    - Position-specific one-hot encoding
    - Dipeptide composition
    - Physicochemical descriptors  

    **Evaluation:**
    - Repeated stratified 70/30 splits
    - Mean ROC-AUC ≈ 0.865
    - Mean Accuracy ≈ 0.82  

    **Disclaimer:**
    Predictions are computational and should be experimentally validated.
    """)
