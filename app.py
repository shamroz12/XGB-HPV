import streamlit as st
import numpy as np
import joblib
from itertools import product
from collections import Counter
import math
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="HPV Epitope Prediction Server",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("hpv_epitope_model.pkl")

# -----------------------------
# Amino acid setup
# -----------------------------
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

# -----------------------------
# Feature extraction
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.title("🧬 HPV Protein Epitope Prediction Server")
st.markdown("Upload or paste an HPV protein FASTA sequence to identify potential 9-mer epitopes.")

threshold = 0.261

fasta_input = st.text_area("Paste FASTA sequence below:", height=200)

if st.button("🔍 Scan Protein for Epitopes"):

    if not fasta_input:
        st.error("Please enter a FASTA sequence.")
    else:
        lines = fasta_input.strip().split("\n")
        sequence = "".join([l.strip() for l in lines if not l.startswith(">")])
        sequence = sequence.upper()

        if not all(c in aa_list for c in sequence):
            st.error("Invalid amino acid characters detected.")
        elif len(sequence) < 9:
            st.error("Protein must be at least 9 amino acids long.")
        else:
            all_results = []

            for i in range(len(sequence) - 8):
                peptide = sequence[i:i+9]
                features = extract_features_full(peptide)
                prob = model.predict_proba([features])[0][1]

                # Confidence categories
                if prob >= 0.60:
                    confidence = "High"
                elif prob >= threshold:
                    confidence = "Moderate"
                else:
                    confidence = "Low"

                all_results.append({
                    "Start_Position": i+1,
                    "Peptide": peptide,
                    "Probability": round(prob,3),
                    "Confidence": confidence
                })

            df_all = pd.DataFrame(all_results)

            predicted_df = df_all[df_all["Probability"] >= threshold]

            st.subheader("📊 Prediction Summary")
            st.write(f"Protein Length: {len(sequence)} amino acids")
            st.write(f"Total 9-mers Scanned: {len(df_all)}")
            st.write(f"Predicted Epitopes (≥ {threshold}): {len(predicted_df)}")

            # -----------------------------
            # Probability Plot
            # -----------------------------
            st.subheader("📈 Epitope Probability Across Protein Length")

            plt.figure(figsize=(12,4))
            plt.plot(df_all["Start_Position"], df_all["Probability"])
            plt.axhline(y=threshold, linestyle='--')
            plt.xlabel("Protein Position")
            plt.ylabel("Epitope Probability")
            plt.title("Sliding Window Epitope Probability")
            st.pyplot(plt)

            # -----------------------------
            # Show Table
            # -----------------------------
            st.subheader("🧾 Predicted Epitopes")
            st.dataframe(predicted_df)

            csv = predicted_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predicted Epitopes (CSV)",
                csv,
                "predicted_epitopes.csv",
                "text/csv"
            )
