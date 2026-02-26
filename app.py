import streamlit as st
import numpy as np
import joblib
from itertools import product
from collections import Counter
import math

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("hpv_epitope_model.pkl")

# -----------------------------
# Amino acid definitions
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
# Feature extraction function
# -----------------------------
def extract_features_full(seq):
    seq = seq.upper()
    length = len(seq)
    
    # Position encoding
    pos_encoding = np.zeros((9,20))
    for pos, aa in enumerate(seq):
        if aa in aa_index:
            pos_encoding[pos, aa_index[aa]] = 1
    pos_encoding = pos_encoding.flatten()
    
    # Dipeptide
    di_count = Counter([seq[i:i+2] for i in range(len(seq)-1)])
    di_features = np.array([di_count[dp]/8 for dp in dipeptides])
    
    # Physicochemical
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
# Streamlit UI
# -----------------------------
st.title("HPV Epitope Prediction Tool")
st.write("Enter a 9-mer peptide sequence to predict epitope probability.")

sequence = st.text_input("Enter 9-mer Peptide:")

if st.button("Predict"):
    
    if len(sequence) != 9:
        st.error("Please enter exactly 9 amino acids.")
    
    elif not all(c.upper() in aa_list for c in sequence):
        st.error("Invalid amino acid characters detected.")
    
    else:
        features = extract_features_full(sequence)
        prob = model.predict_proba([features])[0][1]
        
        threshold = 0.261  # Your optimized threshold
        
        prediction = "Epitope" if prob >= threshold else "Non-Epitope"
        
        st.success(f"Prediction: {prediction}")
        st.write(f"Probability Score: {round(prob, 3)}")
