import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_clinical_pirad.pkl")

st.title("RF Clinical + PI-RADS Prostate Cancer Predictor")

st.write(
    "Predict probability of prostate cancer using clinical variables and PI-RADS."
)

age = st.number_input("Age", value=65.0)

dre = st.selectbox(
    "DRE",
    [0, 1],
    format_func=lambda x: "Negative" if x == 0 else "Positive"
)

psa = st.number_input("PSA (ng/mL)", value=6.0)

freepsa = st.number_input("free PSA (ng/mL)", value=1.0)

phi = st.number_input("PHI", value=35.0)

ratio = st.number_input(
    "free/total PSA ratio",
    value=0.15
)

volume = st.number_input(
    "Prostate Volume (mL)",
    value=40.0
)

p2psa = st.number_input(
    "p2PSA",
    value=20.0
)

psad = st.number_input(
    "PSAD",
    value=0.15
)

pirad = st.selectbox(
    "PI-RADS",
    [1, 2, 3, 4, 5]
)

if st.button("Predict"):

    X = pd.DataFrame([{

        "Age": age,
        "DRE": dre,
        "PSA": psa,
        "freePSA": freepsa,
        "PHI": phi,
        "free_total_PSA_ratio": ratio,
        "Prostate_volume": volume,
        "p2PSA": p2psa,
        "PSAD": psad,
        "PIRAD": pirad

    }])

    prob = model.predict_proba(X)[0,1]

    st.metric(
        "PCa Probability",
        f"{prob*100:.1f}%"
    )

    if prob < 0.20:

        st.success(
            "Low Risk"
        )

    elif prob < 0.50:

        st.warning(
            "Intermediate Risk"
        )

    else:

        st.error(
            "High Risk"
        )
