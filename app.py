import gradio as gr
import pandas as pd
import joblib

model = joblib.load("rf_clinical_pirad.pkl")


def predict_pca(
    Age,
    DRE,
    PSA,
    freePSA,
    PHI,
    free_total_PSA_ratio,
    Prostate_volume,
    p2PSA,
    PSAD,
    PIRAD
):

    X = pd.DataFrame([{
        "Age": Age,
        "DRE": DRE,
        "PSA": PSA,
        "freePSA": freePSA,
        "PHI": PHI,
        "free_total_PSA_ratio": free_total_PSA_ratio,
        "Prostate_volume": Prostate_volume,
        "p2PSA": p2PSA,
        "PSAD": PSAD,
        "PIRAD": PIRAD
    }])

    prob = model.predict_proba(X)[0, 1]

    if prob < 0.20:
        risk = "Low Risk"
    elif prob < 0.50:
        risk = "Intermediate Risk"
    else:
        risk = "High Risk"

    return f"{prob*100:.1f}%", risk


demo = gr.Interface(
    fn=predict_pca,

    inputs=[

        gr.Number(label="Age"),

        gr.Dropdown(
            choices=[0,1],
            value=0,
            label="DRE (0=Negative, 1=Positive)"
        ),

        gr.Number(label="PSA"),

        gr.Number(label="freePSA"),

        gr.Number(label="PHI"),

        gr.Number(label="free_total_PSA_ratio"),

        gr.Number(label="Prostate_volume"),

        gr.Number(label="p2PSA"),

        gr.Number(label="PSAD"),

        gr.Dropdown(
            choices=[1,2,3,4,5],
            value=3,
            label="PI-RADS"
        )

    ],

    outputs=[
        gr.Textbox(label="PCa Probability"),
        gr.Textbox(label="Risk Category")
    ],

    title="RF Clinical + PI-RADS Prostate Cancer Predictor",

    description="""
Random Forest model for prostate cancer prediction
using clinical variables and PI-RADS score.
"""
)

demo.launch()
