import streamlit as st
from predictor.interpretation import classify_model_agreement
from predictor.downloads import prediction_summary_csv

MANUSCRIPT_TITLE = (
    "Predicting Solute Diffusivity in Swollen Polymer Systems "
    "Using Quantile Random Forests and Neural Networks"
)

DISCLAIMER = (
    "This provisional companion predictor is provided for reproducibility, "
    "exploratory analysis, and educational use. It is not an FDA-qualified "
    "Regulatory Science Tool (RST), Medical Device Development Tool (MDDT), "
    "or FDA-approved decision-support tool."
)


def show_disclaimer() -> None:
    st.info(DISCLAIMER)

def show_prediction_summary(
    qrf_p5: float,
    qrf_p50: float,
    qrf_p95: float,
    mlp_p5: float,
    mlp_p50: float,
    mlp_p95: float,
) -> None:
    st.divider()
    st.header("Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("QRF Prediction Distribution")
        st.metric("5th percentile log10(D)", f"{qrf_p5:.3f}")
        st.metric("Median log10(D)", f"{qrf_p50:.3f}")
        st.metric("95th percentile log10(D)", f"{qrf_p95:.3f}")

    with col2:
        st.subheader("MLP Ensemble Prediction Distribution")
        st.metric("5th percentile log10(D)", f"{mlp_p5:.3f}")
        st.metric("Median log10(D)", f"{mlp_p50:.3f}")
        st.metric("95th percentile log10(D)", f"{mlp_p95:.3f}")

    delta = abs(qrf_p50 - mlp_p50)
    agreement_label, agreement_text = classify_model_agreement(delta)

    st.subheader("Model Agreement")
    st.write(f"**{agreement_label}**")
    st.write(f"Difference between QRF and MLP medians: **{delta:.3f} log units**")
    st.write(agreement_text)

    st.subheader("Interpretation")
    st.write(
        "These values are placeholder predictions used to test the interface. "
        "In the final predictor, QRF distributions will be derived from tree-to-tree "
        "variability, while MLP ensemble distributions will be derived from independently "
        "trained neural networks with different random seeds."
    )

    st.subheader("Downloads")
    st.download_button(
        "Download placeholder CSV",
        data=prediction_summary_csv(
            qrf_p5=qrf_p5,
            qrf_p50=qrf_p50,
            qrf_p95=qrf_p95,
            mlp_p5=mlp_p5,
            mlp_p50=mlp_p50,
            mlp_p95=mlp_p95,
        ),
        file_name="prediction_summary_placeholder.csv",
        mime="text/csv",
    )


st.set_page_config(
    page_title="Swollen Polymer Diffusivity Predictor",
    layout="wide",
)

st.title("Swollen Polymer Diffusivity Predictor")
st.caption(f"A provisional companion predictor accompanying: *{MANUSCRIPT_TITLE}*")

show_disclaimer()

st.divider()

mode = st.radio(
    "Choose prediction mode",
    ["Known System", "Custom System", "Example Systems"],
    horizontal=True,
)

st.divider()

if mode == "Known System":
    st.header("Known System")
    st.write("Select a curated polymer–solvent–solute system.")

    col1, col2, col3 = st.columns(3)
    with col1:
        polymer = st.selectbox("Polymer", ["PEBAX 4033", "HDPE", "LDPE", "PVC"], index=0)
    with col2:
        solvent = st.selectbox("Solvent", ["Ethanol", "Isopropanol", "Water"], index=0)
    with col3:
        solute = st.selectbox("Solute", ["Solvent Violet 13", "Ethanol"], index=0)

    temperature = st.number_input("Temperature, T (K)", value=298.15)
    mass_ratio = st.number_input("Swollen/dry mass ratio", value=1.10)

    if st.button("Predict", key="known_predict"):
        show_prediction_summary(
            qrf_p5=-8.20,
            qrf_p50=-7.10,
            qrf_p95=-5.90,
            mlp_p5=-7.00,
            mlp_p50=-6.55,
            mlp_p95=-6.10,
        )

elif mode == "Custom System":
    st.header("Custom System")
    st.write("Enter simplified predictor inputs directly.")

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature, T (K)", value=298.15)
        tg_k = st.number_input("Glass-transition temperature, Tg (K)", value=300.0)
        mass_ratio = st.number_input("Swollen/dry mass ratio", value=1.10)
        rho_polymer = st.number_input("Polymer density", value=1.05)
    with col2:
        rho_solvent = st.number_input("Solvent density", value=0.79)
        polymer_xc = st.number_input("Polymer crystallinity, Xc", value=0.0)
        chris_category = st.selectbox("CHRIS category", ["R", "G"], index=0)
        smiles = st.text_input("SMILES preferred", value="CCO")
        cas = st.text_input("CAS optional", value="")

    if st.button("Predict", key="custom_predict"):
        show_prediction_summary(
            qrf_p5=-8.20,
            qrf_p50=-7.10,
            qrf_p95=-5.90,
            mlp_p5=-7.00,
            mlp_p50=-6.55,
            mlp_p95=-6.10,
        )

else:
    st.header("Example Systems")
    st.write("Explore representative examples from the manuscript.")

    example = st.selectbox(
        "Example",
        ["Moderate rubbery", "Moderate glassy", "PEBAX-like benchmark"],
    )

    if st.button("Load Example", key="load_example"):
        st.success(f"Loaded example: {example}")
        show_prediction_summary(
            qrf_p5=-8.20,
            qrf_p50=-7.10,
            qrf_p95=-5.90,
            mlp_p5=-7.00,
            mlp_p50=-6.55,
            mlp_p95=-6.10,
        )
