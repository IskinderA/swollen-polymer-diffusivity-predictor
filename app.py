import streamlit as st

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

st.set_page_config(
    page_title="Swollen Polymer Diffusivity Predictor",
    layout="wide",
)

st.title("Swollen Polymer Diffusivity Predictor")
st.caption(f"A provisional companion predictor accompanying: *{MANUSCRIPT_TITLE}*")

st.info(DISCLAIMER)

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
        st.selectbox("Polymer", ["PEBAX 4033", "HDPE", "LDPE", "PVC"], index=0)
    with col2:
        st.selectbox("Solvent", ["Ethanol", "Isopropanol", "Water"], index=0)
    with col3:
        st.selectbox("Solute", ["Solvent Violet 13", "Ethanol"], index=0)

    st.number_input("Temperature, T (K)", value=298.15)
    st.number_input("Swollen/dry mass ratio", value=1.10)

    st.button("Predict")

elif mode == "Custom System":
    st.header("Custom System")
    st.write("Enter simplified predictor inputs directly.")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Temperature, T (K)", value=298.15)
        st.number_input("Glass-transition temperature, Tg (K)", value=300.0)
        st.number_input("Swollen/dry mass ratio", value=1.10)
        st.number_input("Polymer density", value=1.05)
    with col2:
        st.number_input("Solvent density", value=0.79)
        st.number_input("Polymer crystallinity, Xc", value=0.0)
        st.selectbox("CHRIS category", ["R", "G"], index=0)
        st.text_input("SMILES preferred", value="CCO")
        st.text_input("CAS optional", value="")

    st.button("Predict")

else:
    st.header("Example Systems")
    st.write("Explore representative examples from the manuscript.")

    st.selectbox(
        "Example",
        ["Moderate rubbery", "Moderate glassy", "PEBAX-like benchmark"],
    )

    st.button("Load Example")
