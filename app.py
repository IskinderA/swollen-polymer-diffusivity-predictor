import streamlit as st

st.set_page_config(
    page_title="Swollen Polymer Diffusivity Predictor",
    layout="wide",
)

st.title("Swollen Polymer Diffusivity Predictor")
st.caption(
    "A provisional companion predictor accompanying the manuscript "
    "\"Predicting Solute Diffusivity in Swollen Polymer Systems Using Quantile Random Forests and Neural Networks\""
)

st.info(
    "This is a provisional research companion predictor for reproducibility, "
    "exploratory analysis, and educational use. It is not an FDA-qualified RST, "
    "MDDT, or FDA-approved decision-support tool."
)

mode = st.radio(
    "Choose prediction mode",
    ["Known System", "Custom System", "Example Systems"],
)

st.write(f"Selected mode: **{mode}**")
