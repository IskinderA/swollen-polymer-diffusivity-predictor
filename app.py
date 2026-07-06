import streamlit as st
from predictor.interpretation import classify_model_agreement
from predictor.downloads import prediction_summary_csv, prediction_summary_json
from predictor.utilities import MANUSCRIPT_TITLE, RESEARCH_USE_DISCLAIMER
from predictor.predict import PredictionResult, SimplePredictorInputs, predict
from predictor.descriptors import compute_descriptors
from predictor.known_system import get_polymers, get_solvents, get_solutes, get_system


def show_prediction_summary(pred: PredictionResult, inputs: SimplePredictorInputs) -> None:
    st.divider()
    st.header("Prediction Summary")
    st.subheader("Inputs Used")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.write(f"**Tg:** {inputs.tg_k:.2f} K")
        st.write(f"**Temperature:** {inputs.temperature_k:.2f} K")
        st.write(f"**Swollen/dry mass ratio:** {inputs.mass_ratio:.3f}")
        st.write(f"**Polymer density:** {inputs.rho_polymer:.3f} g/cc")

    with input_col2:
        st.write(f"**Solvent density:** {inputs.rho_solvent:.3f} g/cc")
        st.write(f"**Crystallinity, Xc:** {inputs.polymer_xc:.3f}")
        st.write(f"**Polymer CHRIS category:** {inputs.chris_category}")
        st.write(f"**Solute/Penetrant SMILES:** {inputs.smiles if inputs.smiles else 'Not provided'}")
        st.write(f"**Solute/Penetrant CAS:** {inputs.cas if inputs.cas else 'Not provided'}")
        st.write(f"**Samples:** {inputs.n_samples}")

    with st.expander("Show computed descriptors"):
        descriptors = compute_descriptors(inputs)
        st.json(descriptors)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("QRF Prediction Distribution")
        st.metric("5th percentile log10(D)", f"{pred.qrf.p5:.3f}")
        st.metric("Median log10(D)", f"{pred.qrf.p50:.3f}")
        st.metric("95th percentile log10(D)", f"{pred.qrf.p95:.3f}")

    with col2:
        st.subheader("MLP Ensemble Prediction Distribution")
        st.metric("5th percentile log10(D)", f"{pred.mlp.p5:.3f}")
        st.metric("Median log10(D)", f"{pred.mlp.p50:.3f}")
        st.metric("95th percentile log10(D)", f"{pred.mlp.p95:.3f}")

    delta = abs(pred.qrf.p50 - pred.mlp.p50)
    agreement_label, agreement_text = classify_model_agreement(delta)

    st.subheader("Model Agreement")
    st.write(f"**{agreement_label}**")
    st.write(f"Difference between QRF and MLP medians: **{delta:.3f} log units**")
    st.write(agreement_text)

    st.subheader("Interpretation")
    st.write(
        "QRF prediction distributions are derived from tree-to-tree variability "
        "within the trained random forest. MLP ensemble prediction distributions "
        "are derived from prediction variability across independently trained "
        "neural networks initialized with different random seeds. These empirical "
        "distributions provide complementary representations of prediction "
        "variability and should not be interpreted as calibrated probabilistic "
        "confidence intervals."
    )

    st.subheader("Downloads")
    st.download_button(
        "Download CSV",
        data=prediction_summary_csv(
            pred=pred,
            inputs=inputs,
        ),
        file_name="prediction_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download JSON",
        data=prediction_summary_json(
            pred=pred,
            inputs=inputs,
        ),
        file_name="prediction_summary.json",
        mime="application/json",
    )


st.set_page_config(
    page_title="Swollen Polymer Diffusivity Predictor",
    layout="wide",
)

def show_disclaimer() -> None:
    with st.expander("Research Use Disclaimer", expanded=False):
        st.write(RESEARCH_USE_DISCLAIMER)
        st.markdown(
            "- [CDRH Regulatory Science Tool (RST) Catalog](https://cdrh-rst.fda.gov/)\n"
            "- [Medical Device Development Tool (MDDT) Program](https://www.fda.gov/medical-devices/medical-device-development-tools-mddt)"
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
        polymer = st.selectbox("Polymer", get_polymers(), index=0)

    with col2:
        solvent_options = get_solvents(polymer)
        solvent = st.selectbox("Solvent", solvent_options, index=0)

    with col3:
        solute_options = get_solutes(polymer, solvent)
        solute = st.selectbox("Solute", solute_options, index=0)

    # Retrieve information for the selected known system
    system = get_system(polymer, solvent, solute)

    with st.container(border=True):
        st.subheader("Known System Information")

        st.success(
            "System recognized. Properties below were automatically populated "
            "from the training (i.e., curated Known System) database."
        )

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write(f"**Glass-transition temperature (Tg):** {float(system['Tg_K']):.2f} K")
            st.write(f"**Polymer crystallinity (Xc):** {float(system['Polymer_Xc']):.3f}")
            st.write(f"**CHRIS category:** {system['CHRIS_Category']}")

        with info_col2:
            smiles_value = system.get("Solute_SMILES", "")
            cas_value = system.get("Solute_CAS", "")
    
            st.write(f"**SMILES:** {smiles_value if str(smiles_value) != 'nan' else 'Not available'}")
            st.write(f"**CAS:** {cas_value if str(cas_value) != 'nan' else 'Not available'}")

    temperature = st.number_input("Temperature, T (K)", value=298.15)
    mass_ratio = st.number_input("Swollen/dry mass ratio", value=1.10)

    rho_polymer = st.number_input(
        "Polymer density (g/cc)",
        value=1.05,
        key="known_rho_polymer",
    )

    rho_solvent = st.number_input(
        "Solvent density (g/cc)",
        value=0.79,
        key="known_rho_solvent",
    )

    st.subheader("Prediction Sampling")
    n_samples = st.number_input(
        "Number of samples to generate",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="known_n_samples",
    )

    st.caption(
        "Sampling will be used to generate empirical prediction distributions. "
        "For QRF, samples are drawn with replacement from tree-level predictions "
        "across the 300 trees in the trained random forest. For the MLP, samples "
        "are drawn with replacement from predictions across the 20 independently "
        "trained neural networks."
    )

    if st.button("Predict", key="known_predict"):
        system = get_system(polymer, solvent, solute)

        inputs = SimplePredictorInputs(
            temperature_k=temperature,
            tg_k=float(system["Tg_K"]),
            mass_ratio=mass_ratio,
            rho_polymer=rho_polymer,
            rho_solvent=rho_solvent,
            polymer_xc=float(system["Polymer_Xc"]),
            chris_category=str(system["CHRIS_Category"]),
            smiles=str(system["Solute_SMILES"]) if system.get("Solute_SMILES") else "",
            cas=str(system["Solute_CAS"]) if system.get("Solute_CAS") else "",
            n_samples=n_samples,
        )
        pred = predict(inputs)
        show_prediction_summary(pred, inputs)

elif mode == "Custom System":
    st.header("Custom System")
    st.write("Enter simplified predictor inputs directly.")

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature, T (K)", value=298.15)
        tg_k = st.number_input("Glass-transition temperature, Tg (K)", value=300.0)
        mass_ratio = st.number_input("Swollen/dry mass ratio", value=1.10)
        rho_polymer = st.number_input("Polymer density, g/cc", value=1.05)
    with col2:
        rho_solvent = st.number_input("Solvent density, g/cc", value=0.79)
        polymer_xc = st.number_input("Polymer crystallinity, Xc", value=0.0)
        chris_category = st.selectbox("Polymer CHRIS category",
            ["R1", "R2", "P1", "P2", "P3", "P4", "G1", "G2"],
            index=0,
            help=(
                "CHRIS refers to the CHemical RISk calculators polymer transport "
                "classification (as reported in Elder et al. 2023, DOI: 10.1002/pol.20230219) used as an input in the trained models in the current work. The refined "
                "categories R1, R2, P1–P4, and G1–G2 distinguish polymer transport "
                "behavior across rubbers, plastics, and glasses."
            ),
        )
        smiles = st.text_input("Solute/Penetrant SMILES preferred", value="CCO")
        cas = st.text_input("Solute/Penetrant CAS optional", value="")

    st.subheader("Prediction Sampling")
    n_samples = st.number_input(
        "Number of samples to generate",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="custom_n_samples",
    )

    st.caption(
        "Sampling will be used to generate empirical prediction distributions. "
        "For QRF, samples are drawn with replacement from tree-level predictions "
        "across the 300 trees in the trained random forest. For the MLP, samples "
        "are drawn with replacement from predictions across the 20 independently "
        "trained neural networks."
    )

    if st.button("Predict", key="custom_predict"):
        inputs = SimplePredictorInputs(
            temperature_k=temperature,
            tg_k=tg_k,
            mass_ratio=mass_ratio,
            rho_polymer=rho_polymer,
            rho_solvent=rho_solvent,
            polymer_xc=polymer_xc,
            chris_category=chris_category,
            smiles=smiles,
            cas=cas,
            n_samples=n_samples,
        )
        pred = predict(inputs)
        show_prediction_summary(pred, inputs)

else:
    st.header("Example Systems")
    st.write("Explore representative examples using the Known System engine.")

    examples = {
        "Moderate rubbery": {
            "polymer": "LDPE",
            "solvent": "benzene",
            "solute": "benzene",
            "temperature_k": 298.15,
            "mass_ratio": 1.10,
            "rho_polymer": 0.92,
            "rho_solvent": 0.87,
        },
        "Moderate glassy": {
            "polymer": "PC",
            "solvent": "Chloroform",
            "solute": "Chloroform",
            "temperature_k": 298.15,
            "mass_ratio": 1.03,
            "rho_polymer": 1.20,
            "rho_solvent": 1.49,
        },
        "PEBAX-like benchmark": {
            "polymer": "PEBAX_4033",
            "solvent": "Ethanol",
            "solute": "Solvent Violet 13",
            "temperature_k": 298.15,
            "mass_ratio": 1.20,
            "rho_polymer": 1.01,
            "rho_solvent": 0.79,
        },
    }

    example = st.selectbox("Example", list(examples.keys()))
    selected = examples[example]

    with st.container(border=True):
        st.subheader("Example Definition")
        st.write(f"**Polymer:** {selected['polymer']}")
        st.write(f"**Solvent:** {selected['solvent']}")
        st.write(f"**Solute:** {selected['solute']}")
        st.write(f"**Temperature:** {selected['temperature_k']:.2f} K")
        st.write(f"**Swollen/dry mass ratio:** {selected['mass_ratio']:.3f}")
        st.write(f"**Polymer density:** {selected['rho_polymer']:.3f} g/cc")
        st.write(f"**Solvent density:** {selected['rho_solvent']:.3f} g/cc")

    if st.button("Run Example", key="run_example"):
        try:
            system = get_system(
                selected["polymer"],
                selected["solvent"],
                selected["solute"],
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        inputs = SimplePredictorInputs(
            temperature_k=selected["temperature_k"],
            tg_k=float(system["Tg_K"]),
            mass_ratio=selected["mass_ratio"],
            rho_polymer=selected["rho_polymer"],
            rho_solvent=selected["rho_solvent"],
            polymer_xc=float(system["Polymer_Xc"]),
            chris_category=str(system["CHRIS_Category"]),
            smiles=str(system["Solute_SMILES"]) if str(system["Solute_SMILES"]) != "nan" else "",
            cas=str(system["Solute_CAS"]) if str(system["Solute_CAS"]) != "nan" else "",
            n_samples=1000,
        )

        pred = predict(inputs)
        show_prediction_summary(pred, inputs)