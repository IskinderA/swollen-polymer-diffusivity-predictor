from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib


HERE = Path(__file__).resolve().parent
SAVED_DIR = HERE / "saved_models"

QRF_PATH = SAVED_DIR / "qrf_pipeline.joblib"
MLP_PATH = SAVED_DIR / "mlp_128_64_pipeline.joblib"
META_PATH = SAVED_DIR / "model_metadata.json"
MLP_ENSEMBLE_DIR = SAVED_DIR / "mlp_ensemble"


def load_metadata():
    with open(META_PATH, "r") as f:
        return json.load(f)


def build_feature_row(
    T,
    Tg_K,
    volume_fraction,
    mass_fraction,
    Qv_minus_1,
    Qw_minus_1,
    Polymer_Xc,
    solute_MW,
    solute_ring_count,
    solute_aromatic_ring_count,
    solute_rotatable_bond_count,
    CHRIS_Category,
):
    T_over_Tg = T / Tg_K

    row = {
        "T_over_Tg": T_over_Tg,
        "Tg_K": Tg_K,
        "solute_MW": solute_MW,
        "volume_fraction": volume_fraction,
        "mass_fraction": mass_fraction,
        "Qv-1": Qv_minus_1,
        "Qw-1": Qw_minus_1,
        "Polymer_Xc": Polymer_Xc,
        "solute_ring_count": solute_ring_count,
        "solute_aromatic_ring_count": solute_aromatic_ring_count,
        "solute_rotatable_bond_count": solute_rotatable_bond_count,
        "rotbond_over_Tg": solute_rotatable_bond_count / T_over_Tg,
        "ring_over_Tg": solute_ring_count / T_over_Tg,
        "aromatic_over_Tg": solute_aromatic_ring_count / T_over_Tg,
        "rotbond_times_phi": solute_rotatable_bond_count * volume_fraction,
        "ring_times_phi": solute_ring_count * volume_fraction,
        "CHRIS Category": CHRIS_Category,
    }

    return pd.DataFrame([row])

def swelling_inputs_from_mass_ratio(
    mass_ratio,
    rho_polymer,
    rho_solvent,
):
    """
    Convert mass swelling ratio and densities into model-required swelling descriptors.

    Parameters
    ----------
    mass_ratio : float
        Ratio of final swollen mass to initial dry polymer mass:
        mass_ratio = mass_final / mass_initial.
    rho_polymer : float
        Polymer density.
    rho_solvent : float
        Solvent density.

    Notes
    -----
    Densities must use self-consistent units.
    """

    if mass_ratio < 1:
        raise ValueError("mass_ratio must be greater than or equal to 1.")
    if rho_polymer <= 0 or rho_solvent <= 0:
        raise ValueError("Densities must be positive.")

    Qw_minus_1 = mass_ratio - 1.0
    mass_fraction = (mass_ratio - 1.0) / mass_ratio

    # Assume dry polymer mass = 1 arbitrary unit.
    mass_initial = 1.0
    mass_solvent = mass_ratio - 1.0

    V_polymer = mass_initial / rho_polymer
    V_solvent = mass_solvent / rho_solvent
    V_total = V_polymer + V_solvent

    volume_fraction = V_solvent / V_total
    Qv_minus_1 = (V_total / V_polymer) - 1.0

    return {
        "volume_fraction": volume_fraction,
        "mass_fraction": mass_fraction,
        "Qv_minus_1": Qv_minus_1,
        "Qw_minus_1": Qw_minus_1,
    }

def descriptors_from_smiles(smiles):
    """
    Calculate required solute descriptors from a SMILES string.

    Requires RDKit.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for descriptors_from_smiles(). "
            "Install RDKit or provide descriptors manually."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    return {
        "solute_MW": float(Descriptors.MolWt(mol)),
        "solute_ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "solute_aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "solute_rotatable_bond_count": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    }

def smiles_from_cas(cas):
    """
    Resolve a CAS number to a SMILES string.

    Notes
    -----
    This is a convenience lookup. SMILES is preferred when available.
    CAS lookup may fail or return ambiguous structures for some chemicals.
    """

    try:
        import pubchempy as pcp
        import cirpy
        import chemicals
    except ImportError as exc:
        raise ImportError(
            "CAS lookup requires pubchempy, cirpy, and chemicals. "
            "Install these packages or provide SMILES directly."
        ) from exc

    smiles = None

    try:
        cm = chemicals.search_chemical(cas)
        smiles = cm.smiles
    except Exception:
        pass

    if not smiles:
        try:
            compounds = pcp.get_compounds(cas, namespace="name")
            smiles = compounds[0].isomeric_smiles
        except Exception:
            pass

    if not smiles:
        try:
            smiles = cirpy.resolve(cas, "smiles")
        except Exception:
            pass

    if isinstance(smiles, list):
        smiles = smiles[0]

    if not smiles:
        raise ValueError(
            f"Could not resolve CAS to SMILES: {cas}. "
            "Please provide SMILES or explicit descriptors."
        )

    return smiles

def build_inputs_from_simple_interface(
    T,
    Tg_K,
    mass_ratio,
    rho_polymer,
    rho_solvent,
    Polymer_Xc,
    CHRIS_Category,
    smiles=None,
    cas=None,
    solute_MW=None,
    solute_ring_count=None,
    solute_aromatic_ring_count=None,
    solute_rotatable_bond_count=None,
):
    """
    Convenience helper that converts practical user inputs into the full
    feature inputs required by predict_diffusivity().

    User provides swelling masses/densities and either SMILES or explicit
    solute structure descriptors.
    """

    swelling = swelling_inputs_from_mass_ratio(
        mass_ratio=mass_ratio,
        rho_polymer=rho_polymer,
        rho_solvent=rho_solvent,
    )

    if smiles is None and cas is not None:
        smiles = smiles_from_cas(cas)

    if smiles is not None:
        desc = descriptors_from_smiles(smiles)
    else:
        required = {
            "solute_MW": solute_MW,
            "solute_ring_count": solute_ring_count,
            "solute_aromatic_ring_count": solute_aromatic_ring_count,
            "solute_rotatable_bond_count": solute_rotatable_bond_count,
        }

        missing = [k for k, v in required.items() if v is None]

        if missing:
            raise ValueError(
                "Provide either smiles or all explicit solute descriptors. "
                f"Missing: {missing}"
            )

        desc = required

    return {
        "T": T,
        "Tg_K": Tg_K,
        "Polymer_Xc": Polymer_Xc,
        "CHRIS_Category": CHRIS_Category,
        **swelling,
        **desc,
    }

def _rf_quantiles(fitted_pipeline, X, quantiles=(5, 50, 95)):
    preprocess = fitted_pipeline.named_steps["preprocess"]
    rf_model = fitted_pipeline.named_steps["model"]

    X_trans = preprocess.transform(X)

    tree_preds = np.vstack([
        tree.predict(X_trans) for tree in rf_model.estimators_
    ])

    q = np.percentile(tree_preds, quantiles, axis=0)

    return {f"p{qq}": float(q[i, 0]) for i, qq in enumerate(quantiles)}


def predict_diffusivity_qrf(**kwargs):
    X = build_feature_row(**kwargs)
    model = joblib.load(QRF_PATH)

    q = _rf_quantiles(model, X, quantiles=(5, 50, 95))

    return {
        "model": "QRF",
        "log10D_p5": q["p5"],
        "log10D_p50": q["p50"],
        "log10D_p95": q["p95"],
        "D_p5": 10 ** q["p5"],
        "D_p50": 10 ** q["p50"],
        "D_p95": 10 ** q["p95"],
        "units": "cm^2/s",
    }

def sample_diffusivity_qrf(n_samples=1000, random_state=None, **kwargs):
    """
    Generate random diffusivity samples from the empirical QRF tree distribution.

    Returns D samples in cm^2/s and corresponding log10D samples.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(random_state)

    X = build_feature_row(**kwargs)
    model = joblib.load(QRF_PATH)

    preprocess = model.named_steps["preprocess"]
    rf_model = model.named_steps["model"]

    X_trans = preprocess.transform(X)

    tree_preds = np.array([
        tree.predict(X_trans)[0] for tree in rf_model.estimators_
    ])

    sampled_log10D = rng.choice(
        tree_preds,
        size=n_samples,
        replace=True,
    )

    sampled_D = 10 ** sampled_log10D

    return {
        "model": "QRF",
        "n_samples": int(n_samples),
        "log10D_samples": sampled_log10D,
        "D_samples": sampled_D,
        "units": "cm^2/s",
    }

def predict_diffusivity_mlp(**kwargs):
    X = build_feature_row(**kwargs)
    model = joblib.load(MLP_PATH)

    log10D = float(model.predict(X)[0])

    return {
        "model": "MLP_128_64",
        "log10D": log10D,
        "D": 10 ** log10D,
        "units": "cm^2/s",
    }


def _load_mlp_ensemble():
    ensemble_files = sorted(
        MLP_ENSEMBLE_DIR.glob("mlp_seed_*.joblib")
    )

    if len(ensemble_files) == 0:
        raise FileNotFoundError(
            f"No MLP ensemble files found in {MLP_ENSEMBLE_DIR}"
        )

    return [
        joblib.load(path)
        for path in ensemble_files
    ]


def _mlp_ensemble_predictions(X):
    models = _load_mlp_ensemble()

    preds = np.array([
        float(model.predict(X)[0])
        for model in models
    ])

    return preds


def predict_diffusivity_mlp_ensemble(**kwargs):
    """
    Predict diffusivity using the MLP ensemble.

    Ensemble spread reflects seed-to-seed model variability,
    not a formally calibrated posterior uncertainty.
    """

    X = build_feature_row(**kwargs)

    preds = _mlp_ensemble_predictions(X)

    p5, p50, p95 = np.percentile(
        preds,
        [5, 50, 95],
    )

    mean_log10D = float(np.mean(preds))
    std_log10D = float(np.std(preds))

    return {
        "model": "MLP_128_64_ensemble",
        "n_ensemble_members": int(len(preds)),

        "mean_log10D": mean_log10D,
        "std_log10D": std_log10D,
        "log10D_p5": float(p5),
        "log10D_p50": float(p50),
        "log10D_p95": float(p95),

        "mean_D": 10 ** mean_log10D,
        "D_p5": 10 ** float(p5),
        "D_p50": 10 ** float(p50),
        "D_p95": 10 ** float(p95),

        "units": "cm^2/s",
    }


def sample_diffusivity_mlp(
    n_samples=1000,
    random_state=None,
    **kwargs,
):
    """
    Generate random diffusivity samples from the empirical MLP ensemble.

    Samples are drawn with replacement from predictions of independently
    trained MLP models with different random seeds.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(random_state)

    X = build_feature_row(**kwargs)

    preds = _mlp_ensemble_predictions(X)

    sampled_log10D = rng.choice(
        preds,
        size=n_samples,
        replace=True,
    )

    sampled_D = 10 ** sampled_log10D

    return {
        "model": "MLP_128_64_ensemble",
        "n_ensemble_members": int(len(preds)),
        "n_samples": int(n_samples),
        "log10D_samples": sampled_log10D,
        "D_samples": sampled_D,
        "units": "cm^2/s",
    }

def predict_diffusivity(model="qrf", **kwargs):
    model = model.lower()

    if model in ["qrf", "rf"]:
        return predict_diffusivity_qrf(**kwargs)

    if model in ["mlp", "nn", "neural_network"]:
        return predict_diffusivity_mlp(**kwargs)

    raise ValueError("model must be 'qrf' or 'mlp'")


def predict_from_simple_interface(model="qrf", **kwargs):
    """
    User-friendly prediction interface.

    Accepts swelling masses/densities and either SMILES or explicit descriptors.
    """

    full_inputs = build_inputs_from_simple_interface(**kwargs)

    return predict_diffusivity(
        model=model,
        **full_inputs,
    )


def sample_from_simple_interface_qrf(
    n_samples=1000,
    random_state=None,
    **kwargs,
):
    """
    User-friendly QRF sampling interface for Monte Carlo workflows.
    """

    full_inputs = build_inputs_from_simple_interface(**kwargs)

    return sample_diffusivity_qrf(
        n_samples=n_samples,
        random_state=random_state,
        **full_inputs,
    )

def predict_from_simple_interface_mlp_ensemble(**kwargs):
    """
    User-friendly MLP ensemble prediction interface.

    Accepts mass ratio/densities and either SMILES, CAS, or explicit descriptors.
    """

    full_inputs = build_inputs_from_simple_interface(**kwargs)

    return predict_diffusivity_mlp_ensemble(
        **full_inputs,
    )


def sample_from_simple_interface_mlp(
    n_samples=1000,
    random_state=None,
    **kwargs,
):
    """
    User-friendly MLP ensemble sampling interface for Monte Carlo workflows.
    """

    full_inputs = build_inputs_from_simple_interface(**kwargs)

    return sample_diffusivity_mlp(
        n_samples=n_samples,
        random_state=random_state,
        **full_inputs,
    )

