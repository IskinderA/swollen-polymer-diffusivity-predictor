"""
Reference prediction engine for Swollen Polymer Diffusivity Predictor v2.

This module is the deployment-facing reference wrapper for the prospective
2039-observation models (2032 primary manuscript observations plus seven
additional PEBAX 4033/hexanes observations). It preserves the manuscript
definition of the QRF, 20-member MLP ensemble, and equal-weight QRF--MLP
consensus predictor.

Model-derived empirical prediction distributions:
    QRF:       300 tree-level predictions
    MLP:        20 independently initialized network predictions
    Consensus: 6000 Cartesian QRF-tree x MLP-network pairings

Requested Monte Carlo samples are drawn with replacement from those fixed
model-derived distributions. The empirical 5th--95th percentile ranges are
prediction-variability estimates, not formally calibrated confidence or
credible intervals.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib


HERE = Path(__file__).resolve().parent
SAVED_DIR = HERE / "saved_models_v2"

QRF_PATH = SAVED_DIR / "qrf_pipeline.joblib"
MLP_PATH = SAVED_DIR / "mlp_128_64_pipeline.joblib"
META_PATH = SAVED_DIR / "model_metadata.json"
MLP_ENSEMBLE_DIR = SAVED_DIR / "mlp_ensemble"

WRAPPER_VERSION = "2.0"
EXPECTED_QRF_TREES = 300
EXPECTED_MLP_ENSEMBLE_MEMBERS = 20
EXPECTED_CONSENSUS_PAIRS = EXPECTED_QRF_TREES * EXPECTED_MLP_ENSEMBLE_MEMBERS


def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # v2 provenance guard: wrapper_v2 must never silently load the original
    # 2032-observation deployment artifacts. The v6 metadata written during
    # training records the prospective deployment population as n_rows=2039.
    n_rows = metadata.get("n_rows")
    if n_rows is None:
        raise ValueError(
            "saved_models_v2/model_metadata.json is missing the required "
            "'n_rows' field; cannot verify the v2 deployment model set."
        )
    if int(n_rows) != 2039:
        raise ValueError(
            "wrapper_v2 requires the 2039-observation deployment models, "
            f"but saved_models_v2/model_metadata.json reports n_rows={n_rows}."
        )

    return metadata


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

def _qrf_tree_predictions(X):
    """Return the empirical QRF tree-level predictions in log10(D) space."""
    model = joblib.load(QRF_PATH)
    preprocess = model.named_steps["preprocess"]
    rf_model = model.named_steps["model"]
    X_trans = preprocess.transform(X)
    tree_preds = np.array([
        float(tree.predict(X_trans)[0])
        for tree in rf_model.estimators_
    ], dtype=float)

    if tree_preds.size != EXPECTED_QRF_TREES:
        raise ValueError(
            f"Expected {EXPECTED_QRF_TREES} QRF tree predictions, "
            f"but found {tree_preds.size}."
        )

    return tree_preds


def _rf_quantiles(fitted_pipeline, X, quantiles=(5, 50, 95)):
    preprocess = fitted_pipeline.named_steps["preprocess"]
    rf_model = fitted_pipeline.named_steps["model"]
    X_trans = preprocess.transform(X)
    tree_preds = np.vstack([
        tree.predict(X_trans) for tree in rf_model.estimators_
    ])
    q = np.percentile(tree_preds, quantiles, axis=0)
    return {f"p{qq}": float(q[i, 0]) for i, qq in enumerate(quantiles)}



def _empirical_distribution_stats(log10_values):
    """Return explicit statistics for an empirical prediction/sample distribution."""
    log10_values = np.asarray(log10_values, dtype=float).reshape(-1)
    if log10_values.size == 0:
        raise ValueError("Empirical distribution is empty.")

    D_values = 10 ** log10_values
    log_p5, log_p50, log_p95 = np.percentile(log10_values, [5, 50, 95])
    D_p5, D_p50, D_p95 = np.percentile(D_values, [5, 50, 95])

    return {
        "n": int(log10_values.size),
        "log10D_mean": float(np.mean(log10_values)),
        "log10D_std": float(np.std(log10_values)),
        "log10D_p5": float(log_p5),
        "log10D_p50": float(log_p50),
        "log10D_p95": float(log_p95),
        "D_mean": float(np.mean(D_values)),
        "D_std": float(np.std(D_values)),
        "D_p5": float(D_p5),
        "D_p50": float(D_p50),
        "D_p95": float(D_p95),
    }


def _prefixed_stats(stats, prefix):
    """Flatten empirical-distribution statistics using an explicit prefix."""
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def predict_diffusivity_qrf(**kwargs):
    """Predict with the QRF and report statistics from all tree predictions."""
    X = build_feature_row(**kwargs)
    tree_preds = _qrf_tree_predictions(X)
    stats = _empirical_distribution_stats(tree_preds)

    result = {
        "model": "QRF",
        "wrapper_version": WRAPPER_VERSION,
        "prediction_distribution_log10D": tree_preds.copy(),
        "prediction_distribution_D": 10 ** tree_preds,
        "central_prediction_definition": "Median of QRF tree predictions",
        "central_log10D": stats["log10D_p50"],
        "central_D": stats["D_p50"],
        "log10D_p5": stats["log10D_p5"],
        "log10D_p50": stats["log10D_p50"],
        "log10D_p95": stats["log10D_p95"],
        "D_p5": stats["D_p5"],
        "D_p50": stats["D_p50"],
        "D_p95": stats["D_p95"],
        "units": "cm^2/s",
        "interval_note": (
            "Empirical 5th--95th percentile prediction-variability range from "
            "the QRF tree-level prediction distribution; not a calibrated "
            "confidence interval."
        ),
    }
    result.update(_prefixed_stats(stats, "prediction_distribution"))
    return result


def sample_diffusivity_qrf(n_samples=1000, random_state=None, **kwargs):
    """Return QRF Monte Carlo samples and summary statistics in a result dictionary."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(random_state)
    X = build_feature_row(**kwargs)
    tree_preds = _qrf_tree_predictions(X)
    sampled_log10D = rng.choice(tree_preds, size=n_samples, replace=True)
    sampled_D = 10 ** sampled_log10D

    prediction_stats = _empirical_distribution_stats(tree_preds)
    sample_stats = _empirical_distribution_stats(sampled_log10D)

    result = {
        "model": "QRF",
        "wrapper_version": WRAPPER_VERSION,
        "n_samples": int(n_samples),
        "log10D_samples": sampled_log10D,
        "D_samples": sampled_D,
        "units": "cm^2/s",
    }
    result.update(_prefixed_stats(prediction_stats, "prediction_distribution"))
    result.update(_prefixed_stats(sample_stats, "sample_distribution"))
    return result


def predict_diffusivity_mlp(**kwargs):
    X = build_feature_row(**kwargs)
    model = joblib.load(MLP_PATH)

    log10D = float(model.predict(X)[0])

    return {
        "model": "MLP_128_64",
        "wrapper_version": WRAPPER_VERSION,
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
    if len(ensemble_files) != EXPECTED_MLP_ENSEMBLE_MEMBERS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_MLP_ENSEMBLE_MEMBERS} MLP ensemble files in "
            f"{MLP_ENSEMBLE_DIR}, but found {len(ensemble_files)}."
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
    """Predict with the final MLP ensemble and report all ensemble statistics."""
    X = build_feature_row(**kwargs)
    preds = _mlp_ensemble_predictions(X)
    stats = _empirical_distribution_stats(preds)

    result = {
        "model": "MLP_128_64_ensemble",
        "wrapper_version": WRAPPER_VERSION,
        "n_ensemble_members": int(len(preds)),
        "prediction_distribution_log10D": preds.copy(),
        "prediction_distribution_D": 10 ** preds,
        "central_prediction_definition": "Median of 20 MLP ensemble predictions",
        "central_log10D": stats["log10D_p50"],
        "central_D": stats["D_p50"],
        # Backward-compatible fields
        "mean_log10D": stats["log10D_mean"],
        "std_log10D": stats["log10D_std"],
        "log10D_p5": stats["log10D_p5"],
        "log10D_p50": stats["log10D_p50"],
        "log10D_p95": stats["log10D_p95"],
        "mean_D": stats["D_mean"],
        "D_p5": stats["D_p5"],
        "D_p50": stats["D_p50"],
        "D_p95": stats["D_p95"],
        "units": "cm^2/s",
        "interval_note": (
            "Empirical 5th--95th percentile prediction-variability range from "
            "the 20 independently initialized MLP predictions; not a calibrated "
            "confidence interval."
        ),
    }
    result.update(_prefixed_stats(stats, "prediction_distribution"))
    return result


def sample_diffusivity_mlp(n_samples=1000, random_state=None, **kwargs):
    """Return MLP Monte Carlo samples and summary statistics in a result dictionary."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(random_state)
    X = build_feature_row(**kwargs)
    preds = _mlp_ensemble_predictions(X)
    sampled_log10D = rng.choice(preds, size=n_samples, replace=True)
    sampled_D = 10 ** sampled_log10D

    prediction_stats = _empirical_distribution_stats(preds)
    sample_stats = _empirical_distribution_stats(sampled_log10D)

    result = {
        "model": "MLP_128_64_ensemble",
        "wrapper_version": WRAPPER_VERSION,
        "n_ensemble_members": int(len(preds)),
        "n_samples": int(n_samples),
        "log10D_samples": sampled_log10D,
        "D_samples": sampled_D,
        "units": "cm^2/s",
    }
    result.update(_prefixed_stats(prediction_stats, "prediction_distribution"))
    result.update(_prefixed_stats(sample_stats, "sample_distribution"))
    return result


def _consensus_distribution(X):
    """
    Construct the empirical equal-weight QRF--MLP consensus distribution.

    The consensus is defined in log10(D) space as

        0.5 * QRF + 0.5 * MLP.

    Following the authoritative PEBAX deployment/benchmark implementation,
    every QRF tree prediction is paired with every MLP ensemble-member
    prediction. With 300 QRF trees and 20 MLP members, this yields 6000
    empirical consensus values for one system.
    """
    qrf_preds = _qrf_tree_predictions(X)
    mlp_preds = _mlp_ensemble_predictions(X)

    consensus = 0.5 * (qrf_preds[:, None] + mlp_preds[None, :])
    consensus = consensus.reshape(-1)

    if consensus.size != EXPECTED_CONSENSUS_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_CONSENSUS_PAIRS} consensus pairings, "
            f"but found {consensus.size}."
        )

    return qrf_preds, mlp_preds, consensus


def predict_diffusivity_consensus(**kwargs):
    """
    Predict diffusivity using the equal-weight QRF--MLP consensus predictor.

    The headline central prediction is the manuscript definition:
    0.5 * QRF_p50 + 0.5 * MLP_p50 in log10(D) space. Distribution statistics
    are calculated separately from all 300 x 20 Cartesian pairings.
    """
    X = build_feature_row(**kwargs)
    qrf_preds, mlp_preds, consensus = _consensus_distribution(X)
    stats = _empirical_distribution_stats(consensus)

    qrf_p50 = float(np.percentile(qrf_preds, 50))
    mlp_p50 = float(np.percentile(mlp_preds, 50))
    equal_weight_central = 0.5 * (qrf_p50 + mlp_p50)

    result = {
        "model": "QRF_MLP_equal_weight_consensus",
        "wrapper_version": WRAPPER_VERSION,
        "weight_qrf": 0.5,
        "weight_mlp": 0.5,
        "qrf_prediction_distribution_log10D": qrf_preds.copy(),
        "mlp_prediction_distribution_log10D": mlp_preds.copy(),
        "prediction_distribution_log10D": consensus.copy(),
        "prediction_distribution_D": 10 ** consensus,
        "n_qrf_tree_predictions": int(len(qrf_preds)),
        "n_mlp_ensemble_members": int(len(mlp_preds)),
        "n_consensus_pairs": int(len(consensus)),
        "qrf_log10D_p50": qrf_p50,
        "mlp_log10D_p50": mlp_p50,
        "central_prediction_definition": (
            "0.5 x QRF p50 + 0.5 x MLP p50 in log10(D) space"
        ),
        "central_log10D": float(equal_weight_central),
        "central_D": 10 ** float(equal_weight_central),
        # Backward-compatible fields: headline p50 remains manuscript central.
        "log10D_equal_weight_central": float(equal_weight_central),
        "D_equal_weight_central": 10 ** float(equal_weight_central),
        "log10D_p5": stats["log10D_p5"],
        "log10D_p50": float(equal_weight_central),
        "distribution_log10D_p50": stats["log10D_p50"],
        "log10D_p95": stats["log10D_p95"],
        "D_p5": stats["D_p5"],
        "D_p50": 10 ** float(equal_weight_central),
        "distribution_D_p50": stats["D_p50"],
        "D_p95": stats["D_p95"],
        "units": "cm^2/s",
        "interval_note": (
            "Empirical 5th--95th percentile prediction-variability range from "
            "the Cartesian QRF-tree x MLP-ensemble consensus distribution; "
            "not a calibrated confidence interval."
        ),
    }
    result.update(_prefixed_stats(stats, "prediction_distribution"))
    return result


def sample_diffusivity_consensus(n_samples=1000, random_state=None, **kwargs):
    """Return consensus Monte Carlo samples and summary statistics in a result dictionary."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(random_state)
    X = build_feature_row(**kwargs)
    qrf_preds, mlp_preds, consensus = _consensus_distribution(X)
    sampled_log10D = rng.choice(consensus, size=n_samples, replace=True)
    sampled_D = 10 ** sampled_log10D

    prediction_stats = _empirical_distribution_stats(consensus)
    sample_stats = _empirical_distribution_stats(sampled_log10D)

    result = {
        "model": "QRF_MLP_equal_weight_consensus",
        "wrapper_version": WRAPPER_VERSION,
        "weight_qrf": 0.5,
        "weight_mlp": 0.5,
        "n_qrf_tree_predictions": int(len(qrf_preds)),
        "n_mlp_ensemble_members": int(len(mlp_preds)),
        "n_consensus_pairs": int(len(consensus)),
        "n_samples": int(n_samples),
        "log10D_samples": sampled_log10D,
        "D_samples": sampled_D,
        "units": "cm^2/s",
    }
    result.update(_prefixed_stats(prediction_stats, "prediction_distribution"))
    result.update(_prefixed_stats(sample_stats, "sample_distribution"))
    return result


def predict_diffusivity_all(**kwargs):
    """Return final QRF, final MLP-ensemble, and consensus summaries together."""
    return {
        "qrf": predict_diffusivity_qrf(**kwargs),
        "mlp": predict_diffusivity_mlp_ensemble(**kwargs),
        "consensus": predict_diffusivity_consensus(**kwargs),
    }


def predict_diffusivity(model="consensus", **kwargs):
    """
    Generic prediction dispatcher.

    The default and recommended deployment output is the equal-weight
    QRF--MLP consensus. ``model='mlp'`` refers to the final 20-member MLP
    ensemble median. The standalone random-state-42 MLP is retained only under
    the explicit ``mlp_single`` aliases.
    """
    model = model.lower()

    if model in ["consensus", "hybrid", "blend", "qrf_mlp"]:
        return predict_diffusivity_consensus(**kwargs)

    if model in ["qrf", "rf"]:
        return predict_diffusivity_qrf(**kwargs)

    if model in ["mlp", "nn", "neural_network", "mlp_ensemble"]:
        return predict_diffusivity_mlp_ensemble(**kwargs)

    if model in ["mlp_single", "mlp_standalone", "nn_single"]:
        return predict_diffusivity_mlp(**kwargs)

    raise ValueError(
        "model must be 'consensus', 'qrf', 'mlp', or 'mlp_single'"
    )


def predict_from_simple_interface(model="consensus", **kwargs):
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

def predict_from_simple_interface_consensus(**kwargs):
    """User-friendly equal-weight consensus prediction interface."""
    full_inputs = build_inputs_from_simple_interface(**kwargs)
    return predict_diffusivity_consensus(**full_inputs)


def predict_from_simple_interface_all(**kwargs):
    """Return QRF, MLP-ensemble, and consensus summaries from simple inputs."""
    full_inputs = build_inputs_from_simple_interface(**kwargs)
    return predict_diffusivity_all(**full_inputs)


def sample_from_simple_interface_consensus(
    n_samples=1000,
    random_state=None,
    **kwargs,
):
    """User-friendly consensus sampling interface for Monte Carlo workflows."""
    full_inputs = build_inputs_from_simple_interface(**kwargs)
    return sample_diffusivity_consensus(
        n_samples=n_samples,
        random_state=random_state,
        **full_inputs,
    )

