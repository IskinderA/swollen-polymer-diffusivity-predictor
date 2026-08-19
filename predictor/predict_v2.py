"""
Application-level prediction interface for Swollen Polymer Diffusivity Predictor v2.

This module adapts ``predictor.reference.wrapper_v2`` for the Streamlit layer.
It keeps the original v1 app-facing fields where practical while adding the
final equal-weight QRF--MLP consensus predictor and an explicit distinction
between:

1. model-derived prediction distributions
   - QRF: 300 tree-level predictions
   - MLP: 20 independently initialized network predictions
   - Consensus: 300 x 20 = 6000 Cartesian pairwise predictions

2. requested Monte Carlo sample distributions
   - n values sampled with replacement from the corresponding model-derived
     prediction distribution.

The recommended headline prediction is the manuscript-defined consensus
central value: 0.5 * QRF p50 + 0.5 * MLP p50 in log10(D) space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionDistribution:
    """Summary of one fixed model-derived prediction distribution."""

    p5: float
    p50: float
    p95: float
    mean: float
    sd: float
    n: int

    D_p5: float
    D_p50: float
    D_p95: float
    D_mean: float
    D_sd: float


@dataclass(frozen=True)
class SampleDistribution:
    """Summary of one requested Monte Carlo sample distribution."""

    p5: float
    p50: float
    p95: float
    mean: float
    sd: float
    n: int

    D_p5: float
    D_p50: float
    D_p95: float
    D_mean: float
    D_sd: float


@dataclass
class PredictionResult:
    """
    Complete v2 result returned to the application layer.

    ``qrf``, ``mlp``, and ``consensus`` are model-derived distribution
    summaries. For consensus, ``consensus.p50`` is the manuscript-defined
    headline central prediction, while ``consensus_distribution_p50`` stores
    the median of the full 6000-member pairwise distribution separately.

    ``*_prediction_log10`` / ``*_prediction_D`` are the fixed model-derived
    distributions. ``*_log10_samples`` / ``*_D_samples`` are requested Monte
    Carlo draws sampled with replacement from those distributions.

    The v1 fields ``qrf_log10_samples``, ``qrf_D_samples``,
    ``mlp_log10_samples``, and ``mlp_D_samples`` are intentionally retained.
    """

    qrf: PredictionDistribution
    mlp: PredictionDistribution
    consensus: PredictionDistribution

    # Manuscript-defined headline consensus central prediction.
    consensus_central_log10D: float
    consensus_central_D: float

    # Median of the 6000-member pairwise consensus distribution; conceptually
    # distinct from the manuscript-defined headline central prediction.
    consensus_distribution_p50: float
    consensus_distribution_D_p50: float

    # Fixed model-derived prediction distributions.
    qrf_prediction_log10: list[float]
    qrf_prediction_D: list[float]
    mlp_prediction_log10: list[float]
    mlp_prediction_D: list[float]
    consensus_prediction_log10: list[float]
    consensus_prediction_D: list[float]

    # Requested Monte Carlo distributions. These preserve v1 QRF/MLP names.
    qrf_log10_samples: list[float]
    qrf_D_samples: list[float]
    mlp_log10_samples: list[float]
    mlp_D_samples: list[float]
    consensus_log10_samples: list[float]
    consensus_D_samples: list[float]

    # Explicit MC summary objects for v2 UI/download use.
    qrf_sample: SampleDistribution
    mlp_sample: SampleDistribution
    consensus_sample: SampleDistribution

    # Provenance / model sizes.
    n_qrf_tree_predictions: int
    n_mlp_ensemble_members: int
    n_consensus_pairs: int
    n_samples_requested: int
    wrapper_version: str


@dataclass(frozen=True)
class SimplePredictorInputs:
    """Simple user-facing inputs used by the Streamlit application."""

    temperature_k: float
    tg_k: float
    mass_ratio: float
    rho_polymer: float
    rho_solvent: float
    polymer_xc: float
    chris_category: str
    smiles: str = ""
    cas: str = ""
    n_samples: int = 1000


def _distribution_summary(result: dict, *, consensus: bool = False) -> PredictionDistribution:
    """Convert wrapper prediction-distribution statistics to app dataclass."""

    p50 = (
        float(result["log10D_p50"])
        if consensus
        else float(result["prediction_distribution_log10D_p50"])
    )
    D_p50 = (
        float(result["D_p50"])
        if consensus
        else float(result["prediction_distribution_D_p50"])
    )

    return PredictionDistribution(
        p5=float(result["prediction_distribution_log10D_p5"]),
        p50=p50,
        p95=float(result["prediction_distribution_log10D_p95"]),
        mean=float(result["prediction_distribution_log10D_mean"]),
        sd=float(result["prediction_distribution_log10D_std"]),
        n=int(result["prediction_distribution_n"]),
        D_p5=float(result["prediction_distribution_D_p5"]),
        D_p50=D_p50,
        D_p95=float(result["prediction_distribution_D_p95"]),
        D_mean=float(result["prediction_distribution_D_mean"]),
        D_sd=float(result["prediction_distribution_D_std"]),
    )


def _sample_summary(result: dict) -> SampleDistribution:
    """Convert wrapper Monte Carlo sample statistics to app dataclass."""

    return SampleDistribution(
        p5=float(result["sample_distribution_log10D_p5"]),
        p50=float(result["sample_distribution_log10D_p50"]),
        p95=float(result["sample_distribution_log10D_p95"]),
        mean=float(result["sample_distribution_log10D_mean"]),
        sd=float(result["sample_distribution_log10D_std"]),
        n=int(result["sample_distribution_n"]),
        D_p5=float(result["sample_distribution_D_p5"]),
        D_p50=float(result["sample_distribution_D_p50"]),
        D_p95=float(result["sample_distribution_D_p95"]),
        D_mean=float(result["sample_distribution_D_mean"]),
        D_sd=float(result["sample_distribution_D_std"]),
    )


def _as_float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    """Return a JSON/Streamlit-friendly list of Python floats."""

    return np.asarray(values, dtype=float).reshape(-1).tolist()


def predict(inputs: SimplePredictorInputs) -> PredictionResult:
    """
    Generate QRF, MLP-ensemble, and equal-weight consensus predictions.

    The model-derived distributions are calculated once per predictor. A
    separate requested Monte Carlo distribution of ``inputs.n_samples`` is
    then sampled with replacement from each empirical distribution using a
    fixed random state for reproducible application downloads.
    """

    if int(inputs.n_samples) <= 0:
        raise ValueError("n_samples must be positive.")

    # Import the explicitly versioned v2 reference engine. This prevents the
    # v2 app from silently falling back to the original 2032-observation wrapper.
    from predictor.reference import wrapper_v2 as wrapper

    common_kwargs = dict(
        T=inputs.temperature_k,
        Tg_K=inputs.tg_k,
        mass_ratio=inputs.mass_ratio,
        rho_polymer=inputs.rho_polymer,
        rho_solvent=inputs.rho_solvent,
        Polymer_Xc=inputs.polymer_xc,
        CHRIS_Category=inputs.chris_category,
        smiles=inputs.smiles or None,
        cas=inputs.cas or None,
    )

    # Fixed model-derived prediction distributions.
    all_results = wrapper.predict_from_simple_interface_all(**common_kwargs)
    qrf_result = all_results["qrf"]
    mlp_result = all_results["mlp"]
    consensus_result = all_results["consensus"]

    # Requested Monte Carlo samples. Keep the same seed across model classes
    # for reproducibility; each function samples from its own empirical pool.
    random_state = 42
    qrf_samples = wrapper.sample_from_simple_interface_qrf(
        n_samples=inputs.n_samples,
        random_state=random_state,
        **common_kwargs,
    )
    mlp_samples = wrapper.sample_from_simple_interface_mlp(
        n_samples=inputs.n_samples,
        random_state=random_state,
        **common_kwargs,
    )
    consensus_samples = wrapper.sample_from_simple_interface_consensus(
        n_samples=inputs.n_samples,
        random_state=random_state,
        **common_kwargs,
    )

    qrf = _distribution_summary(qrf_result)
    mlp = _distribution_summary(mlp_result)
    consensus = _distribution_summary(consensus_result, consensus=True)

    n_qrf = int(consensus_result["n_qrf_tree_predictions"])
    n_mlp = int(consensus_result["n_mlp_ensemble_members"])
    n_consensus = int(consensus_result["n_consensus_pairs"])

    # Hard guards make accidental model-folder/version mismatches visible.
    if n_qrf != 300:
        raise ValueError(f"v2 expected 300 QRF tree predictions, found {n_qrf}.")
    if n_mlp != 20:
        raise ValueError(f"v2 expected 20 MLP ensemble members, found {n_mlp}.")
    if n_consensus != 6000:
        raise ValueError(
            f"v2 expected 6000 consensus pairings, found {n_consensus}."
        )

    return PredictionResult(
        qrf=qrf,
        mlp=mlp,
        consensus=consensus,
        consensus_central_log10D=float(consensus_result["central_log10D"]),
        consensus_central_D=float(consensus_result["central_D"]),
        consensus_distribution_p50=float(
            consensus_result["prediction_distribution_log10D_p50"]
        ),
        consensus_distribution_D_p50=float(
            consensus_result["prediction_distribution_D_p50"]
        ),
        qrf_prediction_log10=_as_float_list(
            qrf_result["prediction_distribution_log10D"]
        ),
        qrf_prediction_D=_as_float_list(
            qrf_result["prediction_distribution_D"]
        ),
        mlp_prediction_log10=_as_float_list(
            mlp_result["prediction_distribution_log10D"]
        ),
        mlp_prediction_D=_as_float_list(
            mlp_result["prediction_distribution_D"]
        ),
        consensus_prediction_log10=_as_float_list(
            consensus_result["prediction_distribution_log10D"]
        ),
        consensus_prediction_D=_as_float_list(
            consensus_result["prediction_distribution_D"]
        ),
        qrf_log10_samples=_as_float_list(qrf_samples["log10D_samples"]),
        qrf_D_samples=_as_float_list(qrf_samples["D_samples"]),
        mlp_log10_samples=_as_float_list(mlp_samples["log10D_samples"]),
        mlp_D_samples=_as_float_list(mlp_samples["D_samples"]),
        consensus_log10_samples=_as_float_list(
            consensus_samples["log10D_samples"]
        ),
        consensus_D_samples=_as_float_list(consensus_samples["D_samples"]),
        qrf_sample=_sample_summary(qrf_samples),
        mlp_sample=_sample_summary(mlp_samples),
        consensus_sample=_sample_summary(consensus_samples),
        n_qrf_tree_predictions=n_qrf,
        n_mlp_ensemble_members=n_mlp,
        n_consensus_pairs=n_consensus,
        n_samples_requested=int(inputs.n_samples),
        wrapper_version=str(consensus_result.get("wrapper_version", "v2")),
    )
