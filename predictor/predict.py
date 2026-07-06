"""
Prediction interface for the Swollen Polymer Diffusivity Predictor.

This module calls the frozen reference prediction engine used for the
manuscript models and returns structured QRF and MLP prediction results.
"""

from dataclasses import dataclass


@dataclass
class PredictionDistribution:
    p5: float
    p50: float
    p95: float


@dataclass
class PredictionResult:
    qrf: PredictionDistribution
    mlp: PredictionDistribution
    qrf_log10_samples: list[float]
    qrf_D_samples: list[float]
    mlp_log10_samples: list[float]
    mlp_D_samples: list[float]

@dataclass
class SimplePredictorInputs:
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


def predict(inputs: SimplePredictorInputs) -> PredictionResult:
    """Generate predictions using the reference prediction engine."""

    from predictor.reference import wrapper

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

    qrf_result = wrapper.predict_from_simple_interface(
        model="qrf",
        **common_kwargs,
    )

    mlp_result = wrapper.predict_from_simple_interface_mlp_ensemble(
        **common_kwargs,
    )

    qrf_samples = wrapper.sample_from_simple_interface_qrf(
        n_samples=inputs.n_samples,
        random_state=42,
        **common_kwargs,
    )

    mlp_samples = wrapper.sample_from_simple_interface_mlp(
        n_samples=inputs.n_samples,
        random_state=42,
        **common_kwargs,
    )

    return PredictionResult(
        qrf=PredictionDistribution(
            p5=qrf_result["log10D_p5"],
            p50=qrf_result["log10D_p50"],
            p95=qrf_result["log10D_p95"],
        ),
        mlp=PredictionDistribution(
            p5=mlp_result["log10D_p5"],
            p50=mlp_result["log10D_p50"],
            p95=mlp_result["log10D_p95"],
        ),
        qrf_log10_samples=qrf_samples["log10D_samples"].tolist(),
        qrf_D_samples=qrf_samples["D_samples"].tolist(),
        mlp_log10_samples=mlp_samples["log10D_samples"].tolist(),
        mlp_D_samples=mlp_samples["D_samples"].tolist(),
    )