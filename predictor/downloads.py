"""
Download utilities for the Swollen Polymer Diffusivity Predictor.
"""

import json

from predictor.predict import PredictionResult, SimplePredictorInputs


def prediction_summary_csv(
    pred: PredictionResult,
    inputs: SimplePredictorInputs,
) -> str:
    """Create a CSV string containing inputs and prediction summary."""

    rows = [
        "section,field,value",
        f"input,temperature_k,{inputs.temperature_k}",
        f"input,tg_k,{inputs.tg_k}",
        f"input,mass_ratio,{inputs.mass_ratio}",
        f"input,rho_polymer,{inputs.rho_polymer}",
        f"input,rho_solvent,{inputs.rho_solvent}",
        f"input,polymer_xc,{inputs.polymer_xc}",
        f"input,chris_category,{inputs.chris_category}",
        f"input,smiles,{inputs.smiles}",
        f"input,cas,{inputs.cas}",
        f"input,n_samples,{inputs.n_samples}",
        f"prediction,qrf_p5_log10D,{pred.qrf.p5:.6f}",
        f"prediction,qrf_p50_log10D,{pred.qrf.p50:.6f}",
        f"prediction,qrf_p95_log10D,{pred.qrf.p95:.6f}",
        f"prediction,mlp_p5_log10D,{pred.mlp.p5:.6f}",
        f"prediction,mlp_p50_log10D,{pred.mlp.p50:.6f}",
        f"prediction,mlp_p95_log10D,{pred.mlp.p95:.6f}",
    ]

    return "\n".join(rows) + "\n"


def prediction_summary_json(
    pred: PredictionResult,
    inputs: SimplePredictorInputs,
) -> str:
    """Create a JSON string containing inputs and prediction summary."""

    payload = {
        "inputs": {
            "temperature_k": inputs.temperature_k,
            "tg_k": inputs.tg_k,
            "mass_ratio": inputs.mass_ratio,
            "rho_polymer": inputs.rho_polymer,
            "rho_solvent": inputs.rho_solvent,
            "polymer_xc": inputs.polymer_xc,
            "chris_category": inputs.chris_category,
            "smiles": inputs.smiles,
            "cas": inputs.cas,
            "n_samples": inputs.n_samples,
        },
        "predictions": {
            "qrf": {
                "p5_log10D": pred.qrf.p5,
                "p50_log10D": pred.qrf.p50,
                "p95_log10D": pred.qrf.p95,
            },
            "mlp": {
                "p5_log10D": pred.mlp.p5,
                "p50_log10D": pred.mlp.p50,
                "p95_log10D": pred.mlp.p95,
            },
        },
    }

    return json.dumps(payload, indent=2)