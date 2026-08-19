"""
Download utilities for Swollen Polymer Diffusivity Predictor v2.

This module is intentionally paired with ``predictor.predict_v2``.

The v2 download structure distinguishes:

1. compact prediction summaries;
2. fixed model-derived prediction distributions:
   - QRF: 300 tree predictions
   - MLP: 20 network predictions
   - Consensus: 6000 QRF-tree x MLP-network pairings
3. requested Monte Carlo sample distributions:
   - n samples drawn with replacement from the corresponding fixed
     model-derived distribution.

The manuscript-defined headline consensus prediction is retained separately
from the median of the 6000-member consensus prediction distribution.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from typing import Iterable

from predictor.predict_v2 import (
    PredictionDistribution,
    PredictionResult,
    SampleDistribution,
    SimplePredictorInputs,
)


def _csv_string(rows: Iterable[Iterable[object]]) -> str:
    """Serialize rows to a CSV string using Python's CSV writer."""

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def _input_rows(inputs: SimplePredictorInputs) -> list[list[object]]:
    """Return user inputs in long-form CSV rows."""

    return [
        ["input", "", "temperature_k", inputs.temperature_k],
        ["input", "", "tg_k", inputs.tg_k],
        ["input", "", "mass_ratio", inputs.mass_ratio],
        ["input", "", "rho_polymer", inputs.rho_polymer],
        ["input", "", "rho_solvent", inputs.rho_solvent],
        ["input", "", "polymer_xc", inputs.polymer_xc],
        ["input", "", "chris_category", inputs.chris_category],
        ["input", "", "smiles", inputs.smiles],
        ["input", "", "cas", inputs.cas],
        ["input", "", "n_samples_requested", inputs.n_samples],
    ]


def _prediction_distribution_rows(
    model: str,
    summary: PredictionDistribution,
) -> list[list[object]]:
    """Return one model-derived prediction-distribution summary in long form."""

    return [
        ["prediction_distribution", model, "n", summary.n],
        ["prediction_distribution", model, "mean_log10D", summary.mean],
        ["prediction_distribution", model, "sd_log10D", summary.sd],
        ["prediction_distribution", model, "p5_log10D", summary.p5],
        ["prediction_distribution", model, "p50_log10D", summary.p50],
        ["prediction_distribution", model, "p95_log10D", summary.p95],
        ["prediction_distribution", model, "mean_D_cm2_s", summary.D_mean],
        ["prediction_distribution", model, "sd_D_cm2_s", summary.D_sd],
        ["prediction_distribution", model, "p5_D_cm2_s", summary.D_p5],
        ["prediction_distribution", model, "p50_D_cm2_s", summary.D_p50],
        ["prediction_distribution", model, "p95_D_cm2_s", summary.D_p95],
    ]


def _sample_distribution_rows(
    model: str,
    summary: SampleDistribution,
) -> list[list[object]]:
    """Return one requested Monte Carlo distribution summary in long form."""

    return [
        ["sample_distribution", model, "n", summary.n],
        ["sample_distribution", model, "mean_log10D", summary.mean],
        ["sample_distribution", model, "sd_log10D", summary.sd],
        ["sample_distribution", model, "p5_log10D", summary.p5],
        ["sample_distribution", model, "p50_log10D", summary.p50],
        ["sample_distribution", model, "p95_log10D", summary.p95],
        ["sample_distribution", model, "mean_D_cm2_s", summary.D_mean],
        ["sample_distribution", model, "sd_D_cm2_s", summary.D_sd],
        ["sample_distribution", model, "p5_D_cm2_s", summary.D_p5],
        ["sample_distribution", model, "p50_D_cm2_s", summary.D_p50],
        ["sample_distribution", model, "p95_D_cm2_s", summary.D_p95],
    ]


def prediction_summary_csv(
    pred: PredictionResult,
    inputs: SimplePredictorInputs,
) -> str:
    """
    Create a compact v2 prediction-summary CSV.

    This file contains inputs, model provenance, the manuscript-defined
    headline consensus prediction, fixed model-derived distribution summaries,
    and requested Monte Carlo sample-distribution summaries.

    Raw 300/20/6000 prediction-distribution values are intentionally exported
    separately by ``prediction_distributions_csv``. Raw requested Monte Carlo
    values are exported separately by ``monte_carlo_samples_csv``.
    """

    rows: list[list[object]] = [
        ["section", "model", "field", "value"],
    ]

    rows.extend(_input_rows(inputs))

    rows.extend(
        [
            ["model_metadata", "", "wrapper_version", pred.wrapper_version],
            [
                "model_metadata",
                "QRF",
                "n_qrf_tree_predictions",
                pred.n_qrf_tree_predictions,
            ],
            [
                "model_metadata",
                "MLP",
                "n_mlp_ensemble_members",
                pred.n_mlp_ensemble_members,
            ],
            [
                "model_metadata",
                "Consensus",
                "n_consensus_pairs",
                pred.n_consensus_pairs,
            ],
            [
                "model_metadata",
                "",
                "n_samples_requested",
                pred.n_samples_requested,
            ],
            [
                "headline_prediction",
                "Consensus",
                "central_definition",
                "0.5 * QRF p50 + 0.5 * MLP p50 in log10(D) space",
            ],
            [
                "headline_prediction",
                "Consensus",
                "central_log10D",
                pred.consensus_central_log10D,
            ],
            [
                "headline_prediction",
                "Consensus",
                "central_D_cm2_s",
                pred.consensus_central_D,
            ],
            [
                "headline_prediction",
                "Consensus",
                "pairwise_distribution_p50_log10D",
                pred.consensus_distribution_p50,
            ],
            [
                "headline_prediction",
                "Consensus",
                "pairwise_distribution_p50_D_cm2_s",
                pred.consensus_distribution_D_p50,
            ],
        ]
    )

    rows.extend(_prediction_distribution_rows("QRF", pred.qrf))
    rows.extend(_prediction_distribution_rows("MLP", pred.mlp))
    rows.extend(_prediction_distribution_rows("Consensus", pred.consensus))

    rows.extend(_sample_distribution_rows("QRF", pred.qrf_sample))
    rows.extend(_sample_distribution_rows("MLP", pred.mlp_sample))
    rows.extend(_sample_distribution_rows("Consensus", pred.consensus_sample))

    rows.extend(
        [
            [
                "interpretation",
                "",
                "prediction_distribution_note",
                (
                    "Model-derived distributions are fixed by the trained models: "
                    "QRF n=300, MLP n=20, Consensus n=6000."
                ),
            ],
            [
                "interpretation",
                "",
                "sample_distribution_note",
                (
                    "Requested Monte Carlo samples are drawn with replacement from "
                    "the corresponding fixed model-derived distribution; therefore "
                    "sample percentiles may differ slightly from model-distribution "
                    "percentiles."
                ),
            ],
            [
                "interpretation",
                "Consensus",
                "consensus_distribution_note",
                (
                    "The 6000-member consensus distribution contains every "
                    "equal-weight pairing of 300 QRF tree predictions and 20 MLP "
                    "network predictions in log10(D) space."
                ),
            ],
            [
                "interpretation",
                "",
                "interval_note",
                (
                    "Prediction-distribution p5-p95 ranges are empirical "
                    "prediction-variability estimates, not formally calibrated "
                    "confidence or credible intervals."
                ),
            ],
        ]
    )

    return _csv_string(rows)


def prediction_distributions_csv(pred: PredictionResult) -> str:
    """
    Export the complete fixed model-derived prediction distributions.

    Expected row counts:
    - QRF: 300
    - MLP: 20
    - Consensus: 6000

    These are model-derived values and do not depend on the requested Monte
    Carlo sample size.
    """

    rows: list[list[object]] = [
        [
            "distribution_type",
            "model",
            "distribution_index",
            "log10D",
            "D_cm2_s",
        ]
    ]

    distributions = [
        (
            "QRF",
            pred.qrf_prediction_log10,
            pred.qrf_prediction_D,
        ),
        (
            "MLP",
            pred.mlp_prediction_log10,
            pred.mlp_prediction_D,
        ),
        (
            "Consensus",
            pred.consensus_prediction_log10,
            pred.consensus_prediction_D,
        ),
    ]

    for model, log_values, d_values in distributions:
        if len(log_values) != len(d_values):
            raise ValueError(
                f"{model} model-derived log10(D) and D arrays have different lengths."
            )

        for index, (log10d, d_value) in enumerate(
            zip(log_values, d_values),
            start=1,
        ):
            rows.append(
                [
                    "model_derived_prediction_distribution",
                    model,
                    index,
                    log10d,
                    d_value,
                ]
            )

    return _csv_string(rows)


def monte_carlo_samples_csv(pred: PredictionResult) -> str:
    """
    Export requested Monte Carlo samples for QRF, MLP, and consensus.

    Each model contributes ``pred.n_samples_requested`` rows. These are sampled
    with replacement from the corresponding fixed model-derived prediction
    distribution.
    """

    rows: list[list[object]] = [
        [
            "distribution_type",
            "model",
            "sample_index",
            "log10D",
            "D_cm2_s",
        ]
    ]

    samples = [
        ("QRF", pred.qrf_log10_samples, pred.qrf_D_samples),
        ("MLP", pred.mlp_log10_samples, pred.mlp_D_samples),
        (
            "Consensus",
            pred.consensus_log10_samples,
            pred.consensus_D_samples,
        ),
    ]

    for model, log_values, d_values in samples:
        if len(log_values) != len(d_values):
            raise ValueError(
                f"{model} Monte Carlo log10(D) and D arrays have different lengths."
            )

        if len(log_values) != pred.n_samples_requested:
            raise ValueError(
                f"{model} Monte Carlo array has {len(log_values)} rows; "
                f"expected {pred.n_samples_requested}."
            )

        for index, (log10d, d_value) in enumerate(
            zip(log_values, d_values),
            start=1,
        ):
            rows.append(
                [
                    "monte_carlo_sample_distribution",
                    model,
                    index,
                    log10d,
                    d_value,
                ]
            )

    return _csv_string(rows)


def prediction_summary_json(
    pred: PredictionResult,
    inputs: SimplePredictorInputs,
) -> str:
    """
    Create a compact JSON summary without raw distribution arrays.

    Use ``prediction_full_json`` when raw model-derived and Monte Carlo arrays
    are also required.
    """

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
        "model_metadata": {
            "wrapper_version": pred.wrapper_version,
            "n_qrf_tree_predictions": pred.n_qrf_tree_predictions,
            "n_mlp_ensemble_members": pred.n_mlp_ensemble_members,
            "n_consensus_pairs": pred.n_consensus_pairs,
            "n_samples_requested": pred.n_samples_requested,
        },
        "headline_consensus": {
            "definition": "0.5 * QRF p50 + 0.5 * MLP p50 in log10(D) space",
            "central_log10D": pred.consensus_central_log10D,
            "central_D_cm2_s": pred.consensus_central_D,
            "pairwise_distribution_p50_log10D": pred.consensus_distribution_p50,
            "pairwise_distribution_p50_D_cm2_s": (
                pred.consensus_distribution_D_p50
            ),
        },
        "prediction_distributions": {
            "qrf": asdict(pred.qrf),
            "mlp": asdict(pred.mlp),
            "consensus": asdict(pred.consensus),
        },
        "sample_distributions": {
            "qrf": asdict(pred.qrf_sample),
            "mlp": asdict(pred.mlp_sample),
            "consensus": asdict(pred.consensus_sample),
        },
        "interpretation": {
            "prediction_distribution": (
                "Model-derived distributions are fixed by the trained models: "
                "QRF n=300, MLP n=20, Consensus n=6000."
            ),
            "monte_carlo_sampling": (
                "Requested Monte Carlo values are sampled with replacement from "
                "the fixed model-derived distributions."
            ),
            "consensus_distribution": (
                "The 6000-member consensus distribution consists of all "
                "300 x 20 equal-weight QRF-tree/MLP-network pairings in "
                "log10(D) space."
            ),
            "variability": (
                "Reported p5-p95 prediction-distribution ranges are empirical "
                "prediction-variability estimates rather than formally calibrated "
                "confidence or credible intervals."
            ),
        },
    }

    return json.dumps(payload, indent=2)


def prediction_full_json(
    pred: PredictionResult,
    inputs: SimplePredictorInputs,
) -> str:
    """
    Create a complete JSON export including all raw arrays.

    This can be substantially larger than ``prediction_summary_json`` because
    it contains 300 QRF, 20 MLP, 6000 consensus model-derived values plus all
    requested Monte Carlo samples for the three predictors.
    """

    payload = json.loads(prediction_summary_json(pred=pred, inputs=inputs))

    payload["raw_prediction_distributions"] = {
        "qrf": {
            "log10D": pred.qrf_prediction_log10,
            "D_cm2_s": pred.qrf_prediction_D,
        },
        "mlp": {
            "log10D": pred.mlp_prediction_log10,
            "D_cm2_s": pred.mlp_prediction_D,
        },
        "consensus": {
            "log10D": pred.consensus_prediction_log10,
            "D_cm2_s": pred.consensus_prediction_D,
        },
    }

    payload["raw_monte_carlo_samples"] = {
        "qrf": {
            "log10D": pred.qrf_log10_samples,
            "D_cm2_s": pred.qrf_D_samples,
        },
        "mlp": {
            "log10D": pred.mlp_log10_samples,
            "D_cm2_s": pred.mlp_D_samples,
        },
        "consensus": {
            "log10D": pred.consensus_log10_samples,
            "D_cm2_s": pred.consensus_D_samples,
        },
    }

    return json.dumps(payload, indent=2)
