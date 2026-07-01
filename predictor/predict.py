"""
Prediction interface for the Swollen Polymer Diffusivity Predictor.

This module will eventually call the published QRF and MLP ensemble models.
For now, it returns placeholder predictions used to test the interface.
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


def predict_placeholder() -> PredictionResult:
    """Return placeholder prediction distributions for interface testing."""

    return PredictionResult(
        qrf=PredictionDistribution(
            p5=-8.20,
            p50=-7.10,
            p95=-5.90,
        ),
        mlp=PredictionDistribution(
            p5=-7.00,
            p50=-6.55,
            p95=-6.10,
        ),
    )
