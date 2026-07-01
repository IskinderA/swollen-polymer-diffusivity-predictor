"""
Prediction interface for the Swollen Polymer Diffusivity Predictor.

This module will eventually call the published QRF and MLP ensemble models.
For now, it returns placeholder predictions used to test the interface.
"""


def predict_placeholder() -> dict:
    """Return placeholder prediction distributions for interface testing."""

    return {
        "qrf": {
            "p5": -8.20,
            "p50": -7.10,
            "p95": -5.90,
        },
        "mlp": {
            "p5": -7.00,
            "p50": -6.55,
            "p95": -6.10,
        },
    }
