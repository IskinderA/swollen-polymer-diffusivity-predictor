"""
Download utilities for the Swollen Polymer Diffusivity Predictor.
"""


def prediction_summary_csv(
    qrf_p5: float,
    qrf_p50: float,
    qrf_p95: float,
    mlp_p5: float,
    mlp_p50: float,
    mlp_p95: float,
) -> str:
    """Create a CSV string for the prediction summary."""

    return (
        "model,p5,p50,p95\n"
        f"QRF,{qrf_p5:.3f},{qrf_p50:.3f},{qrf_p95:.3f}\n"
        f"MLP,{mlp_p5:.3f},{mlp_p50:.3f},{mlp_p95:.3f}\n"
    )
