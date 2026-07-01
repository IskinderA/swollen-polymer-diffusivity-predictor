"""
Scientific interpretation utilities for the
Swollen Polymer Diffusivity Predictor.
"""


def classify_model_agreement(delta_log10: float) -> tuple[str, str]:
    """
    Classify agreement between QRF and MLP median predictions.
    """

    if delta_log10 < 0.25:
        return (
            "Excellent agreement",
            "The QRF and MLP prediction distributions are closely aligned.",
        )

    if delta_log10 < 0.75:
        return (
            "Moderate agreement",
            "The QRF and MLP predictions differ moderately. Interpret the prediction with some caution.",
        )

    return (
        "Large disagreement",
        "The QRF and MLP predictions differ substantially. Predictions should be interpreted cautiously and experimental confirmation is recommended.",
    )
