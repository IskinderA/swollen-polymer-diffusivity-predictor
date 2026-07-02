"""
Descriptor utilities for the Swollen Polymer Diffusivity Predictor.

This module will eventually compute the full published feature set.
For now, it returns placeholder derived descriptors for interface testing.
"""

from predictor.predict import SimplePredictorInputs


def compute_descriptor_placeholders(inputs: SimplePredictorInputs) -> dict:
    """Return placeholder computed descriptors for interface testing."""

    q_w_minus_1 = inputs.mass_ratio - 1.0

    return {
        "T/Tg": inputs.temperature_k / inputs.tg_k if inputs.tg_k else None,
        "Qw - 1": q_w_minus_1,
        "phi1": None,
        "omega1": None,
        "Qv - 1": None,
        "MW": None,
        "ring_count": None,
        "aromatic_ring_count": None,
        "rotatable_bond_count": None,
    }
