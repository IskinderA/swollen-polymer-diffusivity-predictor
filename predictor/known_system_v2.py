"""
Known-system lookup utilities for Swollen Polymer Diffusivity Predictor v2.

This module is intentionally parallel to ``predictor.known_system`` but reads
the v2 Known System database:

    data/known_system_database_v2.csv

The public helper API is kept unchanged so the v2 Streamlit application can
switch to this module without changing its dropdown logic.
"""

from pathlib import Path

import pandas as pd


DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "known_system_database_v2.csv"
)


def load_known_system_database() -> pd.DataFrame:
    """Load the curated v2 Known System database."""
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            "The v2 Known System database was not found:\n"
            f"{DATABASE_PATH}\n\n"
            "Generate it first with:\n"
            "python scripts/build_known_system_database_v2.py"
        )

    return pd.read_csv(DATABASE_PATH)


def get_polymers() -> list[str]:
    """Return available polymer names."""
    db = load_known_system_database()
    return sorted(db["Polymer_Name"].dropna().unique().tolist())


def get_solvents(polymer_name: str) -> list[str]:
    """Return available solvents for a selected polymer."""
    db = load_known_system_database()
    subset = db[db["Polymer_Name"] == polymer_name]
    return sorted(subset["Solvent_Name"].dropna().unique().tolist())


def get_solutes(polymer_name: str, solvent_name: str) -> list[str]:
    """Return available solutes for a selected polymer-solvent pair."""
    db = load_known_system_database()
    subset = db[
        (db["Polymer_Name"] == polymer_name)
        & (db["Solvent_Name"] == solvent_name)
    ]
    return sorted(subset["Solute_Name"].dropna().unique().tolist())


def get_system(
    polymer_name: str,
    solvent_name: str,
    solute_name: str,
) -> dict:
    """Return metadata for a selected polymer-solvent-solute system."""

    db = load_known_system_database()

    def norm(value) -> str:
        return str(value).strip().lower()

    subset = db[
        (db["Polymer_Name"].apply(norm) == norm(polymer_name))
        & (db["Solvent_Name"].apply(norm) == norm(solvent_name))
        & (db["Solute_Name"].apply(norm) == norm(solute_name))
    ]

    if subset.empty:
        raise ValueError(
            f"No v2 known system found for polymer={polymer_name}, "
            f"solvent={solvent_name}, solute={solute_name}"
        )

    return subset.iloc[0].to_dict()
