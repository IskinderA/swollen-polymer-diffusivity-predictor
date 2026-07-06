"""
Known-system lookup utilities for the Swollen Polymer Diffusivity Predictor.
"""

from pathlib import Path

import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "known_system_database.csv"


def load_known_system_database() -> pd.DataFrame:
    """Load the curated known-system database."""
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


def get_system(polymer_name: str, solvent_name: str, solute_name: str) -> dict:
    """Return metadata for a selected polymer-solvent-solute system."""
    db = load_known_system_database()
    subset = db[
        (db["Polymer_Name"] == polymer_name)
        & (db["Solvent_Name"] == solvent_name)
        & (db["Solute_Name"] == solute_name)
    ]

    if subset.empty:
        raise ValueError(
            f"No known system found for polymer={polymer_name}, "
            f"solvent={solvent_name}, solute={solute_name}"
        )

    row = subset.iloc[0].to_dict()
    return row