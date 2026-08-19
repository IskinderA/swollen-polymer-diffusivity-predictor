"""
Build the v2 Known System database for the Streamlit deployment.

Revision b
----------
This version deliberately preserves the existing v1 Known System database for
all original systems and appends only the seven additional PEBAX 4033/hexane
systems introduced for the v2 2039-observation deployment models.

Why preserve v1?
The original 2032-observation literature dataset can contain multiple reported
crystallinity values for the same nominal polymer-solvent-solute identity.
The v1 Known System database already embodies the curated behavior used by the
public application. Rebuilding every original system by forcing one new
aggregate crystallinity value could silently change existing UI behavior.

Therefore v2 uses:
    existing v1 Known System database
    + seven new PEBAX/hexane systems
    + solvent normalization "hexanes" -> "hexane"

Intended location:
    ML_swollen/predictor_github_08192026/
        scripts/build_known_system_database_v2.py

Primary source for the seven new systems:
    ML_swollen/version_11pt5v6/deployment_training_data.csv

Identity/SMILES lookup:
    ML_swollen/version_2/D_unswollen.csv

Output:
    data/known_system_database_v2.csv

Audit:
    data/known_system_database_v2_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ML_SWOLLEN_DIR = REPO_DIR.parent

V1_DATABASE_PATH = REPO_DIR / "data" / "known_system_database.csv"

DEPLOYMENT_DATA_PATH = (
    ML_SWOLLEN_DIR
    / "version_11pt5v6"
    / "deployment_training_data.csv"
)

UNSWOLLEN_PATH = ML_SWOLLEN_DIR / "version_2" / "D_unswollen.csv"

OUTPUT_PATH = REPO_DIR / "data" / "known_system_database_v2.csv"
AUDIT_PATH = REPO_DIR / "data" / "known_system_database_v2_audit.json"


EXPECTED_DEPLOYMENT_ROWS = 2039
EXPECTED_ADDED_ROWS = 7

EXPECTED_ADDED_PEBAX_SOLUTES = {
    "BBOT",
    "BHT",
    "Erucamide",
    "Irganox 1010",
    "Irganox 1076",
    "Octabenzone",
    "Tinuvin 326",
}

OUTPUT_COLUMNS = [
    "Polymer_Name",
    "Solvent_Name",
    "Solute_Name",
    "Tg_K",
    "Polymer_Xc",
    "CHRIS_Category",
    "Solute_CAS",
    "Solute_InChIKey",
    "Solute_SMILES",
    "solute_MW",
    "solute_ring_count",
    "solute_aromatic_ring_count",
    "solute_rotatable_bond_count",
]


# =============================================================================
# Helpers
# =============================================================================

def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found:\n{path}")


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns:\n{list(frame.columns)}"
        )


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    return text


def normalize_solvent_name(value) -> str | None:
    text = clean_text(value)
    if text is None:
        return None

    if text.lower() in {"hexane", "hexanes"}:
        return "hexane"

    return text


def normalize_solute_name(value) -> str | None:
    text = clean_text(value)
    if text is None:
        return None

    key = text.lower()
    key = key.replace("-", " ")
    key = key.replace(",", "")
    key = key.replace("(", " ")
    key = key.replace(")", " ")
    key = key.replace("/", " ")
    key = " ".join(key.split())

    aliases = {
        "methylene chloride": "dichloromethane",
        "p dioxane": "14 dioxane",
        "p xylene": "pxylene",
        "co2": "carbon dioxide",
        "n2": "nitrogen",
        "neohexane": "22 dimethylbutane",
        "bht": "26 di tert butyl 4 methylphenol",
        "tinuvin 326": "bumetrizole",
    }

    return aliases.get(key, key)


def normalize_cas(value) -> str | None:
    text = clean_text(value)
    if text is None:
        return None

    # Known Excel date conversion encountered for Tinuvin 326.
    if text == "11/5/3896":
        return "3896-11-5"

    return text


def first_nonmissing(series: pd.Series):
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() != "nan":
                return value
    return np.nan


def first_numeric(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No valid numeric value found for {series.name}.")
    return float(values.iloc[0])


# =============================================================================
# Load sources
# =============================================================================

def load_v1_database() -> pd.DataFrame:
    require_file(V1_DATABASE_PATH, "v1 Known System database")

    frame = pd.read_csv(V1_DATABASE_PATH)
    frame.columns = frame.columns.astype(str).str.strip()

    require_columns(frame, OUTPUT_COLUMNS, "v1 Known System database")

    # Preserve row ordering and metadata exactly; only normalize the solvent
    # spelling chosen for v2.
    frame = frame[OUTPUT_COLUMNS].copy()
    frame["Solvent_Name"] = frame["Solvent_Name"].map(normalize_solvent_name)

    return frame


def load_deployment_data() -> pd.DataFrame:
    require_file(
        DEPLOYMENT_DATA_PATH,
        "v11pt5v6 deployment training data",
    )

    frame = pd.read_csv(DEPLOYMENT_DATA_PATH)
    frame.columns = frame.columns.astype(str).str.strip()

    required = [
        "Polymer_Name",
        "Solvent_Name",
        "Solute_Name",
        "Tg_K",
        "Polymer_Xc",
        "CHRIS Category",
        "solute_MW",
        "solute_ring_count",
        "solute_aromatic_ring_count",
        "solute_rotatable_bond_count",
        "added_pebax_hexanes",
    ]

    require_columns(
        frame,
        required,
        "v11pt5v6 deployment training data",
    )

    if len(frame) != EXPECTED_DEPLOYMENT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DEPLOYMENT_ROWS} deployment rows, "
            f"found {len(frame)}."
        )

    return frame


def build_unswollen_lookup() -> pd.DataFrame:
    require_file(UNSWOLLEN_PATH, "D_unswollen identity lookup")

    frame = pd.read_csv(UNSWOLLEN_PATH)
    frame.columns = frame.columns.astype(str).str.strip()

    required = [
        "Solute_Name",
        "Solute_CAS",
        "Solute_InChIKey",
        "Solute_SMILES",
    ]
    require_columns(frame, required, "D_unswollen identity lookup")

    lookup = frame[required].copy()
    lookup["Solute_Key"] = lookup["Solute_Name"].map(normalize_solute_name)
    lookup["Solute_CAS"] = lookup["Solute_CAS"].map(normalize_cas)

    rows = []
    for key, group in lookup.dropna(subset=["Solute_Key"]).groupby(
        "Solute_Key",
        sort=False,
    ):
        rows.append(
            {
                "Solute_Key": key,
                "lookup_Solute_CAS": first_nonmissing(group["Solute_CAS"]),
                "lookup_Solute_InChIKey": first_nonmissing(
                    group["Solute_InChIKey"]
                ),
                "lookup_Solute_SMILES": first_nonmissing(
                    group["Solute_SMILES"]
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Build the seven new selectable systems
# =============================================================================

def select_added_pebax_rows(deployment: pd.DataFrame) -> pd.DataFrame:
    added_flag = (
        deployment["added_pebax_hexanes"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    added = deployment.loc[added_flag].copy()

    if len(added) != EXPECTED_ADDED_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_ADDED_ROWS} rows with "
            f"added_pebax_hexanes=True, found {len(added)}."
        )

    polymer_values = set(
        added["Polymer_Name"].dropna().astype(str).str.strip().str.lower()
    )
    if polymer_values != {"pebax_4033"}:
        raise ValueError(
            "The seven deployment augmentation rows are not all PEBAX_4033. "
            f"Found: {sorted(polymer_values)}"
        )

    added["Solvent_Name"] = added["Solvent_Name"].map(normalize_solvent_name)

    solvent_values = set(
        added["Solvent_Name"].dropna().astype(str).str.strip().str.lower()
    )
    if solvent_values != {"hexane"}:
        raise ValueError(
            "The seven deployment augmentation rows are not all normalized "
            f"to hexane. Found: {sorted(solvent_values)}"
        )

    solutes = set(added["Solute_Name"].dropna().astype(str))
    if solutes != EXPECTED_ADDED_PEBAX_SOLUTES:
        raise ValueError(
            "Unexpected seven-solute PEBAX augmentation set.\n"
            f"Expected: {sorted(EXPECTED_ADDED_PEBAX_SOLUTES)}\n"
            f"Found:    {sorted(solutes)}"
        )

    return added


def build_added_systems(
    deployment: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    added = select_added_pebax_rows(deployment)

    # One deployment observation is expected per added solute. Grouping is
    # retained defensively so a future source table can contain duplicates
    # without silently producing duplicate dropdown entries.
    rows = []

    for solute_name, group in added.groupby("Solute_Name", sort=True):
        solute_key = normalize_solute_name(solute_name)

        match = lookup[lookup["Solute_Key"] == solute_key]
        if match.empty:
            raise ValueError(
                f"No identity/SMILES lookup found for added solute {solute_name!r}."
            )

        identity = match.iloc[0]

        deployment_cas = np.nan
        if "solute CAS" in group.columns:
            deployment_cas = first_nonmissing(
                group["solute CAS"].map(normalize_cas)
            )

        cas = (
            deployment_cas
            if pd.notna(deployment_cas)
            else identity["lookup_Solute_CAS"]
        )

        row = {
            "Polymer_Name": first_nonmissing(group["Polymer_Name"]),
            "Solvent_Name": "hexane",
            "Solute_Name": solute_name,
            "Tg_K": first_numeric(group["Tg_K"]),
            "Polymer_Xc": first_numeric(group["Polymer_Xc"]),
            "CHRIS_Category": first_nonmissing(group["CHRIS Category"]),
            "Solute_CAS": cas,
            "Solute_InChIKey": identity["lookup_Solute_InChIKey"],
            "Solute_SMILES": identity["lookup_Solute_SMILES"],
            "solute_MW": first_numeric(group["solute_MW"]),
            "solute_ring_count": first_numeric(group["solute_ring_count"]),
            "solute_aromatic_ring_count": first_numeric(
                group["solute_aromatic_ring_count"]
            ),
            "solute_rotatable_bond_count": first_numeric(
                group["solute_rotatable_bond_count"]
            ),
        }

        rows.append(row)

    added_systems = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    if len(added_systems) != EXPECTED_ADDED_ROWS:
        raise ValueError(
            f"Expected seven unique added systems, found {len(added_systems)}."
        )

    if added_systems["Solute_SMILES"].isna().any():
        missing = added_systems.loc[
            added_systems["Solute_SMILES"].isna(),
            "Solute_Name",
        ].tolist()
        raise ValueError(
            f"Missing SMILES for added PEBAX solutes: {missing}"
        )

    return added_systems


# =============================================================================
# Combine and audit
# =============================================================================

def build_v2_database() -> tuple[pd.DataFrame, dict]:
    v1 = load_v1_database()
    deployment = load_deployment_data()
    lookup = build_unswollen_lookup()
    added = build_added_systems(deployment, lookup)

    # Guard against accidentally appending a solute already present as the same
    # PEBAX/hexane selectable system in v1.
    v1_keys = set(
        zip(
            v1["Polymer_Name"].astype(str).str.strip().str.lower(),
            v1["Solvent_Name"].astype(str).str.strip().str.lower(),
            v1["Solute_Name"].astype(str).str.strip().str.lower(),
        )
    )

    overlap = []
    for _, row in added.iterrows():
        key = (
            str(row["Polymer_Name"]).strip().lower(),
            str(row["Solvent_Name"]).strip().lower(),
            str(row["Solute_Name"]).strip().lower(),
        )
        if key in v1_keys:
            overlap.append(row["Solute_Name"])

    if overlap:
        raise ValueError(
            "One or more added PEBAX/hexane systems already exist in the v1 "
            f"Known System database: {sorted(overlap)}"
        )

    known = pd.concat([v1, added], ignore_index=True)

    # Keep original v1 relative order intact; place added systems afterward,
    # sorted alphabetically among themselves.
    known = known[OUTPUT_COLUMNS].copy()

    pebax_hexane = known[
        (
            known["Polymer_Name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("pebax_4033")
        )
        & (
            known["Solvent_Name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("hexane")
        )
    ].copy()

    pebax_solutes = set(pebax_hexane["Solute_Name"].dropna().astype(str))
    missing = EXPECTED_ADDED_PEBAX_SOLUTES - pebax_solutes
    if missing:
        raise ValueError(
            "The final PEBAX_4033/hexane dropdown would be missing: "
            f"{sorted(missing)}"
        )

    if known["Solvent_Name"].astype(str).str.lower().eq("hexanes").any():
        raise ValueError('The solvent label "hexanes" survived normalization.')

    audit = {
        "strategy": (
            "Preserve existing v1 Known System rows and append the seven new "
            "PEBAX_4033/hexane systems from the 2039-observation v2 deployment "
            "training dataset."
        ),
        "v1_database": str(V1_DATABASE_PATH),
        "deployment_training_data": str(DEPLOYMENT_DATA_PATH),
        "identity_lookup": str(UNSWOLLEN_PATH),
        "output": str(OUTPUT_PATH),
        "v1_rows": int(len(v1)),
        "added_v2_rows": int(len(added)),
        "final_rows": int(len(known)),
        "final_unique_polymers": int(
            known["Polymer_Name"].nunique(dropna=True)
        ),
        "final_unique_solvents": int(
            known["Solvent_Name"].nunique(dropna=True)
        ),
        "final_unique_solutes": int(
            known["Solute_Name"].nunique(dropna=True)
        ),
        "solvent_normalization": {"hexanes": "hexane"},
        "required_added_pebax_solutes": sorted(
            EXPECTED_ADDED_PEBAX_SOLUTES
        ),
        "pebax_4033_hexane_selectable_solutes": sorted(pebax_solutes),
        "added_solutes_have_smiles": bool(
            added["Solute_SMILES"].notna().all()
        ),
        "note": (
            "Existing v1 rows are intentionally not re-aggregated. This avoids "
            "changing existing Known System metadata where the source literature "
            "contains multiple crystallinity values for the same nominal system."
        ),
    }

    return known, audit


def main() -> int:
    known, audit = build_v2_database()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    known.to_csv(OUTPUT_PATH, index=False)

    with AUDIT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    pebax_hexane = known[
        (
            known["Polymer_Name"].astype(str).str.strip().str.lower()
            == "pebax_4033"
        )
        & (
            known["Solvent_Name"].astype(str).str.strip().str.lower()
            == "hexane"
        )
    ]

    print("Known System database v2 revision b completed successfully.")
    print(f"Preserved v1 rows: {audit['v1_rows']}")
    print(f"Added PEBAX/hexane rows: {audit['added_v2_rows']}")
    print(f"Final database rows: {audit['final_rows']}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Audit:  {AUDIT_PATH}")
    print()
    print("PEBAX_4033 / hexane selectable solutes:")
    for solute in sorted(pebax_hexane["Solute_Name"].astype(str).tolist()):
        marker = (
            " [added v2]"
            if solute in EXPECTED_ADDED_PEBAX_SOLUTES
            else ""
        )
        print(f"  - {solute}{marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
