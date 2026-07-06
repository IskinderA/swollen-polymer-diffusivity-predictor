from pathlib import Path
import pandas as pd

def normalize_solute_name(x):
    if pd.isna(x):
        return None

    s = str(x).strip().lower()
    s = s.replace("-", " ")
    s = s.replace(",", "")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    s = s.replace("/", " ")
    s = " ".join(s.split())
    return s


SOLUTE_NAME_ALIASES = {
    "methylene chloride": "dichloromethane",
    "p dioxane": "14 dioxane",
    "p xylene": "pxylene",
    "co2": "carbon dioxide",
    "n2": "nitrogen",
    "neohexane": "22 dimethylbutane",
    "bht": "26 di tert butyl 4 methylphenol",
}


def apply_solute_alias(x):
    if x is None:
        return None
    return SOLUTE_NAME_ALIASES.get(x, x)


REFINEMENT_PATH = Path("../data/model_refinement_dataset.csv")
UNSWOLLEN_PATH = Path("../data/D_unswollen.csv")
OUTPUT_PATH = Path("../data/known_system_database.csv")


REFINEMENT_COLUMNS = [
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
]

UNSWOLLEN_COLUMNS = [
    "Solute_Name",
    "Solute_CAS",
    "Solute_InChIKey",
    "Solute_SMILES",
]


def main() -> None:
    refinement = pd.read_csv(REFINEMENT_PATH)
    unswollen = pd.read_csv(UNSWOLLEN_PATH)

    missing_refinement = [c for c in REFINEMENT_COLUMNS if c not in refinement.columns]
    missing_unswollen = [c for c in UNSWOLLEN_COLUMNS if c not in unswollen.columns]

    if missing_refinement:
        raise ValueError(f"Missing columns in {REFINEMENT_PATH}: {missing_refinement}")

    if missing_unswollen:
        raise ValueError(f"Missing columns in {UNSWOLLEN_PATH}: {missing_unswollen}")

    base = (
        refinement[REFINEMENT_COLUMNS]
        .drop_duplicates()
        .rename(columns={"CHRIS Category": "CHRIS_Category"})
    )

    base["Solute_Key"] = base["Solute_Name"].apply(normalize_solute_name).apply(apply_solute_alias)

    smiles_lookup = unswollen[UNSWOLLEN_COLUMNS].copy()
    smiles_lookup["Solute_Key"] = (
        smiles_lookup["Solute_Name"]
        .apply(normalize_solute_name)
        .apply(apply_solute_alias)
    )

    smiles_lookup = (
        smiles_lookup
        .dropna(subset=["Solute_Key"])
        .drop_duplicates(subset=["Solute_Key"])
        .drop(columns=["Solute_Name"])
    )

    known = base.merge(
        smiles_lookup,
        on="Solute_Key",
        how="left",
    )

    known = (
        known.sort_values(["Polymer_Name", "Solvent_Name", "Solute_Name"])
        .reset_index(drop=True)
    )

    column_order = [
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

    known = known[column_order]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    known.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(known)} rows to {OUTPUT_PATH}")
    print(f"Rows missing SMILES: {known['Solute_SMILES'].isna().sum()}")
    print(f"Unique polymers: {known['Polymer_Name'].nunique()}")
    print(f"Unique solvents: {known['Solvent_Name'].nunique()}")
    print(f"Unique solutes: {known['Solute_Name'].nunique()}")


if __name__ == "__main__":
    main()