# Swollen Polymer Diffusivity Predictor

# Architecture Specification

**Version:** 1.0

**Status:** Internal development document

---

## 1. Purpose

This document defines the repository structure and software architecture for the provisional companion predictor accompanying the manuscript

> **Predicting Solute Diffusivity in Swollen Polymer Systems Using Quantile Random Forests and Neural Networks**

The purpose of this architecture is to keep the user interface, prediction engine, published models, example systems, and documentation clearly separated.

---

## 2. Repository Structure

```text
swollen-polymer-diffusivity-predictor/
│
├── README.md
├── LICENSE
├── requirements.txt
├── app.py
│
├── predictor/
│   ├── __init__.py
│   ├── predict.py
│   ├── descriptors.py
│   ├── known_system.py
│   ├── custom_system.py
│   ├── interpretation.py
│   ├── downloads.py
│   └── utilities.py
│
├── models/
│   ├── qrf/
│   ├── mlp/
│   └── metadata/
│
├── data/
│   ├── known_system_database.csv
│   ├── polymer_properties.csv
│   ├── solvent_properties.csv
│   └── example_systems.csv
│
├── examples/
│   ├── moderate_rubbery.csv
│   ├── moderate_glassy.csv
│   └── pebax4033_sv13.csv
│
├── assets/
│   ├── screenshots/
│   └── figures/
│
├── tests/
│   ├── test_predictor.py
│   ├── test_descriptors.py
│   └── test_examples.py
│
└── docs/
    ├── design/
    │   ├── ProductDesignSpecification_v1.0.md
    │   ├── Wireframes_v1.0.md
    │   └── Architecture_v1.0.md
    │
    └── release/
        └── ReleaseNotes_v1.0.md
```

---

## 3. Architectural Principle

The Streamlit application shall function only as the user interface.

The scientific prediction logic shall reside inside the `predictor/` package and shall be reusable outside the web application.

This separation allows the same prediction engine to support:

- the online companion predictor
- command-line examples
- notebooks
- future batch-prediction workflows
- future application interfaces

---

## 4. High-Level Flow

```text
User Interface
(app.py)
    │
    ▼
Input Mode
Known System or Custom System
    │
    ▼
Simplified Input Dictionary
    │
    ▼
predictor.predict()
    │
    ├── descriptor generation
    ├── swelling calculation
    ├── molecular descriptor calculation
    ├── feature-row construction
    ├── QRF prediction distribution
    └── MLP ensemble prediction distribution
    │
    ▼
Prediction Summary
    │
    ├── QRF 5th / 50th / 95th percentiles
    ├── MLP 5th / 50th / 95th percentiles
    ├── model agreement summary
    ├── interpretation text
    └── downloadable CSV / JSON
```

---

## 5. Module Responsibilities

### `app.py`

User interface only.

Responsibilities:

- page layout
- mode selection
- input widgets
- display of prediction summary
- display of disclaimer
- calls to the prediction engine

`app.py` shall not contain descriptor-generation or model-prediction logic.

---

### `predictor/predict.py`

Primary prediction interface.

Responsibilities:

- accept simplified user inputs
- call descriptor generation
- call QRF and MLP prediction functions
- assemble unified prediction output
- return structured result objects for display and export

---

### `predictor/descriptors.py`

Descriptor generation.

Responsibilities:

- calculate swelling descriptors from mass ratio and density inputs
- calculate molecular descriptors from SMILES
- resolve optional CAS inputs when supported
- build full model feature rows
- expose computed descriptors for advanced users

---

### `predictor/known_system.py`

Known-system mode support.

Responsibilities:

- load curated polymer, solvent, and solute lookup tables
- populate simplified interface inputs from dropdown selections
- provide example systems corresponding to manuscript use cases

---

### `predictor/custom_system.py`

Custom-system mode support.

Responsibilities:

- validate direct user inputs
- convert user-entered quantities into the simplified prediction interface
- flag missing or physically inconsistent values

---

### `predictor/interpretation.py`

Scientific interpretation.

Responsibilities:

- compute QRF/MLP agreement
- assign qualitative agreement labels
- generate user-facing interpretation text
- avoid calibrated confidence language

---

### `predictor/downloads.py`

Export utilities.

Responsibilities:

- generate CSV output
- generate JSON output
- include user inputs, computed descriptors, model outputs, model version, and timestamp

---

### `models/`

Published model artifacts.

Responsibilities:

- store final QRF model
- store final MLP ensemble models
- store preprocessing pipelines where applicable
- store model metadata

---

### `data/`

Reference lookup data.

Responsibilities:

- polymer properties
- solvent properties
- known systems
- example systems

This folder shall not contain the full manuscript training dataset unless explicitly approved for release.

---

### `examples/`

Reproducible examples.

Responsibilities:

- provide lightweight input files for representative systems
- support README examples
- support regression tests

Initial examples:

- moderate rubbery system
- moderate glassy system
- PEBAX-like benchmark

---

### `tests/`

Regression and consistency tests.

Responsibilities:

- verify descriptor calculations
- verify example predictions
- verify output schema
- verify that future code changes do not alter published example outputs unexpectedly

---

## 6. Model Metadata

The repository shall include model metadata sufficient to identify the published model version.

Example metadata fields:

```json
{
  "model_version": "11.5",
  "manuscript_title": "Predicting Solute Diffusivity in Swollen Polymer Systems Using Quantile Random Forests and Neural Networks",
  "training_measurements": 2032,
  "training_polymers": 27,
  "training_solvents": 50,
  "training_solutes": 47,
  "qrf_trees": 300,
  "mlp_ensemble_members": 20,
  "status": "provisional companion predictor"
}
```

---

## 7. Implementation Roadmap

### Stage 1

Create repository structure and placeholder modules.

### Stage 2

Implement Streamlit interface using placeholder predictions.

### Stage 3

Implement simplified input validation.

### Stage 4

Connect QRF prediction engine.

### Stage 5

Connect MLP ensemble prediction engine.

### Stage 6

Implement sampling and downloads.

### Stage 7

Add example systems and regression tests.

### Stage 8

Finalize README, disclaimer, and manuscript/SI references.

---

## 8. Design Constraint

The predictor shall remain intentionally focused.

Version 1.0 shall not include:

- model retraining
- feature-ablation tools
- upload of arbitrary training datasets
- regulatory decision logic
- claims of FDA qualification
- calibrated confidence scores

These constraints preserve the intended role of the predictor as a provisional companion implementation accompanying the manuscript.
