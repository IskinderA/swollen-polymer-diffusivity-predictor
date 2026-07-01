# Swollen Polymer Diffusivity Predictor

# Wireframes

**Version:** 1.0

This document illustrates the intended user workflow and interface layout for the provisional companion predictor accompanying

> **Predicting Solute Diffusivity in Swollen Polymer Systems Using Quantile Random Forests and Neural Networks**

The wireframes describe user interactions and scientific workflows rather than graphical appearance. The objective is to design a simple, transparent, and trustworthy scientific instrument rather than a conventional software application.

---

# Overall User Workflow

```
                         Home
                           │
        ┌──────────────────┴──────────────────┐
        │                  │                  │
        │                  │                  │
 Known System        Custom System      Example Systems
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                 Prediction Summary
                           │
                 Advanced Details
                           │
                           ▼
                 Export Results
```

---

# Screen 1 — Home

## Goal

Allow users to immediately understand the purpose of the predictor and select an appropriate prediction mode.

---

```
--------------------------------------------------------

Swollen Polymer Diffusivity Predictor

A provisional companion predictor accompanying

Predicting Solute Diffusivity in Swollen Polymer Systems Using Quantile Random Forests and Neural Networks

--------------------------------------------------------

Choose Prediction Mode

[ Known System ]

Use curated polymer–solvent–solute systems.

--------------------------------------------------------

[ Custom System ]

Enter experimentally measured descriptors directly.

--------------------------------------------------------

[ Example Systems ]

Explore representative examples from the manuscript.

--------------------------------------------------------

About

Documentation

Citation

Research Use Disclaimer

--------------------------------------------------------
```

---

# Screen 2 — Known System

## Goal

Allow users to obtain predictions with minimal input while automatically populating all remaining model descriptors.

---

```
--------------------------------------------------------

Known System

Polymer

▼

Solvent

▼

Solute

▼

Temperature (K)

____________

Swollen / Dry Mass Ratio

____________

--------------------------------------------------------

[ Predict ]

--------------------------------------------------------
```

Internally populated quantities (hidden by default):

- Tg
- Polymer density
- Solvent density
- Polymer crystallinity
- CHRIS category
- SMILES
- Molecular descriptors
- Swelling descriptors

---

# Screen 3 — Custom System

## Goal

Provide direct access to the simplified prediction interface used by the published machine-learning models.

---

```
--------------------------------------------------------

Custom System

Temperature (K)

____________

Glass-transition temperature (K)

____________

Swollen / Dry Mass Ratio

____________

Polymer Density

____________

Solvent Density

____________

Polymer Crystallinity

____________

CHRIS Category

▼

SMILES

____________________________

(or CAS Number)

____________________________

--------------------------------------------------------

[ Predict ]

--------------------------------------------------------
```

The predictor computes all remaining descriptors internally.

---

# Screen 4 — Prediction Summary

## Goal

Present predictions in a scientifically interpretable manner.

---

```
--------------------------------------------------------

Prediction Summary

Inputs

✓ Polymer

✓ Solvent

✓ Solute

✓ Temperature

✓ Swelling

--------------------------------------------------------

QRF Prediction Distribution

5%

50%

95%

--------------------------------------------------------

MLP Ensemble Prediction Distribution

5%

50%

95%

--------------------------------------------------------

Model Agreement

Excellent

Difference between medians:

0.08 log units

--------------------------------------------------------

Interpretation

The QRF and MLP prediction distributions are in
excellent agreement.

This behavior is consistent with systems reasonably
represented within the published training domain.

--------------------------------------------------------

Download CSV

Download JSON

--------------------------------------------------------
```

---

# Screen 5 — Advanced Details

## Goal

Allow interested users to inspect the internally generated model descriptors without increasing complexity for routine users.

Collapsed by default.

---

```
Show Computed Descriptors ▼

--------------------------------

T/Tg

phi1

omega1

Qv − 1

Qw − 1

MW

Ring Count

Aromatic Ring Count

Rotatable Bond Count

Interaction Features

...
```

---

# Screen 6 — About

## Goal

Provide scientific context and documentation.

Contents include:

- Predictor overview
- Citation
- GitHub repository
- Manuscript link
- Supporting Information
- Research Use Disclaimer

---

# User Experience Principles

Every screen should answer one scientific question.

| Screen | User Question |
|---------|---------------|
| Home | What does this predictor do? |
| Known System | I know my chemistry. Can you fill in the rest? |
| Custom System | I know my descriptors. Predict diffusivity. |
| Prediction Summary | What do the models predict? |
| Advanced Details | What descriptors were actually used? |
| About | How should this predictor be interpreted and cited? |

---

# Future Enhancements

Potential future additions include:

- Batch prediction
- Monte Carlo interface
- PDF report generation
- User-uploaded polymer databases
- Permeability prediction
- Partition-coefficient prediction

These features are outside the scope of Version 1.0.
