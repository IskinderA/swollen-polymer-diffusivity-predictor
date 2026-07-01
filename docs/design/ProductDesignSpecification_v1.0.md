# Swollen Polymer Diffusivity Predictor

# Product Design Specification (PDS)

**Version:** 1.0

**Status:** Internal development document

---

## 1. Overview

The **Swollen Polymer Diffusivity Predictor** is a **provisional companion predictor** accompanying the manuscript

> **Predicting Solute Diffusivity in Swollen Polymer Systems Using Quantile Random Forests and Neural Networks**

The predictor provides researchers with an intuitive interface for applying the published machine-learning models without requiring knowledge of the underlying feature engineering or machine-learning implementation.

The predictor is intended to facilitate:

- manuscript reproducibility
- exploratory analysis
- educational use
- rapid screening of polymer transport behavior

The predictor is **not** intended to replace experimental measurements or established engineering judgment.

---

# 2. Vision

The predictor should function as the **reference implementation** accompanying the manuscript.

The manuscript explains the scientific foundations of the models.

The predictor allows users to apply the final published models using experimentally measurable quantities through a transparent and reproducible interface.

The predictor should feel more like a scientific instrument than a software application.

---

# 3. Intended Users

Three principal user groups are anticipated.

## 3.1 Exploratory Researcher

Users familiar with polymer systems but not necessarily with machine learning.

Typically knows:

- polymer
- solvent
- solute
- temperature
- swelling

Uses the **Known System** interface.

---

## 3.2 Advanced User

Users who possess experimentally measured material properties or wish to evaluate hypothetical systems.

Typically knows:

- temperature
- glass-transition temperature
- densities
- swelling
- crystallinity
- CHRIS category
- SMILES (or CAS)

Uses the **Custom System** interface.

---

## 3.3 Manuscript Reader

Readers wishing to reproduce representative examples presented in the accompanying publication.

Uses the **Example Systems** interface.

---

# 4. Design Philosophy

The predictor shall follow four guiding design principles.

## Principle 1

### Ask only for experimentally measurable quantities.

Users should never be asked to enter engineered descriptors.

Internal features shall be generated automatically.

---

## Principle 2

### Reveal complexity only when requested.

The default interface should remain approachable.

Advanced users may inspect computed descriptors if desired.

---

## Principle 3

### Earn trust rather than impress.

The predictor shall avoid exaggerated claims or artificial confidence scores.

Outputs shall be scientifically interpretable and directly supported by the accompanying manuscript.

---

## Principle 4

### Scientific transparency.

All predictions shall be reproducible.

All derived descriptors shall be generated automatically using documented procedures.

Every prediction shall be exportable for downstream analysis.

---

# 5. Functional Requirements

The predictor shall provide two principal prediction modes.

## Mode A — Known System

The user selects:

- Polymer
- Solvent
- Solute

from curated dropdown menus.

The predictor automatically retrieves:

- polymer density
- solvent density
- glass-transition temperature
- crystallinity
- CHRIS category
- molecular structure

The user supplies only:

- temperature
- swollen/dry mass ratio

The predictor generates all remaining descriptors internally.

---

## Mode B — Custom System

The user supplies the simplified predictor inputs developed for the published manuscript.

Required inputs:

- Temperature
- Glass-transition temperature
- Swollen/dry mass ratio
- Polymer density
- Solvent density
- Polymer crystallinity
- CHRIS category
- SMILES (preferred)

or

- CAS number (optional)

If a CAS number is supplied, the predictor attempts to resolve the corresponding SMILES representation automatically.

The predictor internally computes:

- swelling descriptors
- molecular descriptors
- interaction descriptors
- complete machine-learning feature vector

No additional feature engineering is required from the user.

---

# 6. Prediction Engine

Regardless of input mode, all predictions shall utilize the same published prediction engine.

Internally, the predictor shall call the simplified interfaces

```python
predict_from_simple_interface()

predict_from_simple_interface_mlp_ensemble()
```

Sampling utilities shall also be available through

```python
sample_from_simple_interface_qrf()

sample_from_simple_interface_mlp()
```

This architecture ensures that all user interfaces produce scientifically identical predictions.

---
# 7. Prediction Summary

Every prediction shall produce a concise scientific summary.

The summary shall include:

## Inputs

A summary of all user-supplied quantities together with the internally generated descriptor values used by the published machine-learning models.

---

## QRF Prediction Distribution

The predictor shall report the empirical prediction distribution derived from the published Quantile Random Forest (QRF).

Displayed quantities shall include:

- 5th percentile
- Median (50th percentile)
- 95th percentile

The user may additionally generate arbitrary numbers of sampled predictions from the QRF distribution for downstream uncertainty propagation and Monte Carlo analyses.

---

## MLP Ensemble Prediction Distribution

The predictor shall report the empirical prediction distribution obtained from an ensemble of **20 independently trained neural networks** initialized using different random seeds.

Displayed quantities shall include:

- 5th percentile
- Median (50th percentile)
- 95th percentile

The user may additionally generate arbitrary numbers of sampled predictions from the ensemble distribution for downstream uncertainty propagation and Monte Carlo analyses.

---

## Model Agreement

Agreement between the independent QRF and MLP model classes shall be summarized qualitatively.

Possible categories include:

- Excellent agreement
- Moderate agreement
- Large disagreement

Agreement shall not be interpreted as a calibrated probability.

---

## Interpretation

A concise scientific interpretation shall accompany every prediction.

Example:

> The two independent model classes produce similar prediction distributions. This behavior is consistent with systems reasonably represented within the published training domain.

or

> The two model classes produce substantially different prediction distributions. Predictions should be interpreted cautiously and experimental confirmation is recommended.

---

### Notes

1. Both QRF and MLP outputs are reported as empirical prediction distributions rather than single deterministic values.

2. **QRF prediction intervals and sampling distributions** are derived from tree-to-tree variability within the trained random forest.

3. **MLP ensemble prediction intervals and sampling distributions** are derived from prediction variability across independently trained neural networks initialized using different random seeds.

4. Although both approaches generate empirical prediction distributions, they quantify different sources of model variability and should not be interpreted as calibrated probabilistic confidence intervals.

---

# 8. Downloads

Every prediction shall be exportable.

Initial supported formats:

- CSV
- JSON

Exported files shall include:

- user inputs
- internally generated descriptors
- QRF prediction distribution
- MLP ensemble prediction distribution
- sampled prediction values (if requested)
- model version
- prediction timestamp

The exported CSV files are intended to facilitate downstream statistical analyses, uncertainty propagation, and integration into user-developed workflows.

Future versions may additionally support formatted PDF reports.

---

# 9. User Interface

The predictor shall provide three user entry points.

- Known System
- Custom System
- Example Systems

Detailed interface layouts are documented separately in

```
docs/design/Wireframes_v1.0.md
```

---

# 10. Research Use Disclaimer

This provisional companion predictor accompanies the manuscript

> **Predicting Solute Diffusivity in Swollen Polymer Systems Using Quantile Random Forests and Neural Networks**

It is provided by the authors to facilitate reproducibility, exploratory analysis, educational use, and application of the published machine-learning models. The predictor is a provisional research implementation and is **not** an FDA-qualified Regulatory Science Tool (RST), Medical Device Development Tool (MDDT), or FDA-approved decision-support tool. Predictions generated by this predictor should not replace experimental measurements, engineering judgment, or established regulatory evaluation procedures.

The findings and conclusions in the accompanying manuscript are those of the authors and do not necessarily represent any determination or policy of the U.S. Food and Drug Administration. The mention of commercial products, their sources, or their use in connection with material reported therein is not to be construed as either an actual or implied endorsement of such products by the Department of Health and Human Services.

Information regarding FDA-qualified tools is available through the publicly accessible resources below.

- **CDRH Regulatory Science Tool (RST) Catalog**  
  https://cdrh-rst.fda.gov/

- **Medical Device Development Tool (MDDT) Program**  
  https://www.fda.gov/medical-devices/medical-device-development-tools-mddt

---

# 11. Development Roadmap

## Version 1.0

- Predictor interface
- Known System mode
- Custom System mode
- QRF predictions
- MLP predictions
- CSV export
- GitHub repository
- Online companion predictor

---

## Future Versions

Potential future enhancements include:

- Monte Carlo sampling interface
- Batch prediction
- User-uploaded datasets
- Expanded polymer libraries
- Additional transport-property prediction modules
- Permeability prediction
- Partition-coefficient prediction

---

# 12. Relationship to the Accompanying Manuscript

The manuscript describes the scientific development, validation, and interpretation of the published machine-learning models.

The Supporting Information documents the underlying methodology and validation studies.

The companion predictor provides a practical implementation of the final published models using experimentally measurable quantities through a transparent and reproducible user interface.

Together, the manuscript, Supporting Information, GitHub repository, and companion predictor form a unified research package intended to maximize scientific transparency, reproducibility, and practical utility.
