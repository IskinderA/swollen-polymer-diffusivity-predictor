from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SAVED_DIR = HERE / "saved_models_v3"
QRF_PATH = SAVED_DIR / "qrf_pipeline.joblib"
MLP_PATH = SAVED_DIR / "mlp_128_64_pipeline.joblib"
META_PATH = SAVED_DIR / "model_metadata.json"
AGH_META_PATH = SAVED_DIR / "agh_metadata.json"
MD_PATH = SAVED_DIR / "mahalanobis_reference.joblib"
MLP_ENSEMBLE_DIR = SAVED_DIR / "mlp_ensemble"

WRAPPER_VERSION = "3.0"
EXPECTED_QRF_TREES = 300
EXPECTED_MLP_ENSEMBLE_MEMBERS = 20
EXPECTED_TRAINING_ROWS = 2095
EXPECTED_TRAINING_POLYMERS = 28
BASELINE_NUMERIC_FEATURES = [
    "T_over_Tg","Tg_K","solute_MW","volume_fraction","mass_fraction",
    "Qv-1","Qw-1","Polymer_Xc",
]
CAT_FEATURES = ["CHRIS Category"]

def load_metadata():
    meta=json.loads(META_PATH.read_text(encoding="utf-8"))
    if int(meta.get("training_rows",-1)) != EXPECTED_TRAINING_ROWS:
        raise ValueError("v3 requires Version-12 2095-row deployment artifacts.")
    if int(meta.get("training_polymers",-1)) != EXPECTED_TRAINING_POLYMERS:
        raise ValueError("v3 requires Version-12 28-polymer deployment artifacts.")
    if meta.get("headline_predictor") != "Baseline AGH":
        raise ValueError("v3 metadata does not identify Baseline AGH as headline predictor.")
    if list(meta.get("numeric_features",[])) != BASELINE_NUMERIC_FEATURES:
        raise ValueError("v3 metadata Baseline numeric-feature contract mismatch.")
    if list(meta.get("categorical_features",[])) != CAT_FEATURES:
        raise ValueError("v3 metadata categorical-feature contract mismatch.")
    return meta

def load_agh_metadata():
    agh=json.loads(AGH_META_PATH.read_text(encoding="utf-8"))
    if agh.get("architecture") != "Baseline AGH":
        raise ValueError("AGH metadata architecture mismatch.")
    cutoff=float(agh["mahalanobis_cutoff"])
    if not np.isfinite(cutoff):
        raise ValueError("AGH cutoff is not finite.")
    if list(agh["mahalanobis_definition"]["numeric_features"]) != BASELINE_NUMERIC_FEATURES:
        raise ValueError("AGH Mahalanobis numeric-feature contract mismatch.")
    return agh

def swelling_inputs_from_mass_ratio(mass_ratio,rho_polymer,rho_solvent):
    if mass_ratio < 1: raise ValueError("mass_ratio must be greater than or equal to 1.")
    if rho_polymer <= 0 or rho_solvent <= 0: raise ValueError("Densities must be positive.")
    qwm1=mass_ratio-1.0
    mf=qwm1/mass_ratio
    vp=1.0/rho_polymer
    vs=qwm1/rho_solvent
    vt=vp+vs
    return dict(volume_fraction=vs/vt,mass_fraction=mf,Qv_minus_1=vt/vp-1.0,Qw_minus_1=qwm1)

def descriptors_from_smiles(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as exc:
        raise ImportError("RDKit is required to calculate solute molecular weight from SMILES.") from exc
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Could not parse SMILES: {smiles}")
    return {"solute_MW":float(Descriptors.MolWt(mol))}

def smiles_from_cas(cas):
    try:
        import pubchempy as pcp, cirpy, chemicals
    except ImportError as exc:
        raise ImportError("CAS lookup requires pubchempy, cirpy, and chemicals.") from exc
    smiles=None
    try: smiles=chemicals.search_chemical(cas).smiles
    except Exception: pass
    if not smiles:
        try:
            c=pcp.get_compounds(cas,namespace="name")
            smiles=c[0].isomeric_smiles if c else None
        except Exception: pass
    if not smiles:
        try: smiles=cirpy.resolve(cas,"smiles")
        except Exception: pass
    if isinstance(smiles,list): smiles=smiles[0]
    if not smiles: raise ValueError(f"Could not resolve CAS to SMILES: {cas}.")
    return smiles

def build_inputs_from_simple_interface(T,Tg_K,mass_ratio,rho_polymer,rho_solvent,
                                       Polymer_Xc,CHRIS_Category,smiles=None,cas=None,
                                       solute_MW=None):
    swelling=swelling_inputs_from_mass_ratio(mass_ratio,rho_polymer,rho_solvent)
    if solute_MW is None:
        if not smiles and cas: smiles=smiles_from_cas(cas)
        if not smiles: raise ValueError("Provide SMILES, CAS, or explicit solute_MW.")
        solute_MW=descriptors_from_smiles(smiles)["solute_MW"]
    return dict(T=T,Tg_K=Tg_K,Polymer_Xc=Polymer_Xc,CHRIS_Category=CHRIS_Category,
                solute_MW=float(solute_MW),**swelling)

def build_feature_row(T,Tg_K,volume_fraction,mass_fraction,Qv_minus_1,Qw_minus_1,
                      Polymer_Xc,solute_MW,CHRIS_Category):
    if Tg_K <= 0: raise ValueError("Tg_K must be positive.")
    row={
        "T_over_Tg":T/Tg_K,"Tg_K":Tg_K,"solute_MW":solute_MW,
        "volume_fraction":volume_fraction,"mass_fraction":mass_fraction,
        "Qv-1":Qv_minus_1,"Qw-1":Qw_minus_1,"Polymer_Xc":Polymer_Xc,
        "CHRIS Category":CHRIS_Category,
    }
    return pd.DataFrame([row])

def _feature_row_from_kwargs(kwargs):
    return build_feature_row(**kwargs)

def _empirical_stats(x):
    x=np.asarray(x,dtype=float).reshape(-1)
    if x.size==0: raise ValueError("Empirical prediction distribution is empty.")
    D=10**x
    p5,p50,p95=np.percentile(x,[5,50,95])
    d5,d50,d95=np.percentile(D,[5,50,95])
    return dict(n=int(x.size),log10D_mean=float(x.mean()),log10D_std=float(x.std()),
                log10D_p5=float(p5),log10D_p50=float(p50),log10D_p95=float(p95),
                D_mean=float(D.mean()),D_std=float(D.std()),D_p5=float(d5),
                D_p50=float(d50),D_p95=float(d95))

def _with_stats(result,stats,prefix="prediction_distribution"):
    result.update({f"{prefix}_{k}":v for k,v in stats.items()})
    return result

def _qrf_predictions(X):
    model=joblib.load(QRF_PATH)
    z=model.named_steps["preprocess"].transform(X)
    preds=np.array([float(t.predict(z)[0]) for t in model.named_steps["model"].estimators_])
    if len(preds)!=EXPECTED_QRF_TREES: raise ValueError(f"Expected 300 QRF trees, found {len(preds)}.")
    return preds

def _mlp_predictions(X):
    files=sorted(MLP_ENSEMBLE_DIR.glob("mlp_seed_*.joblib"))
    if len(files)!=EXPECTED_MLP_ENSEMBLE_MEMBERS:
        raise ValueError(f"Expected 20 MLP ensemble files, found {len(files)}.")
    return np.array([float(joblib.load(p).predict(X)[0]) for p in files])

def _mahalanobis_distance(X):
    ref=joblib.load(MD_PATH)
    feats=list(ref["numeric_features"])
    if feats != BASELINE_NUMERIC_FEATURES: raise ValueError("Mahalanobis reference feature mismatch.")
    z=np.asarray(ref["transformer"].transform(X[feats]),float)
    diff=z-np.asarray(ref["centroid"],float)
    inv=np.asarray(ref["covariance_pseudoinverse"],float)
    sq=np.einsum("ij,jk,ik->i",diff,inv,diff)
    return float(np.sqrt(np.maximum(sq,0))[0])

def _result(model,preds,central_definition):
    stats=_empirical_stats(preds)
    result=dict(model=model,wrapper_version=WRAPPER_VERSION,
                prediction_distribution_log10D=np.asarray(preds,float).copy(),
                prediction_distribution_D=10**np.asarray(preds,float),
                central_prediction_definition=central_definition,
                central_log10D=stats["log10D_p50"],central_D=stats["D_p50"],
                log10D_p5=stats["log10D_p5"],log10D_p50=stats["log10D_p50"],
                log10D_p95=stats["log10D_p95"],D_p5=stats["D_p5"],
                D_p50=stats["D_p50"],D_p95=stats["D_p95"],units="cm^2/s",
                interval_note="Empirical model-derived prediction variability; not a calibrated confidence interval.")
    return _with_stats(result,stats)

def predict_diffusivity_qrf(**kwargs):
    return _result("Baseline QRF",_qrf_predictions(_feature_row_from_kwargs(kwargs)),
                   "Median of 300 QRF tree predictions")

def predict_diffusivity_mlp_ensemble(**kwargs):
    return _result("Baseline MLP ensemble",_mlp_predictions(_feature_row_from_kwargs(kwargs)),
                   "Median of 20 independently initialized MLP predictions")

def predict_diffusivity_agh(**kwargs):
    X=_feature_row_from_kwargs(kwargs)
    q=_qrf_predictions(X); m=_mlp_predictions(X)
    qstats=_empirical_stats(q); mstats=_empirical_stats(m)
    dist=_mahalanobis_distance(X)
    aghmeta=load_agh_metadata()
    cutoff=float(aghmeta["mahalanobis_cutoff"])
    selected="MLP" if dist < cutoff else "QRF"
    preds=m if selected=="MLP" else q
    result=_result("Baseline AGH",preds,f"Median of selected {selected} empirical prediction distribution")
    result.update(
        selected_model=selected,mahalanobis_distance=dist,mahalanobis_cutoff=cutoff,
        inside_applicability_region=bool(dist<cutoff),
        applicability_status=("Within learned applicability region" if dist<cutoff
                              else "Beyond learned applicability cutoff"),
        qrf_central_log10D=qstats["log10D_p50"],mlp_central_log10D=mstats["log10D_p50"],
        qrf_mlp_difference_log10D=abs(qstats["log10D_p50"]-mstats["log10D_p50"]),
        training_rows=int(load_metadata()["training_rows"]),
        training_polymers=int(load_metadata()["training_polymers"]),
        routing_rule=aghmeta["routing_rule"],
    )
    return result

def predict_diffusivity_all(**kwargs):
    return {"qrf":predict_diffusivity_qrf(**kwargs),
            "mlp":predict_diffusivity_mlp_ensemble(**kwargs),
            "agh":predict_diffusivity_agh(**kwargs)}

def _sample(result,n_samples,random_state):
    if n_samples<=0: raise ValueError("n_samples must be positive.")
    rng=np.random.default_rng(random_state)
    pool=np.asarray(result["prediction_distribution_log10D"],float)
    s=rng.choice(pool,size=int(n_samples),replace=True)
    stats=_empirical_stats(s)
    out=dict(model=result["model"],wrapper_version=WRAPPER_VERSION,n_samples=int(n_samples),
             log10D_samples=s,D_samples=10**s)
    return _with_stats(out,stats,"sample_distribution")

def sample_diffusivity_qrf(n_samples=1000,random_state=None,**kwargs):
    return _sample(predict_diffusivity_qrf(**kwargs),n_samples,random_state)

def sample_diffusivity_mlp(n_samples=1000,random_state=None,**kwargs):
    return _sample(predict_diffusivity_mlp_ensemble(**kwargs),n_samples,random_state)

def sample_diffusivity_agh(n_samples=1000,random_state=None,**kwargs):
    r=predict_diffusivity_agh(**kwargs)
    out=_sample(r,n_samples,random_state)
    out.update(selected_model=r["selected_model"],mahalanobis_distance=r["mahalanobis_distance"],
               mahalanobis_cutoff=r["mahalanobis_cutoff"],
               inside_applicability_region=r["inside_applicability_region"])
    return out

def predict_diffusivity(model="agh",**kwargs):
    model=model.lower()
    if model in ["agh","hybrid","baseline_agh"]: return predict_diffusivity_agh(**kwargs)
    if model in ["qrf","rf"]: return predict_diffusivity_qrf(**kwargs)
    if model in ["mlp","nn","neural_network","mlp_ensemble"]: return predict_diffusivity_mlp_ensemble(**kwargs)
    raise ValueError("model must be 'agh', 'qrf', or 'mlp'")

def _simple(kwargs):
    return build_inputs_from_simple_interface(**kwargs)

def predict_from_simple_interface(model="agh",**kwargs):
    return predict_diffusivity(model=model,**_simple(kwargs))

def predict_from_simple_interface_all(**kwargs):
    return predict_diffusivity_all(**_simple(kwargs))

def sample_from_simple_interface_qrf(n_samples=1000,random_state=None,**kwargs):
    return sample_diffusivity_qrf(n_samples=n_samples,random_state=random_state,**_simple(kwargs))

def sample_from_simple_interface_mlp(n_samples=1000,random_state=None,**kwargs):
    return sample_diffusivity_mlp(n_samples=n_samples,random_state=random_state,**_simple(kwargs))

def sample_from_simple_interface_agh(n_samples=1000,random_state=None,**kwargs):
    return sample_diffusivity_agh(n_samples=n_samples,random_state=random_state,**_simple(kwargs))
