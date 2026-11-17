from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class PredictionDistribution:
    p5:float; p50:float; p95:float; mean:float; sd:float; n:int
    D_p5:float; D_p50:float; D_p95:float; D_mean:float; D_sd:float

@dataclass(frozen=True)
class SampleDistribution:
    p5:float; p50:float; p95:float; mean:float; sd:float; n:int
    D_p5:float; D_p50:float; D_p95:float; D_mean:float; D_sd:float

@dataclass
class PredictionResult:
    qrf:PredictionDistribution
    mlp:PredictionDistribution
    agh:PredictionDistribution
    agh_central_log10D:float
    agh_central_D:float
    selected_model:str
    mahalanobis_distance:float
    mahalanobis_cutoff:float
    inside_applicability_region:bool
    applicability_status:str
    qrf_mlp_difference_log10D:float
    qrf_prediction_log10:list[float]; qrf_prediction_D:list[float]
    mlp_prediction_log10:list[float]; mlp_prediction_D:list[float]
    agh_prediction_log10:list[float]; agh_prediction_D:list[float]
    qrf_log10_samples:list[float]; qrf_D_samples:list[float]
    mlp_log10_samples:list[float]; mlp_D_samples:list[float]
    agh_log10_samples:list[float]; agh_D_samples:list[float]
    qrf_sample:SampleDistribution; mlp_sample:SampleDistribution; agh_sample:SampleDistribution
    n_qrf_tree_predictions:int; n_mlp_ensemble_members:int; n_agh_distribution_values:int
    n_samples_requested:int; training_rows:int; training_polymers:int; wrapper_version:str

@dataclass(frozen=True)
class SimplePredictorInputs:
    temperature_k:float; tg_k:float; mass_ratio:float; rho_polymer:float
    rho_solvent:float; polymer_xc:float; chris_category:str
    smiles:str=""; cas:str=""; n_samples:int=1000

def _dist(r):
    return PredictionDistribution(
        p5=float(r["prediction_distribution_log10D_p5"]),
        p50=float(r["prediction_distribution_log10D_p50"]),
        p95=float(r["prediction_distribution_log10D_p95"]),
        mean=float(r["prediction_distribution_log10D_mean"]),
        sd=float(r["prediction_distribution_log10D_std"]),
        n=int(r["prediction_distribution_n"]),
        D_p5=float(r["prediction_distribution_D_p5"]),
        D_p50=float(r["prediction_distribution_D_p50"]),
        D_p95=float(r["prediction_distribution_D_p95"]),
        D_mean=float(r["prediction_distribution_D_mean"]),
        D_sd=float(r["prediction_distribution_D_std"]),
    )

def _sample(r):
    return SampleDistribution(
        p5=float(r["sample_distribution_log10D_p5"]),p50=float(r["sample_distribution_log10D_p50"]),
        p95=float(r["sample_distribution_log10D_p95"]),mean=float(r["sample_distribution_log10D_mean"]),
        sd=float(r["sample_distribution_log10D_std"]),n=int(r["sample_distribution_n"]),
        D_p5=float(r["sample_distribution_D_p5"]),D_p50=float(r["sample_distribution_D_p50"]),
        D_p95=float(r["sample_distribution_D_p95"]),D_mean=float(r["sample_distribution_D_mean"]),
        D_sd=float(r["sample_distribution_D_std"]),
    )

def _list(x:Sequence[float]|np.ndarray): return np.asarray(x,float).reshape(-1).tolist()

def predict(inputs:SimplePredictorInputs)->PredictionResult:
    if int(inputs.n_samples)<=0: raise ValueError("n_samples must be positive.")
    from predictor.reference import wrapper_v3 as wrapper
    kw=dict(T=inputs.temperature_k,Tg_K=inputs.tg_k,mass_ratio=inputs.mass_ratio,
            rho_polymer=inputs.rho_polymer,rho_solvent=inputs.rho_solvent,
            Polymer_Xc=inputs.polymer_xc,CHRIS_Category=inputs.chris_category,
            smiles=inputs.smiles or None,cas=inputs.cas or None)
    allr=wrapper.predict_from_simple_interface_all(**kw)
    q,m,a=allr["qrf"],allr["mlp"],allr["agh"]
    seed=42
    qs=wrapper.sample_from_simple_interface_qrf(n_samples=inputs.n_samples,random_state=seed,**kw)
    ms=wrapper.sample_from_simple_interface_mlp(n_samples=inputs.n_samples,random_state=seed,**kw)
    ags=wrapper.sample_from_simple_interface_agh(n_samples=inputs.n_samples,random_state=seed,**kw)
    qd,md,ad=_dist(q),_dist(m),_dist(a)
    if qd.n!=300: raise ValueError(f"v3 expected 300 QRF predictions, found {qd.n}.")
    if md.n!=20: raise ValueError(f"v3 expected 20 MLP predictions, found {md.n}.")
    expected=20 if a["selected_model"]=="MLP" else 300
    if ad.n!=expected: raise ValueError(f"AGH selected {a['selected_model']} but distribution n={ad.n}.")
    return PredictionResult(
        qrf=qd,mlp=md,agh=ad,agh_central_log10D=float(a["central_log10D"]),
        agh_central_D=float(a["central_D"]),selected_model=str(a["selected_model"]),
        mahalanobis_distance=float(a["mahalanobis_distance"]),mahalanobis_cutoff=float(a["mahalanobis_cutoff"]),
        inside_applicability_region=bool(a["inside_applicability_region"]),
        applicability_status=str(a["applicability_status"]),
        qrf_mlp_difference_log10D=float(a["qrf_mlp_difference_log10D"]),
        qrf_prediction_log10=_list(q["prediction_distribution_log10D"]),qrf_prediction_D=_list(q["prediction_distribution_D"]),
        mlp_prediction_log10=_list(m["prediction_distribution_log10D"]),mlp_prediction_D=_list(m["prediction_distribution_D"]),
        agh_prediction_log10=_list(a["prediction_distribution_log10D"]),agh_prediction_D=_list(a["prediction_distribution_D"]),
        qrf_log10_samples=_list(qs["log10D_samples"]),qrf_D_samples=_list(qs["D_samples"]),
        mlp_log10_samples=_list(ms["log10D_samples"]),mlp_D_samples=_list(ms["D_samples"]),
        agh_log10_samples=_list(ags["log10D_samples"]),agh_D_samples=_list(ags["D_samples"]),
        qrf_sample=_sample(qs),mlp_sample=_sample(ms),agh_sample=_sample(ags),
        n_qrf_tree_predictions=qd.n,n_mlp_ensemble_members=md.n,n_agh_distribution_values=ad.n,
        n_samples_requested=int(inputs.n_samples),training_rows=int(a["training_rows"]),
        training_polymers=int(a["training_polymers"]),wrapper_version=str(a["wrapper_version"]),
    )
