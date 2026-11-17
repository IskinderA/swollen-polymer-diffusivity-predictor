from __future__ import annotations
import csv, io, json
from dataclasses import asdict
from predictor.predict_v3 import PredictionResult,SimplePredictorInputs

def _csv(rows):
    b=io.StringIO(newline="");csv.writer(b,lineterminator="\n").writerows(rows);return b.getvalue()

def _inputs(i):
    return [["input","",k,v] for k,v in [
        ("temperature_k",i.temperature_k),("tg_k",i.tg_k),("mass_ratio",i.mass_ratio),
        ("rho_polymer",i.rho_polymer),("rho_solvent",i.rho_solvent),("polymer_xc",i.polymer_xc),
        ("chris_category",i.chris_category),("smiles",i.smiles),("cas",i.cas),
        ("n_samples_requested",i.n_samples)]]

def _drows(name,s):
    return [["prediction_distribution",name,k,v] for k,v in [
        ("n",s.n),("mean_log10D",s.mean),("sd_log10D",s.sd),("p5_log10D",s.p5),
        ("p50_log10D",s.p50),("p95_log10D",s.p95),("mean_D_cm2_s",s.D_mean),
        ("sd_D_cm2_s",s.D_sd),("p5_D_cm2_s",s.D_p5),("p50_D_cm2_s",s.D_p50),("p95_D_cm2_s",s.D_p95)]]

def _srows(name,s):
    return [["sample_distribution",name,k,v] for k,v in [
        ("n",s.n),("mean_log10D",s.mean),("sd_log10D",s.sd),("p5_log10D",s.p5),
        ("p50_log10D",s.p50),("p95_log10D",s.p95),("mean_D_cm2_s",s.D_mean),
        ("sd_D_cm2_s",s.D_sd),("p5_D_cm2_s",s.D_p5),("p50_D_cm2_s",s.D_p50),("p95_D_cm2_s",s.D_p95)]]

def prediction_summary_csv(pred:PredictionResult,inputs:SimplePredictorInputs)->str:
    rows=[["section","model","field","value"]]+_inputs(inputs)
    rows += [
        ["model_metadata","","wrapper_version",pred.wrapper_version],
        ["model_metadata","","training_rows",pred.training_rows],
        ["model_metadata","","training_polymers",pred.training_polymers],
        ["model_metadata","QRF","n_qrf_tree_predictions",pred.n_qrf_tree_predictions],
        ["model_metadata","MLP","n_mlp_ensemble_members",pred.n_mlp_ensemble_members],
        ["headline_prediction","Baseline AGH","central_log10D",pred.agh_central_log10D],
        ["headline_prediction","Baseline AGH","central_D_cm2_s",pred.agh_central_D],
        ["applicability","Baseline AGH","selected_model",pred.selected_model],
        ["applicability","Baseline AGH","mahalanobis_distance",pred.mahalanobis_distance],
        ["applicability","Baseline AGH","mahalanobis_cutoff",pred.mahalanobis_cutoff],
        ["applicability","Baseline AGH","inside_applicability_region",pred.inside_applicability_region],
        ["applicability","Baseline AGH","status",pred.applicability_status],
        ["diagnostic","","qrf_mlp_difference_log10D",pred.qrf_mlp_difference_log10D],
    ]
    rows += _drows("QRF",pred.qrf)+_drows("MLP",pred.mlp)+_drows("Baseline AGH selected distribution",pred.agh)
    rows += _srows("QRF",pred.qrf_sample)+_srows("MLP",pred.mlp_sample)+_srows("Baseline AGH",pred.agh_sample)
    rows += [["interpretation","Baseline AGH","distribution_note",
              "AGH uses the empirical prediction distribution of the constituent selected by the applicability gate; it does not construct a synthetic hybrid distribution."],
             ["interpretation","","interval_note",
              "p5-p95 ranges are empirical prediction-variability estimates, not calibrated confidence or credible intervals."]]
    return _csv(rows)

def prediction_distributions_csv(pred):
    rows=[["distribution_type","model","distribution_index","log10D","D_cm2_s"]]
    for name,l,d in [("QRF",pred.qrf_prediction_log10,pred.qrf_prediction_D),
                     ("MLP",pred.mlp_prediction_log10,pred.mlp_prediction_D),
                     ("Baseline AGH",pred.agh_prediction_log10,pred.agh_prediction_D)]:
        for j,(x,y) in enumerate(zip(l,d),1):
            rows.append(["model_derived_prediction_distribution",name,j,x,y])
    return _csv(rows)

def monte_carlo_samples_csv(pred):
    rows=[["distribution_type","model","sample_index","log10D","D_cm2_s"]]
    for name,l,d in [("QRF",pred.qrf_log10_samples,pred.qrf_D_samples),
                     ("MLP",pred.mlp_log10_samples,pred.mlp_D_samples),
                     ("Baseline AGH",pred.agh_log10_samples,pred.agh_D_samples)]:
        for j,(x,y) in enumerate(zip(l,d),1):
            rows.append(["monte_carlo_sample_distribution",name,j,x,y])
    return _csv(rows)

def prediction_summary_json(pred,inputs):
    payload={
        "inputs":asdict(inputs),
        "model_metadata":{"wrapper_version":pred.wrapper_version,"training_rows":pred.training_rows,
                          "training_polymers":pred.training_polymers,
                          "n_qrf_tree_predictions":pred.n_qrf_tree_predictions,
                          "n_mlp_ensemble_members":pred.n_mlp_ensemble_members,
                          "n_agh_distribution_values":pred.n_agh_distribution_values},
        "headline_baseline_agh":{"central_log10D":pred.agh_central_log10D,
                                "central_D_cm2_s":pred.agh_central_D,
                                "selected_model":pred.selected_model,
                                "mahalanobis_distance":pred.mahalanobis_distance,
                                "mahalanobis_cutoff":pred.mahalanobis_cutoff,
                                "inside_applicability_region":pred.inside_applicability_region,
                                "applicability_status":pred.applicability_status},
        "diagnostics":{"qrf_mlp_difference_log10D":pred.qrf_mlp_difference_log10D},
        "prediction_distributions":{"qrf":asdict(pred.qrf),"mlp":asdict(pred.mlp),"agh_selected":asdict(pred.agh)},
        "sample_distributions":{"qrf":asdict(pred.qrf_sample),"mlp":asdict(pred.mlp_sample),"agh":asdict(pred.agh_sample)},
        "interpretation":{"agh_distribution":"AGH distribution is the empirical distribution of the selected constituent.",
                          "variability":"p5-p95 ranges are empirical prediction variability, not calibrated confidence intervals."}
    }
    return json.dumps(payload,indent=2)

def prediction_full_json(pred,inputs):
    p=json.loads(prediction_summary_json(pred,inputs))
    p["raw_prediction_distributions"]={
        "qrf":{"log10D":pred.qrf_prediction_log10,"D_cm2_s":pred.qrf_prediction_D},
        "mlp":{"log10D":pred.mlp_prediction_log10,"D_cm2_s":pred.mlp_prediction_D},
        "agh_selected":{"selected_model":pred.selected_model,"log10D":pred.agh_prediction_log10,"D_cm2_s":pred.agh_prediction_D}}
    p["raw_monte_carlo_samples"]={
        "qrf":{"log10D":pred.qrf_log10_samples,"D_cm2_s":pred.qrf_D_samples},
        "mlp":{"log10D":pred.mlp_log10_samples,"D_cm2_s":pred.mlp_D_samples},
        "agh":{"selected_model":pred.selected_model,"log10D":pred.agh_log10_samples,"D_cm2_s":pred.agh_D_samples}}
    return json.dumps(p,indent=2)
