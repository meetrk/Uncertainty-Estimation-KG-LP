
import os
import sys
import json
import os.path as osp
import argparse
import time
import logging

import numpy as np
import torch
import yaml

# ----------------------------------------------------------------------
# Command line parsing
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Ensemble LP + Uncertainty with smart baseline control")
parser.add_argument("--dataset",            default="FB15k-237", choices=["WN18RR", "FB15k-237"])
parser.add_argument("--model",              required=True,
                    help="Checkpoint filename (inside --experiment-dir)")
parser.add_argument("--experiment-dir",     default="new_models/experiment_2")
parser.add_argument("--config-path",        default=None,
                    help="Path to YAML config file. If omitted, a default config is built.")
parser.add_argument("--results-root",       default="results_ensemble_uncertainty")
parser.add_argument("--num-runs",           type=int, default=20)
parser.add_argument("--mc-samples",         type=int, default=5)

# --- Granular flags ---
parser.add_argument("--baseline-lp",           action="store_true", default=False)
parser.add_argument("--baseline-unc",          action="store_true", default=False,
                    help="Run uncertainty baseline (no calibration, ensemble type)")
parser.add_argument("--eval-lp",               action="store_true", default=False)
parser.add_argument("--eval-uncertainty",      action="store_true", default=False)

# --- Master switches ---
parser.add_argument("--baseline",  action="store_true", default=False,
                    help="Run all baselines (LP + unc)")
parser.add_argument("--eval-all",  action="store_true", default=False,
                    help="Run all calibrated evaluations (LP + uncertainty)")
parser.add_argument("--all",       action="store_true", default=False,
                    help="Run everything: all baselines + all calibrated")

# --- Scoped convenience flags ---
parser.add_argument("--only-lp",          action="store_true")
parser.add_argument("--only-uncertainty", action="store_true")
parser.add_argument("--only-baseline",    action="store_true")
parser.add_argument("--only-calibrated",  action="store_true")

# --- Subset filters ---
parser.add_argument("--methods", nargs="+",
                    default=["scalar", "platt_scaling", "isotonic_regression"])

# --- Force re-run ---
parser.add_argument("--force", action="store_true", default=False)

args = parser.parse_args()

# === SMART LOGIC ===
if args.all:
    args.baseline = True
    args.eval_all = True

if args.baseline:
    args.baseline_lp  = True
    args.baseline_unc = True

if args.eval_all:
    args.eval_lp          = True
    args.eval_uncertainty = True

if args.only_lp:
    args.baseline_lp      = True
    args.eval_lp          = True
    args.baseline_unc     = False
    args.eval_uncertainty = False

if args.only_uncertainty:
    args.baseline_unc     = True
    args.eval_uncertainty = True
    args.baseline_lp      = False
    args.eval_lp          = False

if args.only_baseline:
    args.baseline_lp      = True
    args.baseline_unc     = True
    args.eval_lp          = False
    args.eval_uncertainty = False

if args.only_calibrated:
    args.eval_lp          = True
    args.eval_uncertainty = True
    args.baseline_lp      = False
    args.baseline_unc     = False

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"eval_ensemble_{args.dataset}_{args.model}_log.txt"),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import os.path as osp
from torch_geometric.nn import GAE
from model.encoder.model import RGCN
from model.decoder.distmult import DistMult
from misc.rel_link_pred_dataset import RelLinkPredDataset
from model.ensemble.deep_ensemble import DeepEnsemble
from model.trainer.ensemble_pipeline import EnsemblePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------
class Config:
    """Thin wrapper that mirrors the Config class used by Pipeline."""

    def __init__(self, config_dict):
        self._config = config_dict

    def get_section(self, section):
        return self._config.get(section, {})

    def get(self, section, key=None):
        if key is None:
            return self.get_section(section)
        return self._config.get(section, {}).get(key)


def load_yaml_config(path):
    with open(path, "r") as f:
        return Config(yaml.safe_load(f))


def build_default_config(dataset_name, model_name):
    """Fallback config when no YAML is provided — mirrors the YAML format."""
    if dataset_name == "FB15k-237":
        return Config({
            "dataset": {"name": "FB15k-237", "path": "./dataset/fb15k237"},
            "model": {
                "encoder": {"type": "RGCN", "hidden_layer_size": 500,
                            "embedding_dim": 500, "dropout": 0.2,
                            "num_bases": 5, "bases_enabled": False},
                "decoder": {"type": "DistMult", "l2_penalty": 0.001,
                            "w_gain": False, "b_init": False},
            },
            "ensemble": {"enabled": True, "num_models": 5},
            "training": {
                "epochs": 10000,
                "sampling": {"negative_sampling_ratio": 3, "edge_dropout": 0.2},
                "optimiser": {"learning_rate": 0.01, "weight_decay": 0},
                "evaluation_frequency": 500,
                "early stopping": {"enabled": True, "patience": 5, "delta": 0.001},
                "load_model": True,
                "save_model": False,
                "checkpoint_path": f"./new_models/experiment_2/{model_name}",
                "test": True,
                "label_smoothing": {"positive": 1, "negative": 0},
            },
            "calibration": {
                "enabled": True,
                "type": "ensemble",
                "mc_samples": args.mc_samples,
                "method": "isotonic_regression",
                "max_iters": 1000,
                "learning_rate": 0.01,
            },
        })

    # WN18RR
    return Config({
        "dataset": {"name": dataset_name},
        "model": {
            "encoder": {"type": "RGCN", "hidden_layer_size": 500,
                        "embedding_dim": 500, "dropout": 0.2,
                        "num_bases": 5, "bases_enabled": False},
            "decoder": {"type": "DistMult", "l2_penalty": 0.001,
                        "w_gain": False, "b_init": False},
        },
        "ensemble": {"enabled": True, "num_models": 5},
        "training": {
            "epochs": 10000,
            "sampling": {"negative_sampling_ratio": 3, "edge_dropout": 0.2},
            "optimiser": {"learning_rate": 0.01, "weight_decay": 0},
            "evaluation_frequency": 100,
            "early stopping": {"enabled": True, "patience": 10, "delta": 0.001},
            "load_model": True,
            "save_model": False,
            "checkpoint_path": f"./new_models/experiment_2/{model_name}",
            "test": True,
            "label_smoothing": {"positive": 1, "negative": 0},
        },
        "calibration": {
            "enabled": True,
            "type": "ensemble",
            "mc_samples": args.mc_samples,
            "method": "scalar",
            "max_iters": 10000,
            "learning_rate": 0.01,
        },
    })


# ----------------------------------------------------------------------
# Data & Model
# ----------------------------------------------------------------------
def load_data(dataset_name):
    path = osp.join(".", "data", "RLPD")
    dataset = RelLinkPredDataset(path, dataset_name)
    data = dataset[0]
    data["num_relations"] = dataset.num_relations
    data = data.to(device)
    return data, dataset


def build_ensemble(data, dataset, config):
    enc_cfg = config.get_section("model")["encoder"]
    dec_cfg = config.get_section("model")["decoder"]
    ens_cfg = config.get_section("ensemble")
    cal_cfg = config.get_section("calibration")

    bases_enabled = enc_cfg.get("bases_enabled", True)

    model_config = {
        "encoder": {
            "embedding_dim":    enc_cfg.get("embedding_dim", 500),
            "hidden_layer_size": enc_cfg.get("hidden_layer_size", 500),
            "num_bases":        enc_cfg.get("num_bases", 5),
            "dropout":          enc_cfg.get("dropout", 0.2),
            "bases_enabled":    bases_enabled,
        },
        "decoder": {
            "margin":      1.0,
            "sparse":      False,
            "calibration": "none",
            "l2_penalty":  dec_cfg.get("l2_penalty", 0.001),
            "w_gain":      dec_cfg.get("w_gain", False),
            "b_init":      dec_cfg.get("b_init", False),
        },
    }

    encoder_args = {
        "num_nodes":     data.num_nodes,
        "num_relations": dataset.num_relations,
        "model_config":  model_config,
    }
    decoder_args = {
        "num_nodes":     data.num_nodes,
        "num_relations": dataset.num_relations // 2,
        "embedding_dim": model_config["encoder"]["embedding_dim"],
    }

    num_models = ens_cfg.get("num_models", 5)
    logger.info(f"Creating DeepEnsemble with {num_models} models...")

    ensemble = DeepEnsemble(
        base_encoder_class=RGCN,
        base_decoder_class=DistMult,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
        num_models=num_models,
        device=device,
        calibration=cal_cfg.get("method", "none"),
    )
    return ensemble


def load_checkpoint_weights(pipeline, checkpoint_path):
    pipeline.load_checkpoint(checkpoint_path, load_optimizer=False)
    logger.info(f"Loaded ensemble checkpoint: {checkpoint_path}")


# ----------------------------------------------------------------------
# Result helpers
# ----------------------------------------------------------------------
def check_existing(out_path, need_lp, need_unc):
    if args.force or not osp.exists(out_path):
        return False, {}
    try:
        with open(out_path, "r") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"  [CORRUPT] {out_path} — invalid JSON, will re-run")
        return False, {}

    missing = []
    if need_lp  and "link_prediction" not in existing:
        missing.append("link_prediction")
    if need_unc and "uncertainty" not in existing:
        missing.append("uncertainty")

    if missing:
        logger.info(f"  [PARTIAL] {out_path} — missing: {missing}, will re-run")
        return False, existing

    logger.info(f"  [SKIP] already complete: {out_path}")
    return True, existing


def stats(lst):
    lst = [float(x) for x in lst]
    return {"runs": lst, "mean": float(np.mean(lst)), "var": float(np.var(lst, ddof=1))}


def save_metrics(dataset_name, model_name, method, ctype,
                 lp_stats=None, unc_stats=None, baseline=False):
    subdir   = "baseline" if baseline else ctype
    out_dir  = osp.join(args.results_root, dataset_name, model_name, method, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, "metrics.json")

    result = {
        "dataset":            dataset_name,
        "model_name":         model_name,
        "calibration_method": "none" if baseline else method,
        "calibration_type":   "baseline" if baseline else ctype,
        "num_runs":           args.num_runs,
        "mc_samples":         args.mc_samples,
        "is_baseline":        baseline,
    }
    if lp_stats:  result["link_prediction"] = lp_stats
    if unc_stats: result["uncertainty"]      = unc_stats

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved → {out_path}")


# ----------------------------------------------------------------------
# Core evaluation functions
# ----------------------------------------------------------------------
# Ensemble always uses type="ensemble"; the ctype arg is kept for
# labelling purposes and future extensibility.
ENSEMBLE_CTYPE = "ensemble"


def run_lp(pipeline, model, edge_index, edge_type,
           calibration_model=None, num_runs=None):
    """Run test_link_pred num_runs times and aggregate stats."""
    num_runs = num_runs or args.num_runs
    mrr, mean_rank, hits1, hits3, hits10, times = [], [], [], [], [], []

    for run_i in range(num_runs):
        kwargs = dict(
            type=ENSEMBLE_CTYPE,
            model=model,
            valid_edge_index=edge_index,
            valid_edge_type=edge_type,
            mc_samples=args.mc_samples,
        )
        if calibration_model is not None:
            kwargs["calibration_model"] = calibration_model

        t0     = time.perf_counter()
        scores = pipeline.test_link_pred(**kwargs)
        times.append(time.perf_counter() - t0)

        mrr.append(scores["mrr"])
        mean_rank.append(scores["mean_rank"])
        hits1.append(scores["hits@1"])
        hits3.append(scores["hits@3"])
        hits10.append(scores["hits@10"])
        logger.info(
            f"    LP  run {run_i+1:02d}/{num_runs} [ensemble] | "
            f"MRR={scores['mrr']:.4f}  "
            f"H@1={scores['hits@1']:.4f}  "
            f"H@3={scores['hits@3']:.4f}  "
            f"H@10={scores['hits@10']:.4f}"
        )

    return {
        "mrr":            stats(mrr),
        "mean_rank":      stats(mean_rank),
        "hits@1":         stats(hits1),
        "hits@3":         stats(hits3),
        "hits@10":        stats(hits10),
        "inference_time": {**stats(times), "unit": "seconds_per_run"},
    }


def run_uncertainty(pipeline, model, edge_index, edge_type,
                    calibration_model=None, num_runs=None):
    """Run test_uncertainty num_runs times and aggregate stats."""
    num_runs = num_runs or args.num_runs
    ece, brier, ace, times = [], [], [], []

    for run_i in range(num_runs):
        params = {
            "type":       ENSEMBLE_CTYPE,
            "mc_samples": args.mc_samples,
        }
        if calibration_model is not None:
            params["calibration_model"] = calibration_model

        t0     = time.perf_counter()
        scores = pipeline.test_uncertainty(model, edge_index, edge_type, params)
        times.append(time.perf_counter() - t0)

        ece.append(scores["ece"])
        brier.append(scores["brier_score"])
        ace.append(scores["ace"])
        logger.info(
            f"    Unc run {run_i+1:02d}/{num_runs} [ensemble] | "
            f"ECE={scores['ece']:.4f}  "
            f"Brier={scores['brier_score']:.4f}  "
            f"ACE={scores['ace']:.4f}"
        )

    return {
        "ece":            stats(ece),
        "brier_score":    stats(brier),
        "ace":            stats(ace),
        "inference_time": {**stats(times), "unit": "seconds_per_run"},
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    checkpoint_path = osp.join(args.experiment_dir, args.model)

    logger.info("=" * 60)
    logger.info("ENSEMBLE EVALUATION PLAN")
    logger.info(f"  Dataset              : {args.dataset}")
    logger.info(f"  Model                : {args.model}")
    logger.info(f"  Checkpoint           : {checkpoint_path}")
    logger.info(f"  Runs                 : {args.num_runs}  |  MC samples: {args.mc_samples}")
    logger.info(f"  Force re-run         : {args.force}")
    logger.info(f"  Baseline LP          : {args.baseline_lp}")
    logger.info(f"  Baseline Unc         : {args.baseline_unc}")
    logger.info(f"  Calibrated LP        : {args.eval_lp}")
    logger.info(f"  Calibrated Unc       : {args.eval_uncertainty}")
    logger.info(f"  Methods              : {args.methods}")
    logger.info("=" * 60)

    # --- Config ---
    if args.config_path and osp.exists(args.config_path):
        logger.info(f"Loading YAML config from: {args.config_path}")
        config = load_yaml_config(args.config_path)
    else:
        logger.info("No YAML config provided — using default config.")
        config = build_default_config(args.dataset, args.model)

    # Sync mc_samples from CLI into config
    config.get_section("calibration")["mc_samples"] = args.mc_samples

    # --- Data & model ---
    data, dataset = load_data(args.dataset)
    ensemble      = build_ensemble(data, dataset, config)
    pipeline      = EnsemblePipeline(
        ensemble_model=ensemble,
        data=data,
        config=config,
        logger=logger,
    )
    load_checkpoint_weights(pipeline, checkpoint_path)

    edge_index = data.test_edge_index
    edge_type  = data.test_edge_type

    cal_cfg = config.get_section("calibration")

    # ------------------------------------------------------------------
    # BASELINE (no calibration)
    # ------------------------------------------------------------------
    if args.baseline_lp or args.baseline_unc:
        logger.info("\n=== BASELINE (ensemble, no calibration) ===")

        out_path = osp.join(
            args.results_root, args.dataset, args.model,
            "baseline", "baseline", "metrics.json",
        )
        skip, existing = check_existing(
            out_path, need_lp=args.baseline_lp, need_unc=args.baseline_unc
        )

        if not skip:
            lp_stats = (
                run_lp(pipeline, ensemble, edge_index, edge_type)
                if args.baseline_lp
                else existing.get("link_prediction")
            )
            unc_stats = (
                run_uncertainty(pipeline, ensemble, edge_index, edge_type)
                if args.baseline_unc
                else existing.get("uncertainty")
            )
            save_metrics(
                args.dataset, args.model, "baseline", "baseline",
                lp_stats, unc_stats, baseline=True,
            )

    # ------------------------------------------------------------------
    # CALIBRATED — all methods
    # ------------------------------------------------------------------
    if args.eval_lp or args.eval_uncertainty:
        logger.info("\n=== CALIBRATED ===")

        for method in args.methods:
            out_path = osp.join(
                args.results_root, args.dataset, args.model,
                method, ENSEMBLE_CTYPE, "metrics.json",
            )
            skip, existing = check_existing(
                out_path,
                need_lp=args.eval_lp,
                need_unc=args.eval_uncertainty,
            )
            if skip:
                continue

            logger.info(f"\n  --- {method} / {ENSEMBLE_CTYPE} ---")

            # Update calibration config for this run
            cal_cfg["method"] = method
            cal_cfg["type"]   = ENSEMBLE_CTYPE

            calibration_model = pipeline.calibrate_pipeline(
                method=method,
                model=ensemble,
                max_iters=cal_cfg.get("max_iters", 1000),
                lr=cal_cfg.get("learning_rate", 0.01),
            )

            lp_stats = (
                run_lp(
                    pipeline, ensemble, edge_index, edge_type,
                    calibration_model=calibration_model,
                )
                if args.eval_lp
                else existing.get("link_prediction")
            )

            unc_stats = (
                run_uncertainty(
                    pipeline, ensemble, edge_index, edge_type,
                    calibration_model=calibration_model,
                )
                if args.eval_uncertainty
                else existing.get("uncertainty")
            )

            save_metrics(
                args.dataset, args.model, method, ENSEMBLE_CTYPE,
                lp_stats, unc_stats,
            )

    logger.info("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()