# eval_lp_uncertainty_20runs.py

import os
import sys
import json
import os.path as osp
import argparse
import time

import numpy as np
import torch
import logging

# ----------------------------------------------------------------------
# Command line parsing
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate LP + Uncertainty with smart baseline control")
parser.add_argument("--dataset",                default="WN18RR", choices=["WN18RR", "FB15k-237"])
parser.add_argument("--model",                  required=True)
parser.add_argument("--experiment-dir",         default="new_models/experiment_1")
parser.add_argument("--results-root",           default="results_lp_uncertainty")
parser.add_argument("--num-runs",               type=int, default=20)
parser.add_argument("--mc-samples",             type=int, default=5)

# --- Granular flags ---
parser.add_argument("--baseline-lp",           action="store_true", default=False,
                    help="Run LP baseline (no calibration, standard)")
parser.add_argument("--baseline-unc-standard", action="store_true", default=False,
                    help="Run uncertainty baseline with type=standard")
parser.add_argument("--baseline-unc-mc",       action="store_true", default=False,
                    help="Run uncertainty baseline with type=mc_dropout")
parser.add_argument("--eval-lp",               action="store_true", default=False,
                    help="Run LP for all calibration methods x types")
parser.add_argument("--eval-uncertainty",      action="store_true", default=False,
                    help="Run uncertainty for all calibration methods x types")

# --- Master switches ---
parser.add_argument("--baseline",              action="store_true", default=False,
                    help="Run all baselines (LP + unc standard + unc mc_dropout)")
parser.add_argument("--eval-all",              action="store_true", default=False,
                    help="Run all calibrated evaluations (LP + uncertainty)")
parser.add_argument("--all",                   action="store_true", default=False,
                    help="Run everything: all baselines + all calibrated")

# --- Scoped convenience flags ---
parser.add_argument("--only-lp",              action="store_true",
                    help="Baseline LP + calibrated LP only (no uncertainty)")
parser.add_argument("--only-uncertainty",      action="store_true",
                    help="Both unc baselines + calibrated uncertainty only (no LP)")
parser.add_argument("--only-baseline",         action="store_true",
                    help="All baselines only (no calibrated evaluation)")
parser.add_argument("--only-calibrated",       action="store_true",
                    help="All calibrated evaluations only (no baselines)")

# --- Subset filters ---
parser.add_argument("--methods", nargs="+",
                    default=["scalar", "platt_scaling", "isotonic_regression"],
                    help="Calibration methods to evaluate (default: all)")
parser.add_argument("--ctypes",  nargs="+",
                    default=["standard", "mc_dropout"],
                    help="Calibration types to evaluate (default: both)")

# --- Force re-run ---
parser.add_argument("--force", action="store_true", default=False,
                    help="Ignore existing results and re-run everything")

args = parser.parse_args()

# === SMART LOGIC ===
if args.all:
    args.baseline = True
    args.eval_all = True

if args.baseline:
    args.baseline_lp           = True
    args.baseline_unc_standard = True
    args.baseline_unc_mc       = True

if args.eval_all:
    args.eval_lp          = True
    args.eval_uncertainty = True

if args.only_lp:
    args.baseline_lp           = True
    args.eval_lp               = True
    args.baseline_unc_standard = False
    args.baseline_unc_mc       = False
    args.eval_uncertainty      = False

if args.only_uncertainty:
    args.baseline_unc_standard = True
    args.baseline_unc_mc       = True
    args.eval_uncertainty      = True
    args.baseline_lp           = False
    args.eval_lp               = False

if args.only_baseline:
    args.baseline_lp           = True
    args.baseline_unc_standard = True
    args.baseline_unc_mc       = True
    args.eval_lp               = False
    args.eval_uncertainty      = False

if args.only_calibrated:
    args.eval_lp               = True
    args.eval_uncertainty      = True
    args.baseline_lp           = False
    args.baseline_unc_standard = False
    args.baseline_unc_mc       = False

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'eval_{args.dataset}_{args.model}_log.txt')
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from torch_geometric.nn import GAE
from model.encoder.model import RGCN
from model.decoder.distmult import DistMult
from misc.rel_link_pred_dataset import RelLinkPredDataset
from model.trainer.pipeline import Pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
class Config:
    def __init__(self, config_dict):
        self._config = config_dict

    def get_section(self, section):
        return self._config.get(section, {})

    def get(self, section, key=None):
        if key is None:
            return self.get_section(section)
        return self._config.get(section, {}).get(key)


def build_config(dataset_name, model_name):
    if dataset_name == "FB15k-237":
        return Config({
            'dataset': {'name': 'FB15k-237', 'path': './dataset/fb15k237'},
            'model': {
                'encoder': {'type': 'RGCN', 'hidden_layer_size': 500, 'embedding_dim': 500,
                            'dropout': 0.2, 'num_bases': 5, 'bases_enabled': False},
                'decoder': {'type': 'DistMult', 'l2_penalty': 0.001, 'w_gain': False, 'b_init': False}
            },
            'ensemble': {'enabled': False, 'num_models': 5},
            'training': {
                'epochs': 10000,
                'sampling': {'negative_sampling_ratio': 3, 'edge_dropout': 0.2},
                'optimiser': {'learning_rate': 0.01, 'weight_decay': 0},
                'evaluation_frequency': 500,
                'early stopping': {'enabled': True, 'patience': 5, 'delta': 0.001},
                'load_model': True, 'save_model': False,
                'checkpoint_path': f'./new_models/experiment_1/{model_name}',
                'test': True,
                'label_smoothing': {'positive': 0.9, 'negative': 0.05}
            },
            'calibration': {
                'enabled': True, 'type': 'mc_dropout', 'mc_samples': args.mc_samples,
                'method': 'isotonic_regression', 'max_iters': 1000,
                'lambda': 0.5, 'learning_rate': 0.01
            }
        })

    # Default: WN18RR
    return Config({
        'dataset': {'name': dataset_name},
        'model': {
            'encoder': {'type': 'RGCN', 'hidden_layer_size': 500, 'embedding_dim': 500,
                        'dropout': 0.2, 'num_bases': 5, 'bases_enabled': True},
            'decoder': {'type': 'DistMult', 'l2_penalty': 0.001, 'w_gain': False, 'b_init': False}
        },
        'training': {
            'epochs': 10000,
            'sampling': {'negative_sampling_ratio': 3, 'edge_dropout': 0.2},
            'optimiser': {'learning_rate': 0.01, 'weight_decay': 0},
            'evaluation_frequency': 100,
            'early stopping': {'enabled': True, 'patience': 10, 'delta': 0.001},
            'load_model': True, 'save_model': True,
            'checkpoint_path': f'./new_models/experiment_1/{model_name}',
            'test': True,
            'label_smoothing': {'positive': 0.9, 'negative': 0.05}
        },
        'calibration': {
            'enabled': True, 'type': 'standard', 'mc_samples': args.mc_samples,
            'method': 'scalar', 'max_iters': 10000,
            'lambda': 0.5, 'learning_rate': 0.01
        }
    })

# ----------------------------------------------------------------------
# Data & Model
# ----------------------------------------------------------------------
def load_data(dataset_name):
    path = osp.join('.', 'data', 'RLPD')
    dataset = RelLinkPredDataset(path, dataset_name)
    data = dataset[0]
    data['num_relations'] = dataset.num_relations
    data = data.to(device)
    return data


def build_model(data, config):
    bases_enabled = config.get('model', 'encoder').get('bases_enabled', True)
    model_config = {
        'encoder': {'embedding_dim': 500, 'hidden_layer_size': 500, 'num_bases': 5,
                    'dropout': 0.2, 'bases_enabled': bases_enabled},
        'decoder': {'margin': 1.0, 'sparse': False, 'calibration': 'none',
                    'l2_penalty': 0.001, 'w_gain': False, 'b_init': False}
    }
    encoder = RGCN(num_nodes=data.num_nodes, num_relations=data['num_relations'],
                   model_config=model_config)
    decoder = DistMult(num_nodes=data.num_nodes, num_relations=data['num_relations'] // 2,
                       embedding_dim=500, margin=1.0, sparse=False, calibration='none')
    return GAE(encoder=encoder, decoder=decoder).to(device)


def load_checkpoint_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info(f"Loaded: {checkpoint_path}, epoch={checkpoint.get('epoch', 'N/A')}")

# ----------------------------------------------------------------------
# Result helpers
# ----------------------------------------------------------------------
def check_existing(out_path, need_lp, need_unc):
    """
    Returns (skip, existing_dict).
    skip=True  → file is complete, nothing to run.
    skip=False → file missing, corrupt, or partial; existing_dict has any
                 already-computed sections for merging.
    """
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
    if need_unc and "uncertainty"      not in existing:
        missing.append("uncertainty")

    if missing:
        logger.info(f"  [PARTIAL] {out_path} — missing: {missing}, will re-run")
        return False, existing

    logger.info(f"  [SKIP] already complete: {out_path}")
    return True, existing


def stats(lst):
    lst = [float(x) for x in lst]   # safely converts tensors → Python floats
    return {'runs': lst, 'mean': float(np.mean(lst)), 'var': float(np.var(lst, ddof=1))}


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
def run_lp(pipeline, model, edge_index, edge_type, ctype,
           calibration_model=None, num_runs=None):
    """
    Calls pipeline.test_link_pred() for num_runs and aggregates stats.
    calibration_model=None → uncalibrated baseline.
    """
    num_runs = num_runs or args.num_runs
    mrr, mean_rank, hits1, hits3, hits10, times = [], [], [], [], [], []

    for run_i in range(num_runs):
        kwargs = dict(
            type=ctype,
            model=model,
            valid_edge_index=edge_index,
            valid_edge_type=edge_type,
            mc_samples=args.mc_samples,
        )
        if calibration_model is not None:
            kwargs['calibration_model'] = calibration_model

        t0     = time.perf_counter()
        scores = pipeline.test_link_pred(**kwargs)
        times.append(time.perf_counter() - t0)

        mrr.append(scores['mrr'])
        mean_rank.append(scores['mean_rank'])
        hits1.append(scores['hits@1'])
        hits3.append(scores['hits@3'])
        hits10.append(scores['hits@10'])
        logger.info(f"    LP  run {run_i+1:02d}/{num_runs} [{ctype}] | "
                    f"MRR={scores['mrr']:.4f}  "
                    f"H@1={scores['hits@1']:.4f}  "
                    f"H@3={scores['hits@3']:.4f}  "
                    f"H@10={scores['hits@10']:.4f}")

    return {
        'mrr':            stats(mrr),
        'mean_rank':      stats(mean_rank),
        'hits@1':         stats(hits1),
        'hits@3':         stats(hits3),
        'hits@10':        stats(hits10),
        'inference_time': {**stats(times), 'unit': 'seconds_per_run'}
    }


def run_uncertainty(pipeline, model, edge_index, edge_type, ctype,
                    calibration_model=None, num_runs=None):
    """
    Calls pipeline.test_uncertainty() for num_runs and aggregates stats.
    calibration_model=None → no calibration injected (baseline).
    """
    num_runs = num_runs or args.num_runs
    ece, brier, ace, times = [], [], [], []

    for run_i in range(num_runs):
        params = {'type': ctype, 'mc_samples': args.mc_samples}
        if calibration_model is not None:
            params['calibration_model'] = calibration_model

        t0     = time.perf_counter()
        scores = pipeline.test_uncertainty(model, edge_index, edge_type, params)
        times.append(time.perf_counter() - t0)

        ece.append(scores['ece'])
        brier.append(scores['brier_score'])
        ace.append(scores['ace'])
        logger.info(f"    Unc run {run_i+1:02d}/{num_runs} [{ctype}] | "
                    f"ECE={scores['ece']:.4f}  "
                    f"Brier={scores['brier_score']:.4f}  "
                    f"ACE={scores['ace']:.4f}")

    return {
        'ece':            stats(ece),
        'brier_score':    stats(brier),
        'ace':            stats(ace),
        'inference_time': {**stats(times), 'unit': 'seconds_per_run'}
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    checkpoint_path = osp.join(args.experiment_dir, args.model)

    logger.info("=" * 60)
    logger.info("EVALUATION PLAN")
    logger.info(f"  Dataset              : {args.dataset}")
    logger.info(f"  Model                : {args.model}")
    logger.info(f"  Runs                 : {args.num_runs}  |  MC samples: {args.mc_samples}")
    logger.info(f"  Force re-run         : {args.force}")
    logger.info(f"  Baseline LP          : {args.baseline_lp}")
    logger.info(f"  Baseline Unc standard: {args.baseline_unc_standard}")
    logger.info(f"  Baseline Unc mc_drop : {args.baseline_unc_mc}")
    logger.info(f"  Calibrated LP        : {args.eval_lp}")
    logger.info(f"  Calibrated Unc       : {args.eval_uncertainty}")
    logger.info(f"  Methods              : {args.methods}")
    logger.info(f"  Types                : {args.ctypes}")
    logger.info("=" * 60)

    # Setup — config before model so bases_enabled is correct
    data     = load_data(args.dataset)
    config   = build_config(args.dataset, args.model)
    model    = build_model(data, config)
    load_checkpoint_weights(model, checkpoint_path)
    pipeline = Pipeline(model, data, config, logger)
    cal_cfg  = config.get_section('calibration')

    edge_index = data.test_edge_index
    edge_type  = data.test_edge_type

    # ------------------------------------------------------------------
    # BASELINE — standard (LP + uncertainty)
    # ------------------------------------------------------------------
    if args.baseline_lp or args.baseline_unc_standard:
        logger.info("\n=== BASELINE (standard) ===")

        out_path = osp.join(args.results_root, args.dataset, args.model,
                            "baseline", "baseline", "metrics.json")
        skip, existing = check_existing(out_path,
                                        need_lp=args.baseline_lp,
                                        need_unc=args.baseline_unc_standard)
        if not skip:
            # Run only missing sections; reuse existing ones from disk
            lp_stats  = run_lp(
                pipeline, model, edge_index, edge_type, ctype='standard'
            ) if args.baseline_lp else existing.get("link_prediction")

            unc_stats = run_uncertainty(
                pipeline, model, edge_index, edge_type, ctype='standard'
            ) if args.baseline_unc_standard else existing.get("uncertainty")

            save_metrics(args.dataset, args.model, "baseline", "baseline",
                         lp_stats, unc_stats, baseline=True)

    # ------------------------------------------------------------------
    # BASELINE — mc_dropout (uncertainty only)
    # ------------------------------------------------------------------
    if args.baseline_unc_mc:
        logger.info("\n=== BASELINE (mc_dropout) ===")

        out_path = osp.join(args.results_root, args.dataset, args.model,
                            "baseline", "mc_dropout", "metrics.json")
        skip, _ = check_existing(out_path, need_lp=False, need_unc=True)

        if not skip:
            unc_stats_mc = run_uncertainty(
                pipeline, model, edge_index, edge_type, ctype='mc_dropout'
            )
            save_metrics(args.dataset, args.model, "baseline", "mc_dropout",
                         unc_stats=unc_stats_mc, baseline=True)

    # ------------------------------------------------------------------
    # CALIBRATED — all methods x types
    # ------------------------------------------------------------------
    if args.eval_lp or args.eval_uncertainty:
        logger.info("\n=== CALIBRATED ===")

        for method in args.methods:
            for ctype in args.ctypes:
                out_path = osp.join(args.results_root, args.dataset, args.model,
                                    method, ctype, "metrics.json")
                skip, existing = check_existing(out_path,
                                                need_lp=args.eval_lp,
                                                need_unc=args.eval_uncertainty)
                if skip:
                    continue

                logger.info(f"\n  --- {method} / {ctype} ---")

                # Update config so calibrate_pipeline reads correct type/method
                cal_cfg['method'] = method
                cal_cfg['type']   = ctype

                calibration_model = pipeline.calibrate_pipeline(
                    method=method, model=model,
                    max_iters=cal_cfg.get('max_iters', 10000),
                    lr=cal_cfg.get('learning_rate', 0.01)
                )

                # Run only missing sections; reuse existing ones from disk
                lp_stats  = run_lp(
                    pipeline, model, edge_index, edge_type,
                    ctype=ctype, calibration_model=calibration_model
                ) if args.eval_lp else existing.get("link_prediction")

                unc_stats = run_uncertainty(
                    pipeline, model, edge_index, edge_type,
                    ctype=ctype, calibration_model=calibration_model
                ) if args.eval_uncertainty else existing.get("uncertainty")

                save_metrics(args.dataset, args.model, method, ctype,
                             lp_stats, unc_stats)

    logger.info("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
