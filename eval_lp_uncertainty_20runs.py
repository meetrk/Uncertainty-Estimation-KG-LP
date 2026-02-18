# eval_lp_uncertainty_20runs.py - SMART BASELINE CONTROL

import os
import sys
import json
import os.path as osp
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import logging

# ----------------------------------------------------------------------
# Command line parsing - SMART BASELINE FLAGS
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate LP + Uncertainty (smart baseline)")
parser.add_argument("--dataset", default="WN18RR", choices=["WN18RR", "FB15k-237"])
parser.add_argument("--model", required=True)
parser.add_argument("--experiment-dir", default="new_models/experiment_1")
parser.add_argument("--results-root", default="results_lp_uncertainty")
parser.add_argument("--num-runs", type=int, default=20)
parser.add_argument("--mc-samples", type=int, default=5)

# Smart evaluation controls
parser.add_argument("--eval-lp", action="store_true", default=False)
parser.add_argument("--eval-uncertainty", action="store_true", default=False)
parser.add_argument("--baseline-lp", action="store_true", default=False)
parser.add_argument("--baseline-uncertainty", action="store_true", default=False)
parser.add_argument("--baseline", action="store_true", default=False)  # Master switch

# ++ NEW: MC Dropout baseline flag
parser.add_argument("--baseline-uncertainty-mc", action="store_true", default=False,
                    help="Run uncertainty baseline with mc_dropout type (raw MC Dropout, no calibration)")

# Convenience flags
parser.add_argument("--only-lp", action="store_true")
parser.add_argument("--only-uncertainty", action="store_true")
parser.add_argument("--only-baseline", action="store_true")

args = parser.parse_args()

# === SMART LOGIC ===
if args.baseline:
    args.baseline_lp = True
    args.baseline_uncertainty = True
    args.baseline_uncertainty_mc = True   # ++ NEW: master switch now also triggers MC dropout baseline
    args.eval_lp = False
    args.eval_uncertainty = False

if args.only_lp:
    args.eval_lp = True
    args.baseline_lp = False

if args.only_uncertainty:
    args.eval_uncertainty = True
    args.baseline_uncertainty = False

if args.only_baseline:
    args.baseline_lp = True
    args.baseline_uncertainty = True
    args.baseline_uncertainty_mc = True   # ++ NEW: only-baseline also triggers MC dropout baseline
    args.eval_lp = False
    args.eval_uncertainty = False

# ----------------------------------------------------------------------
# Imports and basic setup
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from torch_geometric.nn import GAE
from model.encoder.model import RGCN
from model.decoder.distmult import DistMult
from misc.rel_link_pred_dataset import RelLinkPredDataset
from model.trainer.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'eval_{args.dataset}_{args.model}_log.txt')
    ]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Config wrapper (unchanged)
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
    return Config({
        'dataset': {'name': dataset_name},
        'model': {
            'encoder': {'type': 'RGCN', 'hidden_layer_size': 500, 'embedding_dim': 500, 'dropout': 0.2, 'num_bases': 5, 'bases_enabled': True},
            'decoder': {'type': 'DistMult', 'l2_penalty': 0.001, 'w_gain': False, 'b_init': False}
        },
        'training': {
            'epochs': 10000, 'sampling': {'negative_sampling_ratio': 3, 'edge_dropout': 0.2},
            'optimiser': {'learning_rate': 0.01, 'weight_decay': 0},
            'evaluation_frequency': 100, 'early_stopping': {'enabled': True, 'patience': 10, 'delta': 0.001},
            'load_model': True, 'save_model': True, 'checkpoint_path': f'./new_models/experiment_1/{model_name}',
            'test': True, 'label_smoothing': {'positive': 0.9, 'negative': 0.05}
        },
        'calibration': {
            'enabled': True, 'type': 'standard', 'mc_samples': args.mc_samples,
            'method': 'scalar', 'max_iters': 10000, 'lambda': 0.5, 'learning_rate': 0.01
        }
    })

# ----------------------------------------------------------------------
# Data/Model (unchanged)
# ----------------------------------------------------------------------
def load_data(dataset_name):
    path = osp.join('.', 'data', 'RLPD')
    dataset = RelLinkPredDataset(path, dataset_name)
    data = dataset[0]
    data['num_relations'] = dataset.num_relations
    return data


def build_model(data):
    model_config = {
        'encoder': {'embedding_dim': 500, 'hidden_layer_size': 500, 'num_bases': 5, 'dropout': 0.2, 'bases_enabled': True},
        'decoder': {'margin': 1.0, 'sparse': False, 'calibration': 'none', 'l2_penalty': 0.001, 'w_gain': False, 'b_init': False}
    }
    encoder = RGCN(num_nodes=data.num_nodes, num_relations=data['num_relations'], model_config=model_config)
    decoder = DistMult(num_nodes=data.num_nodes, num_relations=data['num_relations']//2, embedding_dim=500, margin=1.0, sparse=False, calibration='none')
    return GAE(encoder=encoder, decoder=decoder).to(device)


def load_checkpoint_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info(f"Model loaded from {checkpoint_path}, epoch={checkpoint.get('epoch', 'N/A')}")
    return checkpoint

# ----------------------------------------------------------------------
# Metric functions (unchanged)
# ----------------------------------------------------------------------
def multi_run_uncertainty(pipe, model, test_edge_index, test_edge_type, uncertainty_params, calibration_model=None, num_runs=None, baseline=False):
    num_runs = num_runs or args.num_runs
    ece, brier, ace, inference_times = [], [], [], []

    for _ in range(num_runs):
        start_time = time.perf_counter()
        params = uncertainty_params.copy()
        if not baseline and calibration_model:
            params['calibration_model'] = calibration_model
        m = pipe.test_uncertainty(model, test_edge_index, test_edge_type, params)
        inference_times.append(time.perf_counter() - start_time)
        ece.append(m['ece'])
        brier.append(m['brier_score'])
        ace.append(m['ace'])

    return {
        'ece':          {'runs': ece,   'mean': float(np.mean(ece)),   'var': float(np.var(ece,   ddof=1))},
        'brier_score':  {'runs': brier, 'mean': float(np.mean(brier)), 'var': float(np.var(brier, ddof=1))},
        'ace':          {'runs': ace,   'mean': float(np.mean(ace)),   'var': float(np.var(ace,   ddof=1))},
        'inference_time': {'runs': inference_times, 'mean': float(np.mean(inference_times)), 'var': float(np.var(inference_times, ddof=1)), 'unit': 'seconds_per_run'}
    }


def multi_run_link_pred(pipe, data, num_runs=None):
    num_runs = num_runs or args.num_runs
    mrr, mean_rank, hits1, hits3, hits10, inference_times = [], [], [], [], [], []

    for _ in range(num_runs):
        start_time = time.perf_counter()
        _, test_scores = pipe.test(test=True)
        inference_times.append(time.perf_counter() - start_time)
        mrr.append(test_scores['mrr'])
        mean_rank.append(test_scores['mean_rank'])
        hits1.append(test_scores['hits@1'])
        hits3.append(test_scores['hits@3'])
        hits10.append(test_scores['hits@10'])

    return {
        'mrr':        {'runs': mrr,        'mean': float(np.mean(mrr)),        'var': float(np.var(mrr,        ddof=1))},
        'mean_rank':  {'runs': mean_rank,  'mean': float(np.mean(mean_rank)),  'var': float(np.var(mean_rank,  ddof=1))},
        'hits@1':     {'runs': hits1,      'mean': float(np.mean(hits1)),      'var': float(np.var(hits1,      ddof=1))},
        'hits@3':     {'runs': hits3,      'mean': float(np.mean(hits3)),      'var': float(np.var(hits3,      ddof=1))},
        'hits@10':    {'runs': hits10,     'mean': float(np.mean(hits10)),     'var': float(np.var(hits10,     ddof=1))},
        'inference_time': {'runs': inference_times, 'mean': float(np.mean(inference_times)), 'var': float(np.var(inference_times, ddof=1)), 'unit': 'seconds_per_run'}
    }


def save_metrics(dataset_name, model_name, method, ctype, lp_stats=None, unc_stats=None, baseline=False):
    out_dir = osp.join(args.results_root, dataset_name, model_name, method, "baseline" if baseline else ctype)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, "metrics.json")

    result = {
        "dataset": dataset_name, "model_name": model_name,
        "calibration_method": "none" if baseline else method,
        "calibration_type": "baseline" if baseline else ctype,
        "num_runs": args.num_runs, "mc_samples": args.mc_samples, "is_baseline": baseline
    }

    if lp_stats:  result["link_prediction"] = lp_stats
    if unc_stats: result["uncertainty"] = unc_stats

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path} (baseline={baseline})")


# ++ NEW: separate save for MC dropout baseline to avoid colliding with
#         the standard baseline's "baseline/baseline/" output path
def save_metrics_mc_baseline(dataset_name, model_name, unc_stats):
    out_dir = osp.join(args.results_root, dataset_name, model_name, "baseline", "mc_dropout")
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, "metrics.json")

    result = {
        "dataset": dataset_name, "model_name": model_name,
        "calibration_method": "none",
        "calibration_type": "baseline_mc_dropout",
        "num_runs": args.num_runs, "mc_samples": args.mc_samples,
        "is_baseline": True,
        "uncertainty": unc_stats
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path} (baseline=True, type=mc_dropout)")


# ----------------------------------------------------------------------
# Main - SMART EXECUTION
# ----------------------------------------------------------------------
def main():
    checkpoint_path = osp.join(args.experiment_dir, args.model)

    logger.info(f"=== Evaluation Plan ===")
    logger.info(f"LP (calib): {args.eval_lp}, LP (baseline): {args.baseline_lp}")
    logger.info(f"Unc (calib): {args.eval_uncertainty}, Unc (baseline/standard): {args.baseline_uncertainty}, Unc (baseline/mc_dropout): {args.baseline_uncertainty_mc}")  # ++ NEW

    data = load_data(args.dataset)
    model = build_model(data)
    load_checkpoint_weights(model, checkpoint_path)
    config = build_config(args.dataset, args.model)
    pipeline = Pipeline(model, data, config, logger)

    # === BASELINE (standard) — unchanged ===
    if args.baseline_lp or args.baseline_uncertainty:
        logger.info("=== RUNNING BASELINE (standard) ===")
        baseline_lp = multi_run_link_pred(pipeline, data) if args.baseline_lp else None
        baseline_unc = multi_run_uncertainty(
            pipeline, model, data.test_edge_index, data.test_edge_type,
            {'type': 'standard', 'mc_samples': args.mc_samples}, baseline=True
        ) if args.baseline_uncertainty else None
        save_metrics(args.dataset, args.model, "baseline", "baseline", baseline_lp, baseline_unc, baseline=True)

    # ++ NEW: BASELINE (mc_dropout)
    if args.baseline_uncertainty_mc:
        logger.info("=== RUNNING BASELINE (mc_dropout) ===")
        baseline_unc_mc = multi_run_uncertainty(
            pipeline, model, data.test_edge_index, data.test_edge_type,
            {'type': 'mc_dropout', 'mc_samples': args.mc_samples}, baseline=True  # baseline=True → no calibration_model injected
        )
        save_metrics_mc_baseline(args.dataset, args.model, baseline_unc_mc)

    # === CALIBRATED — unchanged ===
    if args.eval_lp or args.eval_uncertainty:
        logger.info("=== RUNNING CALIBRATED ===")
        methods = ["scalar", "platt_scaling", "isotonic_regression", "relation_calibrator"]
        types = ["standard", "mc_dropout"]

        for method in methods:
            for ctype in types:
                logger.info(f"  {method}/{ctype}")
                cal_cfg = config.get_section('calibration')
                cal_cfg['method'] = method
                cal_cfg['type'] = ctype

                calibration_model = pipeline.calibrate_pipeline(method=method, model=model,
                    max_iters=cal_cfg['max_iters'], lr=cal_cfg['learning_rate'])

                lp_stats = multi_run_link_pred(pipeline, data) if args.eval_lp else None
                unc_stats = multi_run_uncertainty(
                    pipeline, model, data.test_edge_index, data.test_edge_type,
                    {'type': ctype, 'mc_samples': args.mc_samples}, calibration_model
                ) if args.eval_uncertainty else None

                save_metrics(args.dataset, args.model, method, ctype, lp_stats, unc_stats)

    logger.info("=== COMPLETE ===")


if __name__ == "__main__":
    main()
