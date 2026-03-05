"""
Evaluation metrics and utilities for knowledge graph link prediction.
"""
from random import random

import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.classification.calibration_error import BinaryCalibrationError
from sklearn.metrics import brier_score_loss
from model.ensemble.deep_ensemble import DeepEnsemble
from model.calibrator.isotonic import IsotonicCalibrator
from netcal.metrics import ACE
from sklearn.calibration import calibration_curve
import torch.nn.functional as F

import matplotlib.pyplot as plt


def create_filter_dicts(data):
    """
    Pre-computes hash maps for fast link prediction filtering.
    Returns:
        head_filter: Dict[(tail, rel) -> Set[heads]]
        tail_filter: Dict[(head, rel) -> Set[tails]]
    """
    head_filter = {}
    tail_filter = {}

    edges = [
        (data.train_edge_index, data.train_edge_type),
        (data.valid_edge_index, data.valid_edge_type),
        (data.test_edge_index, data.test_edge_type),
    ]

    for edge_index, edge_type in edges:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        rel = edge_type.tolist()

        for s, d, r in zip(src, dst, rel):
            # Key: (Source, Relation), Value: Set of Dests
            if (s, r) not in tail_filter: tail_filter[(s, r)] = set()
            tail_filter[(s, r)].add(d)

            # Key: (Dest, Relation), Value: Set of Sources
            if (d, r) not in head_filter: head_filter[(d, r)] = set()
            head_filter[(d, r)].add(s)

    return head_filter, tail_filter


@torch.no_grad()
def compute_rank(ranks):
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5

@torch.no_grad()
def compute_mrr_ensemble(train_edge_index, train_edge_type, edge_index, edge_type, data, models: DeepEnsemble,calibration_model=None):
    """
    Optimized MRR computation:
    1. Pre-computes node embeddings (Z) once.
    2. Uses hash maps for O(1) filtering.
    3. Decodes directly using cached Z.
    """
    models.eval()
    device = models.device
    num_nodes = data.num_nodes

    encoded_zs = models.encode(train_edge_index, train_edge_type)
    head_filter, tail_filter = create_filter_dicts(data)
    
    ranks = []

    # Pre-allocate a mask tensor to reuse memory
    mask_container = torch.ones(num_nodes, dtype=torch.bool, device=device)
    all_nodes = torch.arange(num_nodes, device=device)

    iterator = tqdm(range(edge_type.numel()), desc="Computing MRR")
    
    for i in iterator:
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        rel = edge_type[i].item()

        # ========== Tail Prediction ==========
        true_tails = tail_filter.get((src, rel), set())
        
        mask_container.fill_(True)
        
        if true_tails:
            indices = torch.tensor(list(true_tails), device=device)
            mask_container[indices] = False
            

        mask_container[dst] = False
        negatives = all_nodes[mask_container]
        
        tail_candidates = torch.cat([torch.tensor([dst], device=device), negatives])
        
        # Create eval edges
        head_candidates = torch.full_like(tail_candidates, fill_value=src)
        eval_edge_index = torch.stack([head_candidates, tail_candidates], dim=0)
        eval_edge_type = torch.full_like(tail_candidates, fill_value=rel)
        # Use optimized inference
        probs,variance = models.inference_optimised(encoded_zs, eval_edge_index, eval_edge_type)

        if calibration_model is not None:
            if isinstance(calibration_model, IsotonicCalibrator):
                probs = torch.sigmoid(probs)
                probs = calibration_model.predict(probs.detach().cpu().numpy())
                probs = torch.from_numpy(probs).float()
            else:
                probs = torch.sigmoid(calibration_model(probs))
        ranks.append(compute_rank(probs))

        # ========== Head Prediction ==========
        true_heads = head_filter.get((dst, rel), set())
        
        mask_container.fill_(True)
        if true_heads:
            indices = torch.tensor(list(true_heads), device=device)
            mask_container[indices] = False
            
        mask_container[src] = False
        negatives = all_nodes[mask_container]
        
        head_candidates = torch.cat([torch.tensor([src], device=device), negatives])
        tail_candidates = torch.full_like(head_candidates, fill_value=dst)
        
        eval_edge_index = torch.stack([head_candidates, tail_candidates], dim=0)
        eval_edge_type = torch.full_like(head_candidates, fill_value=rel)

        # Use optimized inference
        probs, variance = models.inference_optimised(encoded_zs, eval_edge_index, eval_edge_type)
        if calibration_model is not None:
            if isinstance(calibration_model, IsotonicCalibrator):
                probs = torch.sigmoid(probs)
                probs = calibration_model.predict(probs.detach().cpu().numpy())
                probs = torch.from_numpy(probs).float()

            else:
                probs = torch.sigmoid(calibration_model(probs))

        ranks.append(compute_rank(probs))


    # Calculate final metrics
    ranks_t = torch.tensor(ranks, dtype=torch.float)
    scores = {
        'mrr': (1. / ranks_t).mean().item(),
        'mean_rank': ranks_t.mean().item(),
        'hits@1': (ranks_t <= 1).float().mean().item(),
        'hits@3': (ranks_t <= 3).float().mean().item(),
        'hits@10': (ranks_t <= 10).float().mean().item(),

    }

    return scores

@torch.no_grad()
def compute_mrr_mc_dropout(train_edge_index, train_edge_type,
                            edge_index, edge_type,
                            data, model, mc_samples,
                            calibration_model=None):

    device = train_edge_index.device  # respect whatever device data lives on

    # Encode mc_samples times with dropout active
    all_z = []
    model.encoder.mc_dropout = True
    for _ in range(mc_samples):
        z = model.encode(train_edge_index, train_edge_type)
        all_z.append(z)
    model.encoder.mc_dropout = False

    ranks = []
    variance_true = []

    for i in tqdm(range(edge_type.numel()), desc="Computing MRR"):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # ── TAIL PREDICTION ──────────────────────────────────────────────────
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes, device=device)[tail_mask]
        tail = torch.cat([dst.unsqueeze(0), tail])          # stays on device
        head = torch.full_like(tail, fill_value=src.item()) # stays on device
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type  = torch.full_like(tail, fill_value=rel.item())

        predictions = [model.decode(all_z[k], eval_edge_index, eval_edge_type)
                       for k in range(mc_samples)]

        if calibration_model is not None:
            stacked = torch.stack(predictions, dim=1)  # [num_candidates, mc_samples]
            if isinstance(calibration_model, IsotonicCalibrator):
                probs = calibration_model.predict(
                    torch.sigmoid(stacked).mean(dim=1).cpu().numpy())
                probs = torch.from_numpy(probs).float().to(device)
            else:
                probs = calibration_model(stacked)
        else:
            stacked = torch.stack(predictions, dim=0)  # [mc_samples, num_candidates]
            probs   = torch.sigmoid(stacked).mean(dim=0)

        ranks.append(compute_rank(probs))
        variance_true.append(torch.stack(predictions, dim=0).var(dim=0)[0].item())

        # ── HEAD PREDICTION ──────────────────────────────────────────────────
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes, device=device)[head_mask]
        head = torch.cat([src.unsqueeze(0), head])
        tail = torch.full_like(head, fill_value=dst.item())
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type  = torch.full_like(head, fill_value=rel.item())

        predictions = [model.decode(all_z[k], eval_edge_index, eval_edge_type)
                       for k in range(mc_samples)]

        if calibration_model is not None:
            stacked = torch.stack(predictions, dim=1)
            if isinstance(calibration_model, IsotonicCalibrator):
                probs = calibration_model.predict(
                    torch.sigmoid(stacked).mean(dim=1).cpu().numpy())
                probs = torch.from_numpy(probs).float().to(device)
            else:
                probs = calibration_model(stacked)
        else:
            stacked = torch.stack(predictions, dim=0)
            probs   = torch.sigmoid(stacked).mean(dim=0)

        ranks.append(compute_rank(probs))

    ranks_tensor = torch.tensor(ranks, dtype=torch.float)
    return {
        'mrr':       (1. / ranks_tensor).mean().item(),
        'mean_rank': ranks_tensor.mean().item(),
        'hits@1':    (ranks_tensor <= 1).float().mean().item(),
        'hits@3':    (ranks_tensor <= 3).float().mean().item(),
        'hits@10':   (ranks_tensor <= 10).float().mean().item(),
    }


def compute_uncertainty(y_true, y_probs):
    """
    Computes Brier Score, ECE (weighted), and generates a Reliability Diagram.
    
    Parameters:
    y_true: Ground truth labels (0 or 1)
    y_probs: Predicted probabilities (0.0 to 1.0)
    
    Returns:
    dict: Contains scalar scores and the matplotlib figure.
    """
    # Convert to tensors properly, handling both tensor and numpy inputs
    if isinstance(y_true, torch.Tensor):
        y_true_tensor = y_true.detach().cpu()
    else:
        y_true_tensor = torch.tensor(y_true).cpu()
    
    if isinstance(y_probs, torch.Tensor):
        y_probs_tensor = y_probs.detach().cpu()
    else:
        y_probs_tensor = torch.tensor(y_probs).cpu()

    ece_metric = BinaryCalibrationError(n_bins=10, norm='l1')
    
    ace = ACE(bins=10)
    score = ace.measure(y_probs_tensor.numpy(), y_true_tensor.numpy() )

    ece_metric.update(y_probs_tensor, y_true_tensor)
   
    ece_score = ece_metric.compute().item()
    brier_score = brier_score_loss(y_true, y_probs)

    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10, strategy='quantile')

    return {
        'brier_score': brier_score,
        'ece': ece_score,
        'ace': score,
        'prob_true': prob_true,
        'prob_pred': prob_pred,
    }


@torch.no_grad()
def compute_mrr(edge_index, edge_type, data, model, return_probs= False, negatives = 1, calibration_model=None):
    model.eval()
    z = model.encode(data.edge_index, data.edge_type)
    ranks = []
    diffs = []
    # For calibration
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # --- TAIL PREDICTION ---
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        if calibration_model is not None:
            if isinstance(calibration_model, IsotonicCalibrator):
                probs = calibration_model.predict(torch.sigmoid(out).detach().cpu().numpy())
                probs = torch.from_numpy(probs).float()
            else:
                probs = torch.sigmoid(calibration_model(out))
        else:
            probs = torch.sigmoid(out)

        first_probs, first_indices = probs.topk(2)
        diff = first_probs[0] - first_probs[1]
        diffs.append(diff.item())
        ranks.append(compute_rank(probs))


        # --- HEAD PREDICTION ---
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        
        if calibration_model is not None:
            if isinstance(calibration_model, IsotonicCalibrator):
                probs = calibration_model.predict(torch.sigmoid(out).detach().cpu().numpy())
                probs = torch.from_numpy(probs).float()
                
            else:
                probs = torch.sigmoid(calibration_model(out))
        else:
            probs = torch.sigmoid(out)

        first_probs, first_indices = probs.topk(2)
        diff = first_probs[0] - first_probs[1]
        diffs.append(diff.item())

        ranks.append(compute_rank(probs))



    ranks_tensor = torch.tensor(ranks, dtype=torch.float)

    print_stats(diffs,"(Diff between Top-2 Probabilities)")
    scores = {
        'mrr': (1. / ranks_tensor).mean().item(),
        'mean_rank': ranks_tensor.mean().item(),
        'hits@1': (ranks_tensor <= 1).float().mean().item(),
        'hits@3': (ranks_tensor <= 3).float().mean().item(),
        'hits@10': (ranks_tensor <= 10).float().mean().item(),
    }

    return scores


def print_stats(probab,addtional=""):

    if len(probab) == 0:
        print("No probabilities to analyze.")
        return
    print(f"Average Confidence of Difference: {np.mean(probab)}")
    print(f"Median Confidence of Difference: {np.median(probab)}")
    print(f"Count of Zero Differences: {(np.array(probab) == 0).sum()}")
    print(f"Max Difference: {np.max(probab)}")
    print(f"Min Difference: {np.min(probab)}")
    print(f"Total Samples: {len(probab)}")

    # Plot histogram of confidence scores
    plt.figure(figsize=(10, 6))
    plt.hist(probab, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Diffrences in top two probabilities')
    plt.grid(True, alpha=0.3)
    filename = f'difference_top2_confidence_histogram{np.random.randint(0, 1000)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved as '{filename}'")


