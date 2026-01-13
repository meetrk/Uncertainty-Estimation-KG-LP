"""
Evaluation metrics and utilities for knowledge graph link prediction.
"""
import torch
import numpy as np
from typing import  List
from tqdm import tqdm
from torchmetrics.classification.calibration_error import BinaryCalibrationError
from sklearn.metrics import brier_score_loss
from model.ensemble.deep_ensemble import DeepEnsemble
from netcal.metrics import ACE
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt



def mean_reciprocal_rank(ranks: List[int]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    return float(np.mean([1.0 / rank for rank in ranks]))


def hits_at_k(ranks: List[int], k: int) -> float:
    """Calculate Hits@K metric."""
    return float(np.mean([1.0 if rank <= k else 0.0 for rank in ranks]))

def create_filter_dicts(data):
    """
    Pre-computes hash maps for fast link prediction filtering.
    Returns:
        head_filter: Dict[(tail, rel) -> Set[heads]]
        tail_filter: Dict[(head, rel) -> Set[tails]]
    """
    head_filter = {}
    tail_filter = {}

    # Combine all edges (train, val, test)
    # Move to CPU list for faster Python dictionary insertion
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
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr( edge_index, edge_type, data, model):

    z = model.encode(data.edge_index, data.edge_type)

    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
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
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
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
        rank = compute_rank(out)
        ranks.append(rank)

    scores = {
        'mrr' : (1. / torch.tensor(ranks, dtype=torch.float)).mean(),
        'mean_rank': torch.tensor(ranks, dtype=torch.float).mean(),
        'hits@1': (torch.tensor(ranks, dtype=torch.float) <= 1).float().mean(),
        'hits@3': (torch.tensor(ranks, dtype=torch.float) <= 3).float().mean(),
        'hits@10': (torch.tensor(ranks, dtype=torch.float) <= 10).float().mean(),
    }

    return scores

@torch.no_grad()
def compute_mrr_ensemble(train_edge_index, train_edge_type, edge_index, edge_type, data, models: DeepEnsemble):
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

    iterator = tqdm(range(edge_type.numel()), desc="Computing MRR (Optimized)")
    
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
            
        # Mask out the target from the general candidates to avoid duplication
        mask_container[dst] = False
        
        # Get all negatives
        negatives = all_nodes[mask_container]
        
        tail_candidates = torch.cat([torch.tensor([dst], device=device), negatives])
        
        # Create eval edges
        head_candidates = torch.full_like(tail_candidates, fill_value=src)
        eval_edge_index = torch.stack([head_candidates, tail_candidates], dim=0)
        eval_edge_type = torch.full_like(tail_candidates, fill_value=rel)
        # Use optimized inference
        score = models.inference_optimised(encoded_zs, eval_edge_index, eval_edge_type)
        ranks.append(compute_rank(score))

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
        score = models.inference_optimised(encoded_zs, eval_edge_index, eval_edge_type)
        ranks.append(compute_rank(score))

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
def compute_mrr_mc_dropout(train_edge_index, train_edge_type, edge_index, edge_type, data, model, mc_samples=10):

    # OPTIMIZATION: Encode graph once per model and cache
    all_z = []
    model.encoder.mc_dropout = True
    for _ in range(mc_samples):
        z = model.encode(train_edge_index, train_edge_type)
        all_z.append(z)
    model.encoder.mc_dropout = False

    ranks = []
    for i in tqdm(range(edge_type.numel()), desc="Computing MRR"):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
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

        # Use cached encodings for prediction
        predictions = []
        for model_idx in range(mc_samples):
            pred = model.decode(all_z[model_idx], eval_edge_index, eval_edge_type)
            predictions.append(pred)
        
        out = torch.mean(torch.stack(predictions, dim=0), dim=0)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
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

        # Use cached encodings for prediction
        predictions = []
        for model_idx in range(mc_samples):
            pred = model.decode(all_z[model_idx], eval_edge_index, eval_edge_type)
            predictions.append(pred)
        
        out = torch.mean(torch.stack(predictions, dim=0), dim=0)
        rank = compute_rank(out)
        ranks.append(rank)

    scores = {
        'mrr': (1. / torch.tensor(ranks, dtype=torch.float)).mean(),
        'mean_rank': torch.tensor(ranks, dtype=torch.float).mean(),
        'hits@1': (torch.tensor(ranks, dtype=torch.float) <= 1).float().mean(),
        'hits@3': (torch.tensor(ranks, dtype=torch.float) <= 3).float().mean(),
        'hits@10': (torch.tensor(ranks, dtype=torch.float) <= 10).float().mean(),
    }

    return scores

def compute_uncertainty(y_true, y_probs):
    """
    Computes Brier Score, ECE (weighted), and generates a Reliability Diagram.
    
    Parameters:
    y_true: Ground truth labels (0 or 1)
    y_probs: Predicted probabilities (0.0 to 1.0)
    
    Returns:
    dict: Contains scalar scores and the matplotlib figure.
    """
    # 1. Prepare Tensors (TorchMetrics requires CPU tensors)
    # Ensure targets are integers and probs are floats
    y_true = y_true.detach().cpu().to(torch.int)
    y_probs = y_probs.detach().cpu().to(torch.float)

    # 2. Initialize Metrics
    # task='binary' handles 0/1 classification. 
    # norm='l1' ensures we calculate standard ECE (not RMSCE).
    ece_metric = BinaryCalibrationError(n_bins=10, norm='l1')
    
    ace = ACE(bins=15)
    score = ace.measure(y_probs.numpy(), y_true.numpy())
    print(f"ACE Score: {score}")
    # 3. Update Metrics
    ece_metric.update(y_probs, y_true)
   
    # 4. Compute Scores
    ece_score = ece_metric.compute().item()
    brier_score = brier_score_loss(y_true, y_probs)

    prob_true, prob_pred = calibration_curve(y_true.numpy(), y_probs.numpy(), n_bins=10, strategy='quantile')

    return {
        'brier_score': brier_score,
        'ece': ece_score,
        'ace': score,
        'prob_true': prob_true,
        'prob_pred': prob_pred

    }

