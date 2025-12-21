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
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt



def mean_reciprocal_rank(ranks: List[int]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    return float(np.mean([1.0 / rank for rank in ranks]))


def hits_at_k(ranks: List[int], k: int) -> float:
    """Calculate Hits@K metric."""
    return float(np.mean([1.0 if rank <= k else 0.0 for rank in ranks]))

def filter_scores(scores, batch, true_triples, head=True):
    """Filters a score matrix by setting the scores of known non-target true triples to -infinity"""
    
    device = scores.device
    indices = []  # indices of triples whose scores should be set to -infinity

    # true_triples should be a set of tuples (h, r, t) for faster lookup
    if isinstance(true_triples, tuple):
        heads, tails = true_triples
        # Convert to set format for compatibility
        true_triples_set = set()
        for p in heads:
            for o in heads[p]:
                for h in heads[p][o]:
                    true_triples_set.add((h, p, o))
    else:
        # Assume true_triples is already a set or list of (h, r, t) tuples
        true_triples_set = set(tuple(triple) for triple in true_triples)

    for i, (s, p, o) in enumerate(batch):
        s, p, o = (s.item(), p.item(), o.item())
        if head:
            # Filter head predictions: check all (?, p, o) combinations
            for candidate_h in range(scores.size(1)):
                if candidate_h != s and (candidate_h, p, o) in true_triples_set:
                    indices.append((i, candidate_h))
        else:
            # Filter tail predictions: check all (s, p, ?) combinations
            for candidate_t in range(scores.size(1)):
                if candidate_t != o and (s, p, candidate_t) in true_triples_set:
                    indices.append((i, candidate_t))

    if indices:
        indices = torch.tensor(indices, device=device)
        scores[indices[:, 0], indices[:, 1]] = float('-inf')

@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr(z, edge_index, edge_type, data, model):
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
def compute_mrr_emsemble(train_edge_index, train_edge_type, edge_index, edge_type, data, models: DeepEnsemble):
    """
    Compute MRR for ensemble with optimized encoding caching.
    
    Key optimization: Encode the graph once per model, reuse for all triples.
    This reduces complexity from O(N_models × N_triples × encoding_cost) 
    to O(N_models × encoding_cost + N_models × N_triples × decoding_cost).
    """
    # OPTIMIZATION: Encode graph once per model and cache
    all_z = []
    for model_idx in range(models.num_models):
        model = models.get_model(model_idx)
        z = model.encode(train_edge_index, train_edge_type)
        all_z.append(z)
    
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
        for model_idx in range(models.num_models):
            model = models.get_model(model_idx)
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
        for model_idx in range(models.num_models):
            model = models.get_model(model_idx)
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


@torch.no_grad()
def compute_mrr_mc_dropout(train_edge_index, train_edge_type, edge_index, edge_type, data, model, mc_samples=10):
    """
    Compute MRR for ensemble with optimized encoding caching.
    
    Key optimization: Encode the graph once per model, reuse for all triples.
    This reduces complexity from O(N_models × N_triples × encoding_cost) 
    to O(N_models × encoding_cost + N_models × N_triples × decoding_cost).
    """
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
            pred = torch.sigmoid(pred)
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
            pred = torch.sigmoid(pred)
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
    
    # 3. Update Metrics
    ece_metric.update(y_probs, y_true)
   
    # 4. Compute Scores
    ece_score = ece_metric.compute().item()
    brier_score = brier_score_loss(y_true, y_probs)

    prob_true, prob_pred = calibration_curve(y_true.numpy(), y_probs.numpy(), n_bins=10, strategy='quantile')

    return {
        'brier_score': brier_score,
        'ece': ece_score,
        'prob_true': prob_true,
        'prob_pred': prob_pred

    }