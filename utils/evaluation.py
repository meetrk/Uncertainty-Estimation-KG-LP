"""
Evaluation metrics and utilities for knowledge graph link prediction.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm, trange



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

def compute_uncertainty(y_true, y_probs):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    # 1. Brier Score
    score = brier_score_loss(y_true, y_probs)

    # 2. Reliability Diagram Data
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

    return {
        'brier_score': score,
        'reliability_curve': (prob_true, prob_pred)
    }