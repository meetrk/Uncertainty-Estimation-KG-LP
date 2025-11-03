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

