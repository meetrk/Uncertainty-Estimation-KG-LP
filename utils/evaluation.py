"""
Evaluation metrics and utilities for knowledge graph link prediction.
"""
import torch
import numpy as np
from typing import  List
from torch import softmax

from tqdm import tqdm
from torchmetrics.classification.calibration_error import BinaryCalibrationError
from sklearn.metrics import brier_score_loss
from model.ensemble.deep_ensemble import DeepEnsemble
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
    y_probab = []
    y_true = [] 
    variance_true = []
    variance_false = []
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

        ranks.append(compute_rank(probs))
        
        y_true.append(1)  
        y_probab.append(probs[0].item())
        variance_true.append(variance[0].item())
        k = 10
        indices = torch.randperm(len(probs))[:k]
        for idx in indices:
            if idx != 0:  
                variance_false.append(variance[idx].item())
                y_true.append(0)  
                y_probab.append(probs[idx].item())

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
        
        ranks.append(compute_rank(probs))
        y_true.append(1)  
        y_probab.append(probs[0].item())
        variance_true.append(variance[0].item())
        k = 10
        indices = torch.randperm(len(probs))[:k]
        for idx in indices:
            if idx != 0:  
                variance_false.append(variance[idx].item())
                y_true.append(0)  
                y_probab.append(probs[idx].item())

    y_probs_tensor = torch.tensor(y_probab)
    y_true_tensor = torch.tensor(y_true)
    variance_true_tensor = torch.tensor(variance_true)
    variance_false_tensor = torch.tensor(variance_false)

    # Call your provided function
    uncertainty_results = compute_uncertainty(y_true_tensor, y_probs_tensor)        

    # Calculate final metrics
    ranks_t = torch.tensor(ranks, dtype=torch.float)
    scores = {
        'mrr': (1. / ranks_t).mean().item(),
        'mean_rank': ranks_t.mean().item(),
        'hits@1': (ranks_t <= 1).float().mean().item(),
        'hits@3': (ranks_t <= 3).float().mean().item(),
        'hits@10': (ranks_t <= 10).float().mean().item(),
        'variance_true_mean': variance_true_tensor.mean().item(),
        'variance_false_mean': variance_false_tensor.mean().item(),
        'brier_score': uncertainty_results['brier_score'],
        'ece': uncertainty_results['ece'],
        'ace': uncertainty_results['ace'],
        'prob_true': uncertainty_results['prob_true'],
        'prob_pred': uncertainty_results['prob_pred'],        
    }

    return scores

@torch.no_grad()
def compute_mrr_mc_dropout(train_edge_index, train_edge_type, edge_index, edge_type, data, model, mc_samples, return_probs= False):

    # OPTIMIZATION: Encode graph once per model and cache
    all_z = []
    model.encoder.mc_dropout = True
    for _ in range(mc_samples):
        z = model.encode(train_edge_index, train_edge_type)
        all_z.append(z)
    model.encoder.mc_dropout = False

    ranks = []
    y_probab = []
    y_true = [] 
    variance_true = []
    variance_false = []
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
        
        probs = torch.stack(predictions).mean(dim=0)
        variance = torch.stack(predictions).var(dim=0)
        variance_true.append(variance[0].item())

        rank = compute_rank(probs)
        ranks.append(rank)
        y_true.append(1)  
        y_probab.append(probs[0].item())  
        k = 2
        indices = torch.randperm(len(probs))[:k]
        for idx in indices:
            if idx != 0:  
                variance_false.append(variance[idx].item())
                y_true.append(0)  
                y_probab.append(probs[idx].item())
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
        
        probs = torch.stack(predictions).mean(dim=0)
        variance = torch.stack(predictions).var(dim=0)
        variance_true.append(variance[0].item())
        rank = compute_rank(probs)
        ranks.append(rank)
        y_true.append(1)  
        y_probab.append(probs[0].item())  
        k = 2
        indices = torch.randperm(len(probs))[:k]
        for idx in indices:
            if idx != 0:  
                variance_false.append(variance[idx].item())
                y_true.append(0)  
                y_probab.append(probs[idx].item())


    if return_probs:
        return torch.tensor(y_probab), torch.tensor(y_true)

    if (hasattr(model.decoder, 'use_calibration') and 
        getattr(model.decoder, 'use_calibration', False) and 
        hasattr(model.decoder, 'isotonic_regression_transform') and 
        getattr(model.decoder, 'isotonic_regression_transform', None) is not None):
        # Apply isotonic regression calibration
        y_probs_np = np.array(y_probab)
        calibrated_probs = model.decoder.isotonic_regression_transform.transform(y_probs_np)
        print("After Calibration:")
        print_stats(calibrated_probs)
        y_probs_tensor = torch.tensor(calibrated_probs)
        
    # Call your provided function
    uncertainty_results = compute_uncertainty(y_true, y_probab)

    scores = {
        'mrr': (1. / torch.tensor(ranks, dtype=torch.float)).mean(),
        'mean_rank': torch.tensor(ranks, dtype=torch.float).mean(),
        'hits@1': (torch.tensor(ranks, dtype=torch.float) <= 1).float().mean(),
        'hits@3': (torch.tensor(ranks, dtype=torch.float) <= 3).float().mean(),
        'hits@10': (torch.tensor(ranks, dtype=torch.float) <= 10).float().mean(),
        'variance_true_mean': np.mean(variance_true),
        'variance_false_mean': np.mean(variance_false),
        # Add Uncertainty Metrics
        'brier_score': uncertainty_results['brier_score'],
        'ece': uncertainty_results['ece'],
        'ace': uncertainty_results['ace'],
        'prob_true': uncertainty_results['prob_true'],
        'prob_pred': uncertainty_results['prob_pred'],

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
    y_true_tensor = torch.tensor(y_true)
    y_probs_tensor = torch.tensor(y_probs)

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
def compute_mrr(edge_index, edge_type, data, model, return_probs= False):
    model.eval()
    z = model.encode(data.edge_index, data.edge_type)
    ranks = []
    
    # For calibration
    ranks = []
    y_probab = []
    y_true = []

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
        
        probs = torch.sigmoid(out)
        # probs = F.softmax(out, dim=0)
        
        ranks.append(compute_rank(probs))
        y_true.append(1)  
        y_probab.append(probs[0].item())  
        k = 2
        indices = torch.randperm(len(probs))[:k]
        for idx in indices:
            if idx != 0:  
                y_true.append(0)  
                y_probab.append(probs[idx].item())

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
        
        probs = torch.sigmoid(out)
        ranks.append(compute_rank(probs))
        y_true.append(1)  
        y_probab.append(probs[0].item())  

        k = 2
        indices = torch.randperm(len(probs))[:k]
        for idx in indices:
            if idx != 0:  
                y_true.append(0)  
                y_probab.append(probs[idx].item())

    print("Before Calibration:")
    print_stats(np.array(y_probab))
    uncertainty_results = compute_uncertainty(y_true, y_probab)
    if return_probs: 
        return y_probab, y_true

    ranks_tensor = torch.tensor(ranks, dtype=torch.float)
    
    scores = {
        'mrr': (1. / ranks_tensor).mean().item(),
        'mean_rank': ranks_tensor.mean().item(),
        'hits@1': (ranks_tensor <= 1).float().mean().item(),
        'hits@3': (ranks_tensor <= 3).float().mean().item(),
        'hits@10': (ranks_tensor <= 10).float().mean().item(),

        'brier_score': uncertainty_results['brier_score'],
        'ece': uncertainty_results['ece'],
        'ace': uncertainty_results['ace'],
        'prob_true': uncertainty_results['prob_true'],
        'prob_pred': uncertainty_results['prob_pred'],
    }

    return scores


def print_stats(probab,addtional=""):

    print(f"Average Confidence of Top-1 Predictions: {np.mean(probab)}")
    print(f"Max Confidence of Top-1 Predictions: {np.max(probab)}")
    print(f"Min Confidence of Top-1 Predictions: {np.min(probab)}")
    print(f"Std Dev of Confidence of Top-1 Predictions: {np.std(probab)}")


    # Plot histogram of confidence scores
    plt.figure(figsize=(10, 6))
    plt.hist(probab, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Top-1 Prediction Confidence {addtional}')
    plt.grid(True, alpha=0.3)
    filename = './plots/confidence_histogram{}.png'.format(np.random.randint(10000))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved as '{filename}'")
