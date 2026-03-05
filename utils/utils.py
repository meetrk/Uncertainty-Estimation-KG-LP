import torch
from random import sample
import numpy as np
import torch.nn.functional as F


def negative_sampling(edge_index, edge_type, num_nodes, n_neg=1):
    """
    Samples random negative edges for Knowledge Graph Link Prediction.
    
    Args:
        edge_index (LongTensor): [2, num_edges] Positive edges.
        edge_type (LongTensor): [num_edges] Relation types.
        num_nodes (int): Total number of nodes in the graph.
        n_neg (int): Number of negative samples to generate per positive edge.
    
    Returns:
        neg_edge_index: [2, num_edges * n_neg]
        neg_edge_type: [num_edges * n_neg]
    """
    device = edge_index.device
    
    # 1. Repeat the positive edges and types 'n_neg' times
    #    If we have 100 edges and n_neg=5, we get 500 edges.
    #    We repeat to ensure every positive edge gets 'n_neg' corresponding negatives.
    neg_edge_index = edge_index.repeat(1, n_neg)
    neg_edge_type = edge_type.repeat(n_neg)
    
    num_samples = neg_edge_index.size(1)
    
    # 2. Generate random node indices
    #    We generate enough random integers to potentially replace nodes in all samples.
    random_nodes = torch.randint(num_nodes, (num_samples,), device=device)
    
    # 3. Create a corruption mask (Bernoulli distribution with p=0.5)
    #    True  = Corrupt the Head (Subject)
    #    False = Corrupt the Tail (Object)
    mask_corrupt_head = torch.rand(num_samples, device=device) < 0.5
    
    # 4. Apply corruption using vectorized masking
    #    Replace Heads where mask is True
    neg_edge_index[0, mask_corrupt_head] = random_nodes[mask_corrupt_head]
    
    #    Replace Tails where mask is False
    neg_edge_index[1, ~mask_corrupt_head] = random_nodes[~mask_corrupt_head]
    
    return neg_edge_index, neg_edge_type

def dropout_edges(edge_index, edge_type, dropout_ratio):
    
    num_edges = edge_index.size(1)

    num_real_edges = num_edges // 2
    
    mask_real = torch.rand(num_real_edges, device=edge_index.device) >= dropout_ratio
  
    mask = torch.cat([mask_real, mask_real], dim=0)
    
    dropped_edge_index = edge_index[:, mask]
    dropped_edge_type = edge_type[mask]
    
    return dropped_edge_index, dropped_edge_type