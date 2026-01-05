import torch
from random import sample
import numpy as np

def get_triples(edge_index, edge_type):
    """
    Generate triplets from edge_index and edge_type.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        edge_type: Edge types [num_edges]

    """
    heads = edge_index[0]
    tails = edge_index[1]
    relations = edge_type

    triplets = torch.stack([heads, relations, tails], dim=1)
    return triplets


def get_edges(triplets):
    """
    Generate edge_index and edge_type from triplets.
    
    Args:
        triplets: Triplets [num_triplets, 3] where each row is [head, relation, tail]
        
    Returns:
        edge_index: Edge indices [2, num_triplets]
        edge_type: Edge types [num_triplets]
    """
    heads = triplets[:, 0]
    relations = triplets[:, 1]
    tails = triplets[:, 2]
    
    edge_index = torch.stack([heads, tails], dim=0)
    edge_type = relations
    
    return edge_index, edge_type

# def negative_sampling(batch, num_nodes, head_corrupt_prob, device='mps'):
#     """ Samples negative examples in a batch of triples. Randomly corrupts either heads or tails."""
#     bs, ns, _ = batch.size()

#     # new entities to insert
#     corruptions = torch.randint(size=(bs * ns,),low=0, high=num_nodes, dtype=torch.long, device=device)

#     # boolean mask for entries to corrupt
#     mask = torch.bernoulli(torch.empty(
#         size=(bs, ns, 1), dtype=torch.float, device=device).fill_(head_corrupt_prob)).to(torch.bool)
#     zeros = torch.zeros(size=(bs, ns, 1), dtype=torch.bool, device=device)
#     mask = torch.cat([mask, zeros, ~mask], dim=2)

#     batch[mask] = corruptions

#     return batch.view(bs * ns, -1)


def edge_neighborhood(edge_index,edge_type, sample_size=30000, num_nodes=None, return_indices=False):
    """ Edge neighborhood sampling 
    
    Args:
        train_triples: Training triples tensor of shape [num_triples, 3]
        sample_size: Number of edges to sample
        num_nodes: Total number of nodes (optional, inferred if None)
        return_indices: If True, return indices instead of actual triples
        
    Returns:
        If return_indices=True: List/array of indices into train_triples
        If return_indices=False: List of actual triple tensors (original behavior)
    """

    train_triples = get_triples(edge_index, edge_type)
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(train_triples):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]

    edges = np.zeros((sample_size), dtype=np.int32)

    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in train_triples])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    # Return indices or actual triples based on flag
    if return_indices:
        return edges  # Return numpy array of indices
    else:
        edges = [train_triples[e] for e in edges]  # Original behavior
        return edges



def generate_batch_triples(triples, num_nodes, config, mode, sampling="sample"):

    """ Generate batch for training """
    if mode == "train":
        sample_size = config['sampling']['batch_size'] 
    elif mode == "eval":
        sample_size = triples.size(0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if sampling == "edge-neighborhood":
        indices = edge_neighborhood(triples, sample_size=sample_size, num_nodes=num_nodes)
        # Stack list of tensors into a single tensor
        batch = triples[indices]
    elif sampling == "sample":
        indices = sample(range(triples.size(0)), k=sample_size)
        batch = triples[indices]
    elif sampling == "full":
        batch = triples
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")

    # Ensure batch has shape [batch_size, 3]
    if batch.dim() != 2 or batch.size(1) != 3:
        raise ValueError(f"Expected batch shape [batch_size, 3], got {batch.shape}")
    
    return batch


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