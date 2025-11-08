from utils.dataset_loader import load_dataset, generate_data
import torch
from random import sample
import numpy as np


def setup_and_load_dataset(dataset_config, logger):
    """
    Load dataset and generate PyTorch Geometric graph data.
    
    Args:
        dataset_config: Configuration for dataset loading
        logger: Logger instance
        
    Returns:
        PyTorch Geometric Data object
    """
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_config)
    
    logger.info("Generating graph data...")
    data = generate_data(*dataset)

    
    logger.info(f"Graph data created with {data.num_nodes} nodes")
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        logger.info(f"Number of edges: {data.edge_index.shape[1]}")
    if hasattr(data, 'edge_type') and data.edge_type is not None:
        logger.info(f"Number of relation types: {data.edge_type.max().item() + 1}")
    
    logger.info("Dataset loading completed successfully!")
    return data


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

def negative_sampling(batch, num_nodes, head_corrupt_prob, device='mps'):
    """ Samples negative examples in a batch of triples. Randomly corrupts either heads or tails."""
    bs, ns, _ = batch.size()

    # new entities to insert
    corruptions = torch.randint(size=(bs * ns,),low=0, high=num_nodes, dtype=torch.long, device=device)

    # boolean mask for entries to corrupt
    mask = torch.bernoulli(torch.empty(
        size=(bs, ns, 1), dtype=torch.float, device=device).fill_(head_corrupt_prob)).to(torch.bool)
    zeros = torch.zeros(size=(bs, ns, 1), dtype=torch.bool, device=device)
    mask = torch.cat([mask, zeros, ~mask], dim=2)

    batch[mask] = corruptions

    return batch.view(bs * ns, -1)


def edge_neighborhood(train_triples, sample_size=30000, num_nodes=None):
    """ Edge neighborhood sampling """

    if num_nodes is None:
        num_nodes = train_triples.max().item() + 1
    
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

    edges = [train_triples[e] for e in edges]
    return edges



def generate_batch_triples(triples, num_nodes, config, device, mode, sampling="sample",):

    """ Generate batch for training """
    if mode == "train":
        sample_size = config['sampling']['batch_size'] 
    elif mode == "eval":
        sample_size = triples.size(0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if sampling == "edge-neighborhood":
        batch = edge_neighborhood(triples, sample_size=sample_size, num_nodes=num_nodes)
        # Stack list of tensors into a single tensor
        batch = torch.stack(batch).to(device)
    elif sampling == "sample":
        indices = sample(range(triples.size(0)), k=sample_size)
        batch = triples[indices].to(device)
    elif sampling == "full":
        batch = triples.to(device)
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")

    # Ensure batch has shape [batch_size, 3]
    if batch.dim() != 2 or batch.size(1) != 3:
        raise ValueError(f"Expected batch shape [batch_size, 3], got {batch.shape}")
    
    return batch

