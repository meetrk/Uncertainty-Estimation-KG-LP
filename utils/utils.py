from utils.dataset_loader import load_dataset, generate_data
import torch

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

def negative_sampling(batch, num_nodes, head_corrupt_prob, device='cpu'):
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