from utils.dataset_loader import load_dataset, generate_data

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