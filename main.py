import argparse
import logging
import sys
from pathlib import Path
from utils.config_loader import ConfigLoader
from model.encoder.model import RGCN
from model.decoder.distmult import DistMult
from model.decoder.transe import TransE
from model.trainer.pipeline import Pipeline
import os.path as osp
from torch_geometric.datasets.word_net import WordNet18RR
from torch_geometric.datasets import RelLinkPredDataset
import torch
from torch_geometric.nn import GAE

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """
    Main function that handles command line arguments and loads configuration.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Link Prediction with configurable datasets and models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level'
    )
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # try:
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Initialize config loader with the provided path
    logger.info(f"Loading configuration from: {config_path}")
    config_loader = ConfigLoader(str(config_path))
    dataset_config = config_loader.get_section('dataset') 
    model_config = config_loader.get_section('model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_config['name'] == "WN18RR":
        path = osp.join('.', 'data', 'RLPD')
        dataset = WordNet18RR(path)
        
        data = dataset[0]
        data['num_relations'] = data['edge_type'].max().item() + 1
        data['train_edge_index'] = data.edge_index[:,data.train_mask]
        data['train_edge_type'] = data.edge_type[data.train_mask]
        data['valid_edge_index'] = data.edge_index[:,data.val_mask]
        data['valid_edge_type'] = data.edge_type[data.val_mask]   
        data['test_edge_index'] = data.edge_index[:,data.test_mask]
        data['test_edge_type'] = data.edge_type[data.test_mask]     
        # Add reverse edges
        edge_index = data.edge_index 
        rev_edge_index = torch.flip(edge_index,[0])
        data.edge_index = torch.concat([edge_index,rev_edge_index],dim=1)
        rev_edge_type = data.edge_type + data.num_relations
        data.edge_type = torch.concat([data.edge_type,rev_edge_type],dim=0)
        data.num_relations = len(data.edge_type.unique())


        data.to(device)

    elif dataset_config['name'] == "FB15k-237":
        path = osp.join('.', 'data', 'RLPD')
        dataset = RelLinkPredDataset(path, 'FB15k-237')
        data = dataset[0].to(device)
        data['num_relations'] = dataset.num_relations
    else:
        raise ValueError("Unsupported dataset specified")

    # Initialize decoder
    if config_loader.get_section('model')['decoder']['type'] == 'DistMult':
        decoder = DistMult
    elif config_loader.get_section('model')['decoder']['type'] == 'TransE':
        decoder = TransE
    else:
        raise ValueError("Unsupported decoder type specified")

    decoder = decoder(
        num_nodes=data.num_nodes,
        num_relations=data.num_relations // 2,
        hidden_channels=config_loader.get_section('model')['encoder']['embedding_dim'],
    )
    logger.info(f"Decoder initialized: {decoder}")
    logger.info(f"Decoder parameters count: {sum(p.numel() for p in decoder.parameters())}")

    encoder = RGCN(
        num_nodes=data.num_nodes,
        num_relations=data.num_relations,
        model_config=model_config
    )
    logger.info("Encoder initialized successfully.")
    logger.info(f"Encoder parameters count: {sum(p.numel() for p in encoder.parameters())}")

    logger.info("Total encoder parameters: {}".format(sum(p.numel() for p in encoder.parameters())))

    logger.info(f"Encoder architecture:\n{encoder}")


    model = GAE(encoder=encoder, decoder=decoder).to(device)
    
    # Initialize trainer
    pipeline = Pipeline(
        model=model,
        data=data,
        config=config_loader,
        logger=logger
    )

    # Start training
    logger.info("Starting training process...")
    training_results = pipeline.start_pipeline()
    logger.info("Training process completed.")
    
    logger.info(f"Training results: {training_results}")


if __name__ == "__main__":
    main()
