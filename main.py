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
from misc.rel_link_pred_dataset import RelLinkPredDataset
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
        dataset = RelLinkPredDataset(path, 'WN18RR')
        data = dataset[0].to(device)
        data['num_relations'] = dataset.num_relations

    elif dataset_config['name'] == "FB15k-237":
        path = osp.join('.', 'data', 'RLPD')
        dataset = RelLinkPredDataset(path, 'FB15k-237')
        data = dataset[0].to(device)
        data['num_relations'] = dataset.num_relations
    else:
        raise ValueError("Unsupported dataset specified")
    logger.info(f"Dataset '{dataset_config['name']}' loaded successfully.")
    logger.info(f"Number of nodes: {data}")
    # Initialize decoder
    if config_loader.get_section('model')['decoder']['type'] == 'DistMult':
        decoder = DistMult
    elif config_loader.get_section('model')['decoder']['type'] == 'TransE':
        decoder = TransE
    else:
        raise ValueError("Unsupported decoder type specified")

    decoder = decoder(
        num_nodes=data.num_nodes,
        num_relations=dataset.num_relations // 2,
        hidden_channels=config_loader.get_section('model')['encoder']['embedding_dim'],
    )
    logger.info(f"Decoder initialized: {decoder}")
    logger.info(f"Decoder parameters count: {sum(p.numel() for p in decoder.parameters())}")

    encoder = RGCN(
        num_nodes=data.num_nodes,
        num_relations=dataset.num_relations,
        model_config=model_config
    )
    logger.info("Encoder initialized successfully.")
    logger.info(f"Encoder parameters count: {sum(p.numel() for p in encoder.parameters())}")

    logger.info("Total encoder parameters: {}".format(sum(p.numel() for p in encoder.parameters())))

    logger.info(f"Encoder architecture:\n{encoder}")

    logger.info(f"Decoder architecture:\n{decoder}")

    model = GAE(encoder=encoder, decoder=decoder).to(device)
    logger.info(f"GAE architecture:\n{model}")
    # Initialize trainer
    pipeline = Pipeline(
        model=model,
        data=data,
        config=config_loader,
        logger=logger
    )
    train_config = config_loader.get_section('training')
    if train_config['test_uncertainty']:
        logger.info("Starting uncertainty evaluation on test set...")
        test_scores = pipeline.load_pipeline(train_config['checkpoint_path'])
        logger.info(f"Uncertainty Evaluation - Brier Score: {test_scores['brier_score']:.4f}")
        logger.info(f"Uncertainty Evaluation - Reliability Curve: {test_scores['reliability_curve']:.4f}")
        return
    # Start training
    logger.info("Starting training process...")
    training_results = pipeline.start_pipeline()
    logger.info("Training process completed.")
    
    logger.info(f"Training results: {training_results}")


if __name__ == "__main__":
    main()
