import argparse
import logging
import sys
from pathlib import Path
from utils.config_loader import ConfigLoader
from torch_geometric.datasets import WordNet18RR,FB15k_237
from model.encoder.model import RGCN
from model.decoder.distmult import DistMult
from model.decoder.transe import TransE
from model.trainer.pipeline import Pipeline

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
    training_config = config_loader.get_section('training')
    model_config = config_loader.get_section('model')


    if dataset_config['name'] == "WN18RR":
        dataset = WordNet18RR
    elif dataset_config['name'] == "FB15k-237":
        dataset = FB15k_237
    else:
        raise ValueError("Unsupported dataset specified")

    data = dataset(root=f"./dataset/{dataset_config['name']}").data
    data.num_relations = len(data.edge_type.unique())
    logger.info(f"Dataset '{dataset_config['name']}' loaded with {data.num_nodes} nodes and {data.num_relations} relations.")
    logger.info(f"Data object: {data}")
    
    

    # Initialize decoder
    if config_loader.get_section('model')['decoder']['type'] == 'DistMult':
        decoder = DistMult
    elif config_loader.get_section('model')['decoder']['type'] == 'TransE':
        decoder = TransE
    else:
        raise ValueError("Unsupported decoder type specified")

    decoder = decoder(
        num_nodes=data.num_nodes,
        num_relations=(data.num_relations * 2) + 1 ,
        hidden_channels=config_loader.get_section('model')['encoder']['embedding_dim'],
    )
    logger.info(f"Decoder initialized: {decoder}")
    logger.info(f"Decoder parameters count: {sum(p.numel() for p in decoder.parameters())}")

    model = RGCN(
        num_nodes=data.num_nodes,
        num_relations=data.num_relations,
        model_config=model_config,
        decoder=decoder
    )
    logger.info("Model initialized successfully.")
    logger.info(f"Model parameters count: {sum(p.numel() for p in model.parameters())}")

    logger.info("Total model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    logger.info(f"Model architecture:\n{model}")

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
    # pipeline.plot_training_history()

    # except FileNotFoundError as e:
    #     logger.error(f"File not found: {e}")
    #     sys.exit(1)
    # except KeyError as e:
    #     logger.error(f"Configuration error: {e}")
    #     sys.exit(1)
    # except Exception as e:
    #     logger.error(f"Unexpected error: {e}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()
