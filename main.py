import argparse
import logging
import sys
from pathlib import Path
from utils.config_loader import ConfigLoader
from utils.utils import setup_and_load_dataset
from model.encoder.model import RGCN
from model.decoder.distmult import DistMultDecoder
import torch

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
    
    try:
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Initialize config loader with the provided path
        logger.info(f"Loading configuration from: {config_path}")
        config_loader = ConfigLoader(str(config_path))
        dataset_config = config_loader.get_section('dataset')  

        # Load dataset and generate a PyG data object
        data = setup_and_load_dataset(dataset_config, logger)
        
        # Determine number of nodes
        if hasattr(data, 'num_nodes') and data.num_nodes is not None:
            num_nodes = data.num_nodes
        elif hasattr(data, 'edge_index') and data.edge_index is not None:
            num_nodes = data.edge_index.max().item() + 1
        else:
            raise ValueError("Cannot determine number of nodes from data")
        
        # model = LinkPredictor(config_loader.get_section('model'), data, logger)

        decoder = DistMultDecoder(
            num_relations=data.num_relations,
            embedding_dim=config_loader.get_section('model')['encoder']['embedding_dim'],
            num_nodes=num_nodes,
            num_rel=data.num_relations,
            w_init=config_loader.get_section('model')['decoder']['w_init'],
            w_gain=config_loader.get_section('model')['decoder']['w_gain'],
            b_init=config_loader.get_section('model')['decoder']['b_init'],
        )

        # Create entity indices (all entities in the graph)
        entity_indices = torch.arange(num_nodes, dtype=torch.long)
    
        # Initialize the RGCN model
        model = RGCN(
            num_entities=num_nodes,
            num_relations=data.num_relations,
            embedding_dim=config_loader.get_section('model')['encoder']['embedding_dim'],
            num_bases=config_loader.get_section('model')['encoder']['num_bases'],
            dropout=config_loader.get_section('model')['encoder']['dropout'],
            hidden_layer_size=config_loader.get_section('model')['encoder']['hidden_layer_size'],
            decoder=decoder
        )
        
        logger.info(f"Model architecture:\n{model}")

        # Forward pass through the encoder
        logger.info("Running forward pass...")
        entity_embeddings = model.forward(entity_indices, data.edge_index, data.edge_type)
        logger.info(f"Entity embeddings shape: {entity_embeddings.shape}")
        
        # Example: Score some triplets using the decoder
        # Take first 10 training triplets as example
        if hasattr(data, 'train_triplets') and len(data.train_triplets) > 0:
            sample_triplets = data.train_triplets[:10]
            scores = model.decoder(entity_embeddings, sample_triplets)
            logger.info(f"Sample triplet scores: {scores}")
        
        logger.info("Forward pass completed successfully!")

        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
