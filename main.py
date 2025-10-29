import argparse
import logging
import sys
from pathlib import Path
from utils.config_loader import ConfigLoader
from utils.dataset_loader import load_dataset, generate_data


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

        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset(dataset_config)
        
        # Generate PyTorch Geometric data
        logger.info("Generating graph data...")
        data = generate_data(*dataset)
        
        logger.info(f"Graph data created with {data.num_nodes} nodes")
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            logger.info(f"Number of edges: {data.edge_index.shape[1]}")
        if hasattr(data, 'edge_type') and data.edge_type is not None:
            logger.info(f"Number of relation types: {data.edge_type.max().item() + 1}")
        
        # TODO: Add model training and evaluation here
        logger.info("Dataset loading completed successfully!")
        
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
