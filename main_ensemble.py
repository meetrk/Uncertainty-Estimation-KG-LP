import argparse
import logging
import sys
from pathlib import Path
from utils.config_loader import ConfigLoader
from model.encoder.model import RGCN
from model.decoder.distmult import DistMult
from model.decoder.transe import TransE
from model.ensemble.deep_ensemble import DeepEnsemble
from model.trainer.ensemble_pipeline import EnsemblePipeline
import os.path as osp
from misc.rel_link_pred_dataset import RelLinkPredDataset
import torch


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ensemble_training.log')
        ]
    )


def main():
    """
    Main function for Deep Ensemble training and uncertainty estimation.
    """
    parser = argparse.ArgumentParser(
        description='Deep Ensemble for Knowledge Graph Link Prediction with Uncertainty Estimation',
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
    
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load for evaluation'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config_loader = ConfigLoader(str(config_path))
    dataset_config = config_loader.get_section('dataset')
    model_config = config_loader.get_section('model')
    ensemble_config = config_loader.get_section('ensemble')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
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
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of relations: {dataset.num_relations}")
    
    # Select decoder class
    if model_config['decoder']['type'] == 'DistMult':
        decoder_class = DistMult
    elif model_config['decoder']['type'] == 'TransE':
        decoder_class = TransE
    else:
        raise ValueError("Unsupported decoder type specified")
    
    # Prepare encoder and decoder arguments
    encoder_args = {
        'num_nodes': data.num_nodes,
        'num_relations': dataset.num_relations,
        'model_config': model_config
    }
    
    decoder_args = {
        'num_nodes': data.num_nodes,
        'num_relations': dataset.num_relations // 2,
        'hidden_channels': model_config['encoder']['embedding_dim']
    }
    
    # Create Deep Ensemble
    num_models = ensemble_config.get('num_models', 5)
    logger.info(f"Creating Deep Ensemble with {num_models} models...")
    
    ensemble = DeepEnsemble(
        base_encoder_class=RGCN,
        base_decoder_class=decoder_class,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
        num_models=num_models,
        device=device
    )
    
    total_params = sum(p.numel() for model in ensemble.models for p in model.parameters())
    logger.info(f"Total ensemble parameters: {total_params:,}")
    logger.info(f"Average parameters per model: {total_params // num_models:,}")
    
    # Initialize pipeline
    pipeline = EnsemblePipeline(
        ensemble_model=ensemble,
        data=data,
        config=config_loader,
        logger=logger
    )
    
    # Check if we should load checkpoint for evaluation
    train_config = config_loader.get_section('training')
    if args.load_checkpoint or train_config.get('test_uncertainty', False):
        checkpoint_path = args.load_checkpoint or train_config.get('checkpoint_path')
        if checkpoint_path:
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            pipeline.load_checkpoint(checkpoint_path)
            
            logger.info("Evaluating ensemble uncertainty on test set...")
            uncertainty_scores = pipeline.test_uncertainty()

        else:
            logger.error("No checkpoint path provided for evaluation")
            sys.exit(1)
    else:
        # Start training
        logger.info("Starting ensemble training...")
        training_results = pipeline.start_pipeline()
        logger.info("Ensemble training completed.")
        
        # Final test evaluation
        logger.info("Running final uncertainty evaluation...")
        uncertainty_scores = pipeline.test_uncertainty()
        
        logger.info("\n" + "="*50)
        logger.info("FINAL ENSEMBLE RESULTS")
        logger.info("="*50)
        for metric, value in uncertainty_scores.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("="*50 + "\n")


if __name__ == "__main__":
    main()
