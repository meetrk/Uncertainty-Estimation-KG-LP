"""
Utility script to compare uncertainty estimation methods:
- MC Dropout
- Deep Ensemble
"""
import torch
import argparse
import logging
from pathlib import Path
from utils.config_loader import ConfigLoader
from model.trainer.pipeline import Pipeline
from model.trainer.ensemble_pipeline import EnsemblePipeline
from model.encoder.model import RGCN
from model.decoder.distmult import DistMult
from model.decoder.transe import TransE
from model.ensemble.deep_ensemble import DeepEnsemble
from misc.rel_link_pred_dataset import RelLinkPredDataset
from torch_geometric.nn import GAE
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_mc_dropout_model(config_path, checkpoint_path, device):
    """Load MC Dropout model."""
    logger = logging.getLogger(__name__)
    config_loader = ConfigLoader(config_path)
    dataset_config = config_loader.get_section('dataset')
    model_config = config_loader.get_section('model')
    
    # Load dataset
    if dataset_config['name'] == "WN18RR":
        path = osp.join('.', 'data', 'RLPD')
        dataset = RelLinkPredDataset(path, 'WN18RR')
    elif dataset_config['name'] == "FB15k-237":
        path = osp.join('.', 'data', 'RLPD')
        dataset = RelLinkPredDataset(path, 'FB15k-237')
    else:
        raise ValueError("Unsupported dataset")
    
    data = dataset[0].to(device)
    data['num_relations'] = dataset.num_relations
    
    # Initialize model
    if model_config['decoder']['type'] == 'DistMult':
        decoder = DistMult
    elif model_config['decoder']['type'] == 'TransE':
        decoder = TransE
    else:
        raise ValueError("Unsupported decoder type")
    
    decoder = decoder(
        num_nodes=data.num_nodes,
        num_relations=dataset.num_relations // 2,
        hidden_channels=model_config['encoder']['embedding_dim'],
    )
    
    encoder = RGCN(
        num_nodes=data.num_nodes,
        num_relations=dataset.num_relations,
        model_config=model_config
    )
    
    model = GAE(encoder=encoder, decoder=decoder).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, data, config_loader


def load_ensemble_model(config_path, checkpoint_path, device):
    """Load Deep Ensemble model."""
    logger = logging.getLogger(__name__)
    config_loader = ConfigLoader(config_path)
    dataset_config = config_loader.get_section('dataset')
    model_config = config_loader.get_section('model')
    ensemble_config = config_loader.get_section('ensemble')
    
    # Load dataset
    if dataset_config['name'] == "WN18RR":
        path = osp.join('.', 'data', 'RLPD')
        dataset = RelLinkPredDataset(path, 'WN18RR')
    elif dataset_config['name'] == "FB15k-237":
        path = osp.join('.', 'data', 'RLPD')
        dataset = RelLinkPredDataset(path, 'FB15k-237')
    else:
        raise ValueError("Unsupported dataset")
    
    data = dataset[0].to(device)
    data['num_relations'] = dataset.num_relations
    
    # Select decoder class
    if model_config['decoder']['type'] == 'DistMult':
        decoder_class = DistMult
    elif model_config['decoder']['type'] == 'TransE':
        decoder_class = TransE
    else:
        raise ValueError("Unsupported decoder type")
    
    # Prepare arguments
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
    
    # Create ensemble
    num_models = ensemble_config.get('num_models', 5)
    ensemble = DeepEnsemble(
        base_encoder_class=RGCN,
        base_decoder_class=decoder_class,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
        num_models=num_models,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for i, state_dict in enumerate(checkpoint['models']):
        ensemble.models[i].load_state_dict(state_dict)
    
    return ensemble, data, config_loader


def compare_uncertainty_methods(
    mc_config_path,
    mc_checkpoint_path,
    ensemble_config_path,
    ensemble_checkpoint_path,
    mc_samples=10
):
    """Compare MC Dropout and Deep Ensemble uncertainty estimates."""
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*60)
    logger.info("COMPARING UNCERTAINTY ESTIMATION METHODS")
    logger.info("="*60)
    
    # Load MC Dropout model
    logger.info("\n1. Loading MC Dropout model...")
    mc_model, mc_data, mc_config = load_mc_dropout_model(
        mc_config_path, mc_checkpoint_path, device
    )
    mc_pipeline = Pipeline(mc_model, mc_data, mc_config, logger)
    mc_pipeline.load_checkpoint(mc_checkpoint_path)
    
    # Load Ensemble model
    logger.info("\n2. Loading Deep Ensemble model...")
    ensemble, ensemble_data, ensemble_config = load_ensemble_model(
        ensemble_config_path, ensemble_checkpoint_path, device
    )
    ensemble_pipeline = EnsemblePipeline(ensemble, ensemble_data, ensemble_config, logger)
    ensemble_pipeline.load_checkpoint(ensemble_checkpoint_path)
    
    # Evaluate MC Dropout
    logger.info(f"\n3. Evaluating MC Dropout (samples={mc_samples})...")
    mc_scores = mc_pipeline.test_uncertainty(mc_samples=mc_samples)
    
    # Evaluate Deep Ensemble
    logger.info("\n4. Evaluating Deep Ensemble...")
    ensemble_scores = ensemble_pipeline.test_uncertainty()
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)
    
    comparison = {
        'MC Dropout': mc_scores,
        'Deep Ensemble': ensemble_scores
    }
    
    # Print comparison table
    metrics = ['brier_score', 'ece']
    
    logger.info("\n{:<20} {:<20} {:<20}".format("Metric", "MC Dropout", "Deep Ensemble"))
    logger.info("-" * 60)
    
    for metric in metrics:
        if metric in mc_scores and metric in ensemble_scores:
            logger.info("{:<20} {:<20.4f} {:<20.4f}".format(
                metric,
                mc_scores[metric],
                ensemble_scores[metric]
            ))
    
    # Additional ensemble-specific metrics
    if 'mean_epistemic_uncertainty' in ensemble_scores:
        logger.info("\nEnsemble-specific metrics:")
        logger.info(f"  Mean Epistemic Uncertainty: {ensemble_scores['mean_epistemic_uncertainty']:.4f}")
        logger.info(f"  Std Epistemic Uncertainty: {ensemble_scores['std_epistemic_uncertainty']:.4f}")
    
    logger.info("\n" + "="*60)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare uncertainty estimation methods')
    parser.add_argument('--mc-config', required=True, help='MC Dropout config path')
    parser.add_argument('--mc-checkpoint', required=True, help='MC Dropout checkpoint path')
    parser.add_argument('--ensemble-config', required=True, help='Ensemble config path')
    parser.add_argument('--ensemble-checkpoint', required=True, help='Ensemble checkpoint path')
    parser.add_argument('--mc-samples', type=int, default=10, help='Number of MC samples')
    
    args = parser.parse_args()
    
    results = compare_uncertainty_methods(
        args.mc_config,
        args.mc_checkpoint,
        args.ensemble_config,
        args.ensemble_checkpoint,
        args.mc_samples
    )


if __name__ == "__main__":
    main()
