"""
Training pipeline for Deep Ensemble uncertainty estimation.
"""
import torch
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils import negative_sampling, dropout_edges
from utils.evaluation import compute_uncertainty, compute_mrr_ensemble
from model.trainer.base_pipeline import BasePipeline
import numpy as np


class EnsemblePipeline(BasePipeline):
    """
    Training pipeline for Deep Ensemble models.
    
    Trains multiple models independently and provides uncertainty estimation
    through prediction variance across the ensemble.
    """
    
    def __init__(self, ensemble_model, data, config, logger):
        # Call parent constructor
        super().__init__(ensemble_model, data, config, logger)
        
        # Store reference to ensemble (model is already stored in parent)
        self.ensemble = self.model
        self.ensemble_config = self.config.get_section('ensemble')
        
        # Create separate optimizer for each ensemble member
        self.optimizers = self.ensemble.get_optimizers(
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Add ensemble-specific training history
        self.training_history['ensemble_diversity'] = []
    
    def _get_experiment_name(self):
        """Get experiment name for TensorBoard logs."""
        return f"ensemble_experiment_{self.config.get_section('dataset')}"
    
    def _get_hyperparameters(self):
        """Get hyperparameters including ensemble-specific ones."""
        hparams = super()._get_hyperparameters()
        hparams['num_ensemble_models'] = self.ensemble.num_models
        return hparams
    
    def start_pipeline(self):
        """Start the ensemble training pipeline."""
        max_epochs = self.train_config['epochs']
        eval_frequency = self.train_config.get('evaluation_frequency', 10)
        early_stopping = self.train_config['early stopping']['enabled']
        patience = self.train_config['early stopping'].get('patience', 10)
        delta = self.train_config['early stopping'].get('delta', 0.0)
        
        self.logger.info(f"Starting ensemble training with {self.ensemble.num_models} models for {max_epochs} epochs")
        
        best_val_mrr = -float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(1, max_epochs + 1), desc="Training Ensemble"):
            # Train all ensemble members
            ensemble_losses = self.train_ensemble()
            avg_loss = sum(ensemble_losses) / len(ensemble_losses)
            
            # Log individual model losses
            for i, loss in enumerate(ensemble_losses):
                self.writer.add_scalar(f'Loss/Model_{i}', loss, epoch)
            
            # Use parent class method for logging
            self.log_training_metrics(epoch, avg_loss)
            
            # Evaluation
            if epoch % eval_frequency == 0:
                valid_scores, test_scores = self.test_ensemble()
                
                # Compute ensemble diversity
                diversity = self.compute_diversity()
                self.training_history['ensemble_diversity'].append({
                    "epoch": epoch,
                    "diversity": diversity
                })
                self.writer.add_scalar('Ensemble/Diversity', diversity, epoch)
                
                current_val_mrr = valid_scores['mrr']
                
                # Log metrics using parent class method
                self.log_evaluation_metrics(epoch, valid_scores, test_scores)
                self.logger.info(f"Ensemble Diversity = {diversity:.4f}")
                
                # Check for improvement
                if current_val_mrr - best_val_mrr > delta:
                    best_val_mrr = current_val_mrr
                    patience_counter = 0
                    self.logger.info(f"New best ensemble! MRR: {best_val_mrr:.4f}")
                    self.save_checkpoint(epoch)
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                
                # Early stopping
                if early_stopping and patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break
        
        self.writer.close()
        self.logger.info("Ensemble training completed!")
        return self.training_history
    
    def train(self):
        """Train all ensemble members for one epoch. Wrapper for train_ensemble."""
        sum_loss = sum(self.train_ensemble())
        return sum_loss
    
    def train_ensemble(self):
        """Train all ensemble members for one epoch."""
        ensemble_losses = []
        
        for model_idx in range(self.ensemble.num_models):
            model = self.ensemble.get_model(model_idx)
            optimizer = self.optimizers[model_idx]
            
            model.train()
            optimizer.zero_grad()
            
            # Edge dropout if enabled
            if self.train_config['sampling']['edge_dropout'] > 0:
                edge_index, edge_type = dropout_edges(
                    self.data.edge_index,
                    self.data.edge_type,
                    self.train_config['sampling']['edge_dropout']
                )
            else:
                edge_index, edge_type = self.data.edge_index, self.data.edge_type
            
            # Forward pass
            z = model.encode(edge_index, edge_type)
            pos_out = model.decode(z, self.data.train_edge_index, self.data.train_edge_type)
            
            # Negative sampling
            neg_edge_index, neg_edge_type = negative_sampling(self.data.train_edge_index,self.data.train_edge_type, self.data.num_nodes,1)
            neg_out = model.decode(z, neg_edge_index, neg_edge_type)
            
            # Compute loss
            out = torch.cat([pos_out, neg_out])
            gt = torch.cat([torch.full_like(pos_out, self.train_config['label_smoothing']['positive']), torch.full_like(neg_out, self.train_config['label_smoothing']['negative'])])
            
            cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
            reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
            loss = cross_entropy_loss + self.model_config['decoder']['l2_penalty'] * reg_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            
            ensemble_losses.append(float(loss))
        
        return ensemble_losses
    
    def test(self):
        """Test ensemble. Wrapper for test_ensemble."""
        return self.test_ensemble()
    
    @torch.no_grad()
    def test_ensemble(self):
        """Test ensemble with uncertainty quantification."""

        self.ensemble.eval()
        
        scores = compute_mrr_ensemble(
            self.data.edge_index,
            self.data.edge_type,
            self.data.valid_edge_index,
            self.data.valid_edge_type,
            self.data,
            self.ensemble
        )
        valid_scores = scores
         
        
        return valid_scores, None
    
    @torch.no_grad()
    def test_uncertainty(self):
        """
        Evaluate uncertainty on test set using ensemble variance.
        
        Returns uncertainty metrics: Brier score, ECE, etc.
        """
        self.ensemble.eval()
        
        # Get predictions with uncertainty
        mean_pred, std_pred = self.ensemble.predict_with_uncertainty(
            self.data.edge_index,
            self.data.edge_type,
            self.data.valid_edge_index,
            self.data.valid_edge_type
        )
        
        # Also get negative samples for evaluation
        neg_edge_index, neg_edge_type = negative_sampling(self.data.valid_edge_index, self.data.valid_edge_type ,self.data.num_nodes, 1)
        neg_mean_pred, neg_std_pred = self.ensemble.predict_with_uncertainty(
            self.data.edge_index,
            self.data.edge_type,
            neg_edge_index,
            neg_edge_type,
        )
        
        # Combine positive and negative predictions
        all_mean_pred = torch.cat([mean_pred, neg_mean_pred])
        all_std_pred = torch.cat([std_pred, neg_std_pred])
        labels = torch.cat([
            torch.ones_like(mean_pred),
            torch.zeros_like(neg_mean_pred)
        ])
        
        # Apply sigmoid to get probabilities
        all_mean_pred = torch.sigmoid(all_mean_pred)
        
        # Compute uncertainty metrics
        scores = compute_uncertainty(labels, all_mean_pred)
        

        # Add epistemic uncertainty stats
        scores['mean_epistemic_uncertainty'] = all_std_pred.mean().item()
        scores['std_epistemic_uncertainty'] = all_std_pred.std().item()

        
        self.logger.info(f"Brier Score: {scores['brier_score']:.4f}")
        self.logger.info(f"ECE: {scores['ece']:.4f}")
        self.logger.info(f"Mean Epistemic Uncertainty: {scores['mean_epistemic_uncertainty']:.4f}")
        self.logger.info(f"Probability True: {np.asarray(scores['prob_true'])}")
        self.logger.info(f"Probability Predicted: {np.asarray(scores['prob_pred'])}")
        
        return scores

    def load_pipeline(self, checkpoint_path):
        """Load ensemble pipeline from checkpoint."""
        self.load_checkpoint(checkpoint_path)
        self.logger.info("Checkpoint loaded. Evaluating ensemble uncertainty on test set...")

        scores,_ = self.test_ensemble()

        self.logger.info(f"MRR = {scores['mrr']:.4f}")
        self.logger.info(f"Mean Rank = {scores['mean_rank']:.4f}")
        self.logger.info(f"Hits@1 = {scores['hits@1']:.4f}")
        self.logger.info(f"Hits@3 = {scores['hits@3']:.4f} ")
        self.logger.info(f"Hits@10 = {scores['hits@10']:.4f}")

        return scores

    @torch.no_grad()
    def compute_diversity(self):
        """
        Compute ensemble diversity metric.
        
        Measures how different the predictions are across ensemble members.
        Higher diversity generally leads to better uncertainty estimates.
        """
        # Sample a batch of validation edges
        sample_size = min(1000, self.data.valid_edge_index.size(1))
        indices = torch.randperm(self.data.valid_edge_index.size(1))[:sample_size]
        
        sample_edge_index = self.data.valid_edge_index[:, indices]
        sample_edge_type = self.data.valid_edge_type[indices]
        
        # Get predictions from all models
        predictions = self.ensemble.forward_ensemble(
            self.data.edge_index,
            self.data.edge_type,
            sample_edge_index,
            sample_edge_type
        )
        
        preds_stack = torch.stack(predictions, dim=0)  # (num_models, num_samples)
        
        # Compute pairwise disagreement
        diversity = preds_stack.var(dim=0).mean().item()
        
        return diversity
    
    def _build_checkpoint(self, epoch):
        """Build ensemble checkpoint dictionary."""
        return {
            'epoch': epoch,
            'num_models': self.ensemble.num_models,
            'encoder_args': self.ensemble.encoder_args,
            'decoder_args': self.ensemble.decoder_args,
            'models': [model.state_dict() for model in self.ensemble.models],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'training_history': self.training_history,
            'config': self.config
        }
    
    def _restore_from_checkpoint(self, checkpoint):
        """Restore ensemble state from checkpoint."""
        for i, state_dict in enumerate(checkpoint['models']):
            self.ensemble.models[i].load_state_dict(state_dict)
        
        for i, state_dict in enumerate(checkpoint['optimizers']):
            self.optimizers[i].load_state_dict(state_dict)
        
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'eval_metrics': [], 'ensemble_diversity': []
        })

