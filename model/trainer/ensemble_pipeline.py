import torch
from pathlib import Path
from model.trainer.basepipeline import BasePipeline
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils import negative_sampling, dropout_edges
from utils.evaluation import  compute_uncertainty,compute_mrr_ensemble


class EnsemblePipeline(BasePipeline):
    
    def __init__(self, ensemble_model, data, config, logger):

        super().__init__(ensemble_model, data ,config, logger)

        self.ensemble = ensemble_model
        self.optimizers = self.ensemble.get_optimizers(
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.training_history.update({'ensemble_diversity': []})
    
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
            
            self.writer.add_scalar('Loss/Ensemble_Mean', avg_loss, epoch)
            
            self.logger.info(f'Epoch: {epoch:05d}, Avg Ensemble Loss: {avg_loss:.4f}')
            self.training_history['train_loss'].append({"epoch": epoch, "epoch_loss": avg_loss})
            
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
                
                # Log metrics
                self.logger.info(f"Epoch {epoch}: Ensemble Diversity = {diversity:.4f}")
                self.logger.info(f"Epoch {epoch}: Val MRR = {valid_scores['mrr']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Mean Rank = {valid_scores['mean_rank']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@1 = {valid_scores['hits@1']:.4f} ")
                self.logger.info(f"Epoch {epoch}: Val Hits@3 = {valid_scores['hits@3']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@10 = {valid_scores['hits@10']:.4f}")

                self.writer.add_scalar('MRR/Validation', valid_scores['mrr'], epoch)
                self.writer.add_scalar('Mean_Rank/Validation', valid_scores['mean_rank'], epoch)
                self.writer.add_scalar('Hits@1/Validation', valid_scores['hits@1'], epoch)
                self.writer.add_scalar('Hits@3/Validation', valid_scores['hits@3'], epoch)
                self.writer.add_scalar('Hits@10/Validation', valid_scores['hits@10'], epoch)

                scores = self.test_uncertainty(
                        self.model,
                        self.config.get_section('calibration')['method'],
                        self.data.edge_index,
                        self.data.edge_type,
                        self.data.valid_edge_index,
                        self.data.valid_edge_type,
                    )

                self.writer.add_scalar('MC_Uncertainty/Brier_Score', scores['brier_score'], epoch)
                self.writer.add_scalar('MC_Uncertainty/ECE', scores['ece'], epoch)

                
                # Check for improvement
                if current_val_mrr - best_val_mrr > delta:
                    best_val_mrr = current_val_mrr
                    patience_counter = 0
                    self.logger.info(f"New best ensemble! MRR: {best_val_mrr:.4f}")
                    
                    if self.train_config['save_model']:
                        self.save_checkpoint(epoch, name=f"{self.config.get_section('dataset')['name']}_ensemble_checkpoint_{self.config.get_section('calibration')['method']}_epoch_{epoch}.pth") 
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                
                # Early stopping
                if early_stopping and patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break
                
                self.training_history['eval_metrics'].append({
                    "epoch": epoch,
                    "val_mrr": valid_scores["mrr"],
                    "val_mean_rank": valid_scores["mean_rank"],
                    "val_hits@1": valid_scores["hits@1"],
                    "val_hits@3": valid_scores["hits@3"],
                    "val_hits@10": valid_scores["hits@10"],
                    "brier_score": scores['brier_score'],
                    "ece": scores['ece']

                })
        
        self.writer.close()
        self.logger.info("Ensemble training completed!")
        return self.training_history
    
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
    
    def _inference_helper(self, model, edge_index, edge_type, test_edge_index, test_edge_type, return_logits=False, apply_isotonic=False, enable_grad=False):
        """Helper method for standard inference without gradient control.
        
        Args:
            return_logits: If True, return raw logits; if False, return sigmoid probabilities
            apply_isotonic: If True, apply isotonic regression calibration (only for uncertainty evaluation)
        """

        # Get predictions with uncertainty
        mean_pred, std_pred = model.predict_with_uncertainty(
            edge_index,
            edge_type,
            test_edge_index,
            test_edge_type,
            enable_grad=enable_grad
        )
        
        # Also get negative samples for evaluation
        neg_edge_index, neg_edge_type = negative_sampling( test_edge_index, test_edge_type,self.data.num_nodes, 1)
        neg_mean_pred, neg_std_pred = model.predict_with_uncertainty(
            edge_index,
            edge_type,
            neg_edge_index,
            neg_edge_type,
            enable_grad=enable_grad
        )
        
        # Combine positive and negative predictions
        all_mean_pred = torch.cat([mean_pred, neg_mean_pred])
        all_std_pred = torch.cat([std_pred, neg_std_pred])
        labels = torch.cat([
            torch.ones_like(mean_pred),
            torch.zeros_like(neg_mean_pred)
        ])
        
        # Apply isotonic regression if requested (for uncertainty quantification only)
        if apply_isotonic and hasattr(model, 'isotonic_regression_transform') and model.isotonic_regression_transform is not None:
            device = all_mean_pred.device
            logits_np = all_mean_pred.cpu().numpy().flatten()
            calibrated_logits_np = model.isotonic_regression_transform.predict(logits_np)
            all_mean_pred = torch.tensor(calibrated_logits_np, dtype=torch.float32, device=device)
        elif not return_logits:
            all_mean_pred = torch.sigmoid(all_mean_pred)

        return all_mean_pred, labels

    @torch.no_grad()
    def inference(self, model, edge_index, edge_type, test_edge_index, test_edge_type):
        """Standard inference with gradients disabled."""
        return self._inference_helper(model, edge_index, edge_type, test_edge_index, test_edge_type, return_logits=False)


    @torch.no_grad()
    def test_uncertainty(self,model, method, edge_index, edge_type, test_edge_index, test_edge_type, uncertainty_samples = None):
        """
        Evaluate uncertainty on test set using ensemble variance.
        
        Returns uncertainty metrics: Brier score, ECE, etc.
        """
        
        # Check if isotonic regression calibration is available
        use_isotonic = (hasattr(model, 'isotonic_regression_transform') and 
                       model.isotonic_regression_transform is not None and
                       hasattr(model, 'use_isotonic_calibration') and
                       model.use_isotonic_calibration)
        
        all_mean_pred, labels = self._inference_helper(
                model,
                edge_index,
                edge_type,
                test_edge_index,
                test_edge_type,
                return_logits=False,
                apply_isotonic=use_isotonic,
                enable_grad=False
            )
        
        labels = labels.flatten()
        all_mean_pred = all_mean_pred.flatten()
        
        # Compute uncertainty metrics
        scores = compute_uncertainty(labels, all_mean_pred)
    
        self.logger.info(f"Ensemble Uncertainty - Brier Score: {scores['brier_score']:.4f}")
        self.logger.info(f"Ensemble Uncertainty - ECE: {scores['ece']:.4f}")
        self.logger.info(f"Probability True: {scores['prob_true']}")
        self.logger.info(f"Probability Predicted: {scores['prob_pred']}")

        return scores

    def load_pipeline(self, checkpoint_path, type,save):
        """Load ensemble pipeline from checkpoint."""
        self.load_checkpoint(checkpoint_path, load_optimizer=False)
        self.logger.info("Checkpoint loaded. Evaluating ensemble uncertainty on test set...")

        scores = self.test_link_pred(
            type=type,
            model=self.ensemble,
            valid_edge_index=self.data.valid_edge_index,
            valid_edge_type=self.data.valid_edge_type)
        
        scores = self.test_uncertainty(
            self.model,
            type,
            self.data.edge_index,
            self.data.edge_type,
            self.data.valid_edge_index,
            self.data.valid_edge_type,
            )
        
        calibration_results = self.calibrate_pipeline(
            method=self.config.get_section('calibration')['method'],
            model=self.ensemble,
            max_iters=self.config.get_section('calibration').get('max_iters', 50),
            lr=self.config.get_section('calibration').get('learning_rate', 0.01)
        )

        scores = self.test_link_pred(
            type=type,
            model=self.ensemble,
            valid_edge_index=self.data.valid_edge_index,
            valid_edge_type=self.data.valid_edge_type)
        
        scores = self.test_uncertainty(
            self.model,
            type,
            self.data.edge_index,
            self.data.edge_type,
            self.data.test_edge_index,
            self.data.test_edge_type,
            )
        if save:
            self.save_checkpoint(epoch=0, name=f'calibrated_{Path(checkpoint_path).name}')

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
    
    def save_checkpoint(self, epoch, name=None):
        """Save ensemble checkpoint."""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        if name is None:
            name = f'{self.config.get_section("dataset")["name"]}_ensemble_checkpoint_epoch_{epoch}.pth'
           

        checkpoint_path = checkpoint_dir / name
        
        checkpoint = {
            'epoch': epoch,
            'num_models': self.ensemble.num_models,
            'encoder_args': self.ensemble.encoder_args,
            'decoder_args': self.ensemble.decoder_args,
            'models': [model.state_dict() for model in self.ensemble.models],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Ensemble checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=False):
        """Load ensemble checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        for i, state_dict in enumerate(checkpoint['models']):
            self.ensemble.models[i].load_state_dict(state_dict, strict=False)
        
        if load_optimizer:
            for i, state_dict in enumerate(checkpoint['optimizers']):
                self.optimizers[i].load_state_dict(state_dict)
        
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'eval_metrics': [], 'ensemble_diversity': []
        })
        self.epoch = checkpoint['epoch']
        
        self.logger.info(f"Ensemble checkpoint loaded from {checkpoint_path}")

    def calibrate_pipeline(self, method, model, max_iters=50, lr=0.01):
        """Main entry point for calibration."""

        self.logger.info("Starting calibration process...")
        type_params = {
            'type': self.config.get_section('calibration').get('type', 'standard'),
            'mc_samples': self.config.get_section('calibration').get('mc_samples', 10)
        }
        self.logger.info(f"Uncertainty model type: {type_params}")
        if method == 'scalar':
            return self.calibrate_scalar_temperature(model, max_iters, lr, type_params)
        elif method == 'input_dependent':
            return self.calibrate_input_dependent_temperature(model, max_iters, lr, type_params)
        elif method == 'isotonic_regression':
            return self.calibrate_isotonic_regression(model)
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
    
    def _get_temperature_stats(self, model, edge_index, edge_type, num_samples=None):
        """Compute temperature statistics for validation samples.
        
        Returns:
            dict: Temperature statistics (mean, std, min, max)
        """
        with torch.no_grad():
            all_temps = []
            for member_model in model.models:
                z = member_model.encode(self.data.edge_index, self.data.edge_type)
                if num_samples is not None:
                    sample_edge_index = edge_index[:, :num_samples]
                    sample_edge_type = edge_type[:num_samples]
                else:
                    sample_edge_index = edge_index
                    sample_edge_type = edge_type
                
                heads = z[sample_edge_index[0]]
                rels = member_model.decoder.rel_emb[sample_edge_type]
                temps = member_model.decoder.compute_temperature(heads, rels)
                all_temps.append(temps)
            
            temps = torch.cat(all_temps, dim=0)
            
            return {
                'mean': temps.mean().item(),
                'std': temps.std().item(),
                'min': temps.min().item(),
                'max': temps.max().item()
            }

    def compute_nll_loss(self, model, params = None):
        """
        Compute Negative Log-Likelihood loss for calibration.
        
        Args:
            model: The GAE model
            params: Parameters dict with 'type' and 'mc_samples'
            
        Returns:
            NLL loss value
        """
        logits, labels = self._inference_helper(
                model,
                self.data.edge_index,
                self.data.edge_type,
                self.data.valid_edge_index,
                self.data.valid_edge_type,
                return_logits=True,
                enable_grad=True
            )
        
        nll_loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return nll_loss

    def calibrate_input_dependent_temperature(self, model, max_iters=50, lr=0.01, type_params= None):
        """Calibrate input-dependent temperature network on validation set for ensemble.
        
        This method learns a small neural network that predicts per-query
        temperature values T(h,r) for each query, allowing each model in the
        ensemble to express uncertainty adaptively while preserving ranking.
        
        Args:
            model: Ensemble model with multiple GAE models
            max_iters: Maximum optimization iterations
            lr: Learning rate for Adam optimizer
            type_params: Additional parameters for calibration
            
        Returns:
            dict: Final temperature statistics and loss

        """

        self.logger.info(f"Calibration method: {self.config.get_section('calibration')['method']}")

        stats = self._get_temperature_stats(
                    model, 
                    self.data.valid_edge_index, 
                    self.data.valid_edge_type, 
                    num_samples=100
                )
        self._log_temperature_stats(stats, prefix="  Sample ")
        
        # Enable input-dependent temperature for all models in the ensemble
        temp_enabled = True
        for member_model in model.models:
            if hasattr(member_model.decoder, 'use_input_dependent_temp'):
                member_model.decoder.use_input_dependent_temp = True
            else:
                self.logger.error(f"Model decoder does not support input-dependent temperature!")
                temp_enabled = False
                break
        
        if not temp_enabled:
            return {}
        
        # Collect temperature parameters from all ensemble members
        temp_params = []
        for member_model in model.models:
            for name, param in member_model.named_parameters():
                if 'temp_network' in name or 'temperature' in name:
                    param.requires_grad = True
                    temp_params.append(param)
                else:
                    param.requires_grad = False
        
        if not temp_params:
            self.logger.error("No temperature parameters found in ensemble!")
            return {}
        
        self.logger.info(f"Collected {len(temp_params)} temperature parameters from {len(model.models)} models")
        optimizer = torch.optim.Adam(temp_params, lr=lr)
        
        self.logger.info("="*60)
        self.logger.info("Starting Input-Dependent Temperature Calibration")
        self.logger.info(f"Parameters to optimize: {len(temp_params)}")
        self.logger.info(f"Max iterations: {max_iters}, Learning rate: {lr}")
        self.logger.info("="*60)

        
        # Training loop with early stopping
        best_loss = float('inf')
        patience, patience_counter = 5, 0
        
        for iteration in range(1, max_iters + 1):
            # Optimization step
            optimizer.zero_grad()
            loss = self.compute_nll_loss(model, params=type_params)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temp_params, max_norm=1.0)
            optimizer.step()
            
            # Track best loss for early stopping
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Periodic logging
            if iteration % 10 == 0:
                self.logger.info(f"Iter {iteration}/{max_iters}: NLL={loss_val:.4f}, Best={best_loss:.4f}")
                
                # Log sample temperature distribution
                stats = self._get_temperature_stats(
                    model, 
                    self.data.valid_edge_index, 
                    self.data.valid_edge_type, 
                    num_samples=100
                )
                self._log_temperature_stats(stats, prefix="  Sample ")
            
            # Early stopping check
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at iteration {iteration}")
                break
        
        # Compute and log final statistics
        final_stats = self._get_temperature_stats(
            model,
            self.data.valid_edge_index,
            self.data.valid_edge_type
        )
        final_stats['final_loss'] = best_loss
        
        self.logger.info("="*60)
        self.logger.info("Calibration Complete!")
        self._log_temperature_stats(final_stats, prefix="Final ")
        self.logger.info(f"Final NLL Loss: {best_loss:.4f}")
        self.logger.info("="*60)
        
        # Unfreeze all parameters in all ensemble members
        for member_model in model.models:
            for param in member_model.parameters():
                param.requires_grad = True
        
        return final_stats

    def calibrate_scalar_temperature(self, model, max_iters=50, lr=0.01, type_params = None):
        """Calibrate scalar temperature parameters on validation set for ensemble.
        
        This is the traditional temperature scaling approach where each model
        in the ensemble has its own scalar T that divides all logits uniformly.
        
        Args:
            model: Ensemble model with scalar temperature parameters
            max_iters: Maximum LBFGS iterations
            lr: Learning rate for LBFGS
            type_params: Additional parameters for calibration
            
        Returns:
            dict: Calibrated temperature values for each model
        """
        self.logger.info(f"Calibration method: {self.config.get_section('calibration')['method']}")
        
        # Ensure input-dependent temperature is disabled for all models in ensemble
        for member_model in model.models:
            if hasattr(member_model.decoder, 'use_input_dependent_temp'):
                member_model.decoder.use_input_dependent_temp = False

        # Collect temperature parameters from all ensemble members
        temp_params = []
        initial_temps = []
        for member_model in model.models:
            for name, param in member_model.named_parameters():
                if 'temperature' in name and 'temp_network' not in name:
                    param.requires_grad = True
                    temp_params.append(param)
                    initial_temps.append(param.item())
                else:
                    param.requires_grad = False
        
        if not temp_params:
            self.logger.error("No temperature parameters found in ensemble!")
            return {}
        
        self.logger.info("="*60)
        self.logger.info("Starting Scalar Temperature Calibration for Ensemble")
        self.logger.info(f"Number of models: {len(model.models)}")
        self.logger.info(f"Initial temperatures: {initial_temps}")
        self.logger.info("="*60)
        
        # Use Adam optimizer for ensemble (LBFGS can be unstable with multiple parameters)
        optimizer = torch.optim.Adam(temp_params, lr=lr)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience, patience_counter = 5, 0
        
        for iteration in range(1, max_iters + 1):
            optimizer.zero_grad()
            loss = self.compute_nll_loss(model, params=type_params)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temp_params, max_norm=1.0)
            optimizer.step()
            
            # Track best loss for early stopping
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            if iteration % 10 == 0:
                current_temps = [param.item() for param in temp_params]
                self.logger.info(
                    f"Iter {iteration}/{max_iters}: "
                    f"NLL={loss_val:.4f}, Best={best_loss:.4f}, T={current_temps}"
                )
            
            # Early stopping check
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at iteration {iteration}")
                break
        
        final_temps = [param.item() for param in temp_params]
        
        self.logger.info("="*60)
        self.logger.info("Calibration Complete!")
        self.logger.info(f"Final temperatures: {final_temps}")
        self.logger.info(f"Temperature changes: {[f'{final-init:+.4f}' for final, init in zip(final_temps, initial_temps)]}")
        self.logger.info(f"Final NLL Loss: {best_loss:.4f}")
        self.logger.info("="*60)
        
        # Unfreeze all parameters in all ensemble members
        for member_model in model.models:
            for param in member_model.parameters():
                param.requires_grad = True
        
        return {f'model_{i}_temp': t for i, t in enumerate(final_temps)}
    
    def calibrate_isotonic_regression(self, model):
        """Calibrate using isotonic regression on validation set for ensemble.
        
        For ensembles, we fit isotonic regression on the ensemble mean predictions
        to calibrate the aggregated uncertainty estimates.
        
        Args:
            model: Ensemble model
            
        Returns:
            dict: Calibration results
        """
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        self.logger.info("Starting Isotonic Regression Calibration for Ensemble")

        # Get ensemble predictions (raw logits) and labels
        model.eval()
        with torch.no_grad():
            logits, labels = self._inference_helper(
                model,
                self.data.edge_index,
                self.data.edge_type,
                self.data.valid_edge_index,
                self.data.valid_edge_type,
                return_logits=True  # Get raw ensemble mean logits
            )
        
        logits_np = logits.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        # Fit isotonic regression on ensemble mean predictions
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(logits_np, labels_np)

        self.logger.info("Isotonic Regression model fitted on ensemble predictions.")
        
        # Store the isotonic regression transform in the ensemble
        # This will be applied to the ensemble mean predictions
        model.isotonic_regression_transform = iso_reg
        model.use_isotonic_calibration = True
        
        self.logger.info("Calibration Complete!")

        return {"isotonic_model": iso_reg}


