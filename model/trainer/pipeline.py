from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import torch
from model.trainer.basepipeline import BasePipeline
from utils.utils import negative_sampling
from utils.evaluation import compute_mrr, compute_uncertainty, compute_mrr_mc_dropout
from utils.utils import dropout_edges
from model.trainer.basepipeline import BasePipeline


class Pipeline(BasePipeline):

    def __init__(self, model, data, config, logger):
        super().__init__(model, data, config, logger)
        

    def start_pipeline(self):
        max_epochs = self.train_config['epochs']
        eval_frequency = self.train_config.get('evaluation_frequency', 10)
        early_stopping = self.train_config['early stopping']['enabled']
        patience = self.train_config['early stopping'].get('patience', 10)
        delta = self.train_config['early stopping'].get('delta', 0.0)
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        best_val_mrr = -float('inf') 
        patience_counter = 0
        tqdm_range = range(1, max_epochs + 1)
        tqdm_range = tqdm(tqdm_range, desc="Training", unit="batch") 
        self.logger.info(f"Negative sampling ratio: {self.train_config['sampling']['negative_sampling_ratio']}")       

        for epoch in tqdm_range:
            
            loss = self.train()
            val_loss = 0
            print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
            self.writer.add_scalar('Loss/Train', loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)

            self.training_history['train_loss'].append({"epoch": epoch, "epoch_loss": loss})
            self.training_history['val_loss'].append({"epoch": epoch, "epoch_loss": val_loss})

            # Log gradients periodically
            if epoch % 10 == 0:  # Log gradients every 10 epochs
                self.log_model_gradients(epoch)


            # Evaluation
            if epoch % eval_frequency == 0:

                valid_scores, test_scores = self.test(test=False)

                current_val_mrr = valid_scores['mrr']
            
                # Check if current model is better than the best found so far
                if current_val_mrr - best_val_mrr > delta:
                    best_val_mrr = current_val_mrr
                    patience_counter = 0
                    
                    self.logger.info(f"New best model found! MRR: {current_val_mrr:.4f} > Previous: {best_val_mrr:.4f}")
                    if self.train_config['save_model']:
                        self.save_checkpoint(epoch) 
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                # Early Stopping Trigger
                if early_stopping and patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

                if test_scores is None:
                    test_scores = {"mrr": 0, "mean_rank": 0, "hits@1": 0, "hits@3": 0, "hits@10": 0}


                self.training_history['eval_metrics'].append({
                    "epoch": epoch,
                    "val_mrr": valid_scores["mrr"],
                    "val_mean_rank": valid_scores["mean_rank"],
                    "val_hits@1": valid_scores["hits@1"],
                    "val_hits@3": valid_scores["hits@3"],
                    "val_hits@10": valid_scores["hits@10"],
                    "test_mrr": test_scores["mrr"],
                    "test_mean_rank": test_scores["mean_rank"],
                    "test_hits@1": test_scores["hits@1"],  
                    "test_hits@3": test_scores["hits@3"],
                    "test_hits@10": test_scores["hits@10"],
                })
                self.logger.info(f"Epoch {epoch}: Val MRR = {valid_scores['mrr']:.4f}, Test MRR = {test_scores['mrr']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Mean Rank = {valid_scores['mean_rank']:.4f}, Test Mean Rank = {test_scores['mean_rank']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@1 = {valid_scores['hits@1']:.4f}, Test Hits@1 = {test_scores['hits@1']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@3 = {valid_scores['hits@3']:.4f}, Test Hits@3 = {test_scores['hits@3']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Hits@10 = {valid_scores['hits@10']:.4f}, Test Hits@10 = {test_scores['hits@10']:.4f}")

                # Log evaluation metrics to TensorBoard
                self.writer.add_scalar('MRR/Validation', valid_scores['mrr'], epoch)
                self.writer.add_scalar('MRR/Test', test_scores['mrr'], epoch)
                self.writer.add_scalar('Mean_Rank/Validation', valid_scores['mean_rank'], epoch)
                self.writer.add_scalar('Mean_Rank/Test', test_scores['mean_rank'], epoch)
                self.writer.add_scalar('Hits@1/Validation', valid_scores['hits@1'], epoch)
                self.writer.add_scalar('Hits@1/Test', test_scores['hits@1'], epoch)
                self.writer.add_scalar('Hits@3/Validation', valid_scores['hits@3'], epoch)
                self.writer.add_scalar('Hits@3/Test', test_scores['hits@3'], epoch)
                self.writer.add_scalar('Hits@10/Validation', valid_scores['hits@10'], epoch)
                self.writer.add_scalar('Hits@10/Test', test_scores['hits@10'], epoch)

                scores = self.test_uncertainty(
                        self.model,
                        self.config.get_section('calibration')['type'],
                        self.data.edge_index,
                        self.data.edge_type,
                        self.data.valid_edge_index,
                        self.data.valid_edge_type,
                        uncertainty_samples=self.config.get_section('calibration')['mc_samples']
                    )

                self.writer.add_scalar('MC_Uncertainty/Brier_Score', scores['brier_score'], epoch)
                self.writer.add_scalar('MC_Uncertainty/ECE', scores['ece'], epoch)

        
        # Close TensorBoard writer
        self.writer.close()
        self.logger.info("Training completed!")
        return self.training_history

    def load_pipeline(self, checkpoint_path, type,save):


        self.load_checkpoint(checkpoint_path)
        self.logger.info("Pipeline loaded from checkpoint.")
        uncertainty_samples = self.config.get_section('calibration')['mc_samples']
        scores = self.test_link_pred(
            type=type,
            model=self.model,
            valid_edge_index=self.data.test_edge_index,
            valid_edge_type=self.data.test_edge_type,
            mc_samples=uncertainty_samples)

        scores = self.test_uncertainty(
            self.model,
            type,
            self.data.edge_index,
            self.data.edge_type,
            self.data.test_edge_index,
            self.data.test_edge_type,
            uncertainty_samples
            )

        calibration_model = self.calibrate_pipeline(
            method=self.config.get_section('calibration')['method'],
            model=self.model,
            max_iters=self.config.get_section('calibration').get('max_iters', 50),
            lr=self.config.get_section('calibration').get('learning_rate', 0.01)
        )

        scores = self.test_link_pred(
            type=type,
            model=self.model,
            valid_edge_index=self.data.test_edge_index,
            valid_edge_type=self.data.test_edge_type,
            mc_samples=uncertainty_samples)


        scores = self.test_uncertainty(
            self.model,
            type,
            self.data.edge_index,
            self.data.edge_type,
            self.data.test_edge_index,
            self.data.test_edge_type,
            uncertainty_samples
        )
        if save:
            self.save_checkpoint(self.epoch, name=f'calibrated_{Path(checkpoint_path).name}')
        
    def _inference_mc_helper(self, model, edge_index, edge_type, test_edge_index, test_edge_type, mc_samples=10, return_logits=False):
        """Helper method for MC dropout inference without gradient control.
        
        Args:
            return_logits: If True, return raw logits; if False, return sigmoid probabilities
        """
        self.model.eval()
        if mc_samples > 1:
            self.model.encoder.mc_dropout = True  

        neg_edge_index, neg_edge_type = negative_sampling(test_edge_index, test_edge_type, self.data.num_nodes, 1)
      
        preds_list = []
        pos_out = None
        neg_out = None

        for _ in range(mc_samples):
            z = self.model.encode(edge_index, edge_type)
            pos_out = self.model.decode(z, test_edge_index, test_edge_type)
            neg_out = self.model.decode(z, neg_edge_index, neg_edge_type)
            out = torch.cat([pos_out, neg_out])
            preds_list.append(out)

        self.model.encoder.mc_dropout = False
        
        # pos_out and neg_out are guaranteed to be set after the loop
        assert pos_out is not None and neg_out is not None
        labels = torch.cat([
            torch.ones_like(pos_out),
            torch.zeros_like(neg_out)
        ])
        
        preds_stack = torch.stack(preds_list)
        preds_mean = preds_stack.mean(dim=0)
        
        # Only apply calibration when returning probabilities, not raw logits
        if not return_logits:
            # First convert logits to probabilities
            preds_mean = torch.sigmoid(preds_mean)
            
            # Then apply isotonic regression to probabilities if available
            if hasattr(model.decoder, 'isotonic_regression_transform') and model.decoder.isotonic_regression_transform is not None and model.decoder.use_calibration:
                device = preds_mean.device
                preds_mean_np = preds_mean.cpu().numpy().flatten()
                calibrated_preds_mean_np = self.model.decoder.isotonic_regression_transform.predict(preds_mean_np)
                preds_mean = torch.tensor(calibrated_preds_mean_np, dtype=torch.float32, device=device)

        return preds_mean, labels

    @torch.no_grad()
    def inference_mc(self,model, edge_index, edge_type, test_edge_index, test_edge_type, mc_samples=10):
        """MC dropout inference with gradients disabled."""
        return self._inference_mc_helper(model, edge_index, edge_type, test_edge_index, test_edge_type, mc_samples, return_logits=False)

    def _inference_helper(self, model, edge_index, edge_type, test_edge_index, test_edge_type, return_logits=False):
        """Helper method for standard inference without gradient control.
        
        Args:
            return_logits: If True, return raw logits; if False, return sigmoid probabilities
            apply_isotonic: If True, apply isotonic regression calibration (only for uncertainty evaluation)
        """
        model.eval()
        z = model.encoder(edge_index, edge_type)
        pos_scores = model.decode(z, test_edge_index, test_edge_type)
        neg_edge_index, neg_edge_type = negative_sampling(test_edge_index, test_edge_type, self.data.num_nodes, 1)
        neg_scores = model.decode(z, neg_edge_index, neg_edge_type)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        
        # Only apply calibration when returning probabilities, not raw logits
        if not return_logits:
            # First convert logits to probabilities
            scores = torch.sigmoid(scores)
            
            # Then apply isotonic regression to probabilities if available
            if hasattr(model.decoder, 'isotonic_regression_transform') and model.decoder.isotonic_regression_transform is not None and model.decoder.use_calibration:
                device = scores.device
                scores_np = scores.cpu().numpy().flatten()
                calibrated_scores_np = model.decoder.isotonic_regression_transform.predict(scores_np)
                scores = torch.tensor(calibrated_scores_np, dtype=torch.float32, device=device)

        return scores, labels

    @torch.no_grad()
    def inference(self, model, edge_index, edge_type, test_edge_index, test_edge_type, apply_isotonic=False):
        """Standard inference with gradients disabled."""
        return self._inference_helper(model, edge_index, edge_type, test_edge_index, test_edge_type, return_logits=False)

    def train(self):
        """
        Train the model on a single batch.
        """

        self.model.train()
        # Ensure input-dependent temperature is disabled during training
        if hasattr(self.model.decoder, 'use_input_dependent_temp'):
            self.model.decoder.use_input_dependent_temp = False
        
        self.optimizer.zero_grad()

        # dropout some edge randomly for training
        if self.train_config['sampling']['edge_dropout'] > 0:
            edge_index, edge_type = dropout_edges(self.data.edge_index, self.data.edge_type, self.train_config['sampling']['edge_dropout'])
        else:
            edge_index, edge_type = self.data.edge_index, self.data.edge_type

        z = self.model.encode(edge_index, edge_type)

        pos_out = self.model.decode(z, self.data.train_edge_index, self.data.train_edge_type)

        neg_edge_index,neg_edge_type = negative_sampling(self.data.train_edge_index, self.data.train_edge_type, self.data.num_nodes, self.train_config['sampling']['negative_sampling_ratio'])
        neg_out = self.model.decode(z, neg_edge_index, neg_edge_type)

        out = torch.cat([pos_out, neg_out])
        
        gt = torch.cat([torch.full_like(pos_out, self.train_config['label_smoothing']['positive']), torch.full_like(neg_out, self.train_config['label_smoothing']['negative'])])
        
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()

        loss = cross_entropy_loss + self.model_config['decoder']['l2_penalty'] * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()

        return float(loss)

    @torch.no_grad()
    def test(self, test = True):

        self.model.eval()
        z = self.model.encode(self.data.edge_index, self.data.edge_type)
        valid_scores = compute_mrr(self.data.valid_edge_index, self.data.valid_edge_type,self.data, self.model)
        if test:
            test_scores = compute_mrr(self.data.test_edge_index, self.data.test_edge_type,self.data, self.model)
            return valid_scores, test_scores

        return valid_scores, None

    @torch.no_grad()
    def test_uncertainty(self, model, method, edge_index, edge_type, test_edge_index, test_edge_type, uncertainty_samples=None):

        
        if method == 'mc_dropout':
            assert uncertainty_samples is not None, "MC Dropout requires specifying number of samples."
            scores, labels = self.inference_mc(
                model,
                edge_index,
                edge_type,
                test_edge_index,
                test_edge_type,
                mc_samples=uncertainty_samples
            )
            val_scores = compute_uncertainty(labels,scores)
            
        elif method == 'standard':
        
            scores, labels = self.inference(
                model, edge_index, edge_type, test_edge_index, test_edge_type,
            )
            val_scores = compute_uncertainty(labels,scores)

        else:
            raise ValueError(f"Unsupported uncertainty estimation method: {method}")

        self.logger.info(f"Brier Score: {val_scores['brier_score']:.4f}")
        self.logger.info(f"ECE: {val_scores['ece']:.4f}")
        self.logger.info(f"ACE: {val_scores['ace']:.4f}")
        self.logger.info(f"Probability True: {val_scores['prob_true']}")
        self.logger.info(f"Probability Predicted: {val_scores['prob_pred']}")

        self.logger.info(f" {val_scores}")

        return val_scores

    def calibrate_pipeline(self, method, model, max_iters=50, lr=0.01):
        """Main entry point for calibration."""

        self.logger.info("Starting calibration process...")
        type_params = {
            'type': self.config.get_section('calibration').get('type', 'standard'),
            'mc_samples': self.config.get_section('calibration').get('mc_samples', 10),
            'return_logits': True
        }
        self.logger.info(f"Uncertainty model type: {type_params}")
        if method == 'scalar':
            return self.calibrate_scalar_temperature(model, max_iters, lr, type_params)
        elif method == 'input_dependent':
            return self.calibrate_input_dependent_temperature(model, max_iters, lr, type_params)
        elif method == 'isotonic_regression':
            return self.calibrate_isotonic_regression(model, type_params)
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
    
    def _get_temperature_stats(self, model, edge_index, edge_type, num_samples=None):
        """Compute temperature statistics for validation samples.
        
        Returns:
            dict: Temperature statistics (mean, std, min, max)
        """
        with torch.no_grad():
            z = model.encode(self.data.edge_index, self.data.edge_type)
            
            if num_samples is not None:
                edge_index = edge_index[:, :num_samples]
                edge_type = edge_type[:num_samples]
            
            heads = z[edge_index[0]]
            rels = model.decoder.rel_emb[edge_type]
            temps = model.decoder.compute_temperature(heads, rels)
            
            return {
                'mean': temps.mean().item(),
                'std': temps.std().item(),
                'min': temps.min().item(),
                'max': temps.max().item()
            }

    def compute_nll_loss(self, model, params):
        """
        Compute Negative Log-Likelihood loss for calibration.
        
        Args:
            model: The GAE model
            params: Parameters dict with 'type' and 'mc_samples'
            
        Returns:
            NLL loss value
        """
        # For calibration, we need gradients to flow through temperature
        # Call helper methods without @torch.no_grad() decorator to enable gradients
        
        if params['type'] == 'mc_dropout':
            logits, labels = self._inference_mc_helper(
                model,
                self.data.edge_index,
                self.data.edge_type,
                self.data.valid_edge_index,
                self.data.valid_edge_type,
                mc_samples=params['mc_samples'],
                return_logits=params['return_logits'] 
            )
        elif params['type'] == 'standard':
            logits, labels = self._inference_helper(
                model,
                self.data.edge_index,
                self.data.edge_type,
                self.data.valid_edge_index,
                self.data.valid_edge_type,
                return_logits=params['return_logits'] 
            )
        else:
            raise ValueError(f"Unsupported evaluation method: {params['type']}")
        
        nll_loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return nll_loss

    def calibrate_input_dependent_temperature(self, model, max_iters=50, lr=0.01, type_params= None):
        """Calibrate input-dependent temperature network on validation set.
        
        This method learns a small neural network that predicts per-query
        temperature values T(h,r) for each query, allowing the model to
        express uncertainty adaptively while preserving ranking.
        
        Args:
            model: GAE model with temp_network
            max_iters: Maximum optimization iterations
            lr: Learning rate for Adam optimizer
            
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
        
        # Enable input-dependent temperature for this calibration method
        if hasattr(model.decoder, 'use_input_dependent_temp'):
            model.decoder.use_input_dependent_temp = True
        else:
            self.logger.error("Model decoder does not support input-dependent temperature!")
            return {}
        
        # Freeze all parameters except temperature network
        temp_params = self._freeze_non_temperature_params(
            model, 
            lambda name: 'temp_network' in name or 'temperature' in name
        )
        
        if not temp_params:
            return {}
        
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
        
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        return final_stats

    def calibrate_scalar_temperature(self, model, max_iters=50, lr=0.01, type_params = None):
        """Calibrate single scalar temperature parameter on validation set.
        
        This is the traditional temperature scaling approach where a single
        scalar T divides all logits uniformly.
        
        Args:
            model: GAE model with scalar temperature parameter
            max_iters: Maximum LBFGS iterations
            lr: Learning rate for LBFGS
            
        Returns:
            float: Calibrated temperature value
        """
        self.logger.info(f"Calibration method: {self.config.get_section('calibration')['method']}")
        
        # Ensure input-dependent temperature is disabled for scalar calibration
        if hasattr(model.decoder, 'use_input_dependent_temp'):
            model.decoder.use_input_dependent_temp = False

        # Freeze all parameters except temperature
        temp_params = self._freeze_non_temperature_params(
            model,
            lambda name: 'temperature' in name
        )
        
        if not temp_params:
            self.logger.error("No temperature parameter found!")
            return None
        
        initial_temp = model.decoder.temperature.item()
        self.logger.info("="*60)
        self.logger.info("Starting Scalar Temperature Calibration")
        self.logger.info(f"Initial temperature: {initial_temp:.4f}")
        self.logger.info("="*60)
        
        # Use LBFGS for scalar optimization (quasi-Newton method)
        optimizer = torch.optim.LBFGS(temp_params, lr=lr, max_iter=max_iters)
        
        def eval_closure():
            optimizer.zero_grad()
            loss = self.compute_nll_loss(model, params=type_params)
            loss.backward()
            return loss
        
        # Optimize temperature
        for iteration in range(1, max_iters + 1):
            loss = optimizer.step(eval_closure)
            current_temp = model.decoder.temperature.item()
            
            if iteration % 10 == 0:
                self.logger.info(
                    f"Iter {iteration}/{max_iters}: "
                    f"NLL={loss.item():.4f}, T={current_temp:.4f}"
                )
        
        final_temp = model.decoder.temperature.item()
        
        self.logger.info("="*60)
        self.logger.info("Calibration Complete!")
        self.logger.info(f"Final temperature: {final_temp:.4f} (Δ={final_temp - initial_temp:+.4f})")
        self.logger.info("="*60)
        
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        return final_temp
    
    def calibrate_isotonic_regression(self, model, type_params):
        """Calibrate using isotonic regression on validation set.
        
        Args:
            model: GAE model
            
        Returns:
            dict: Calibration results
        """
        from sklearn.isotonic import IsotonicRegression

        self.logger.info("Starting Isotonic Regression Calibration")

        model.eval()
        with torch.no_grad():
            if type_params['type'] == 'mc_dropout':
                logits, labels = self._inference_mc_helper(
                    model,
                    self.data.edge_index,
                    self.data.edge_type,
                    self.data.valid_edge_index,
                    self.data.valid_edge_type,
                    mc_samples=type_params['mc_samples'],
                    return_logits=True,
                )
            elif type_params['type'] == 'standard':
                logits, labels = self._inference_helper(
                    model,
                    self.data.edge_index,
                    self.data.edge_type,
                    self.data.valid_edge_index,
                    self.data.valid_edge_type,
                    return_logits=True,
                )
            else:
                raise ValueError(f"Unsupported evaluation method: {type_params['type']}")
            
        # Convert logits to probabilities for isotonic regression
        # Isotonic regression should be fit on bounded [0,1] probabilities, not unbounded logits
        probs = torch.sigmoid(logits)
        probs_np = probs.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(probs_np, labels_np)

        self.logger.info("Isotonic Regression model fitted.")
        self.model.decoder.isotonic_regression_transform = iso_reg
        self.model.decoder.use_calibration = True
        self.logger.info("Calibration Complete!")

        return {"isotonic_model": iso_reg}

