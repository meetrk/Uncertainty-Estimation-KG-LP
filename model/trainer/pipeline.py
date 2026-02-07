from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import torch
from model.trainer.basepipeline import BasePipeline
from utils.utils import negative_sampling
from utils.evaluation import compute_mrr, compute_mrr_mc_dropout
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

                valid_scores, test_scores = self.test(test=self.train_config.get('test', True))

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
                self.logger.info(f"Epoch {epoch}: Val ECE= {valid_scores['ece']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val ACE= {valid_scores['ace']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val Brier Score = {valid_scores['brier_score']:.4f}")
                self.logger.info(f"Epoch {epoch}: Val True Probability = {valid_scores['prob_true']}")
                self.logger.info(f"Epoch {epoch}: Val Predicted Probability = {valid_scores['prob_pred']}")

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
                self.writer.add_scalar('MC_Uncertainty/Brier_Score', valid_scores['brier_score'], epoch)
                self.writer.add_scalar('MC_Uncertainty/ECE', valid_scores['ece'], epoch)
                self.writer.add_scalar('MC_Uncertainty/ACE', valid_scores['ace'], epoch)

        
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
        
        if self.config.get_section('calibration')['enabled']:
            self.logger.info("Starting calibration on test set...")
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

        else:
            self.logger.info("Calibration not enabled; skipping calibration step.")

        if save:
            self.save_checkpoint(self.epoch, name=f'calibrated_{Path(checkpoint_path).name}')
        

    def train(self):
        """
        Train the model on a single batch.
        """

        self.model.train()
        # Ensure input-dependent temperature is disabled during training
        if hasattr(self.model.decoder, 'use_calibration'):
            self.model.decoder.use_calibration = False
        
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
        valid_scores = compute_mrr(self.data.valid_edge_index, self.data.valid_edge_type,self.data, self.model, return_probs= False)
        if test:
            test_scores = compute_mrr(self.data.test_edge_index, self.data.test_edge_type,self.data, self.model, return_probs= False)
            return valid_scores, test_scores

        return valid_scores, None


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
        if hasattr(model.decoder, 'use_calibration'):
            model.decoder.use_calibration = True
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
            loss = self.compute_nll_loss_ranking(model, params=type_params)
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
                    num_samples=10
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
        
        self.model.decoder.use_calibration = True

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
            loss = self.compute_nll_loss_ranking(model, params=type_params)
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
                probs, labels = compute_mrr_mc_dropout(
                    self.data.edge_index,
                    self.data.edge_type,
                    self.data.valid_edge_index,
                    self.data.valid_edge_type,
                    self.data,
                    model,
                    mc_samples=type_params['mc_samples'],
                    return_probs=True
                )
            elif type_params['type'] == 'standard':
                probs, labels = compute_mrr(
                    self.data.valid_edge_index,
                    self.data.valid_edge_type,
                    self.data,
                    model,
                    return_probs=True
                )
            else:
                raise ValueError(f"Unsupported evaluation method: {type_params['type']}")
            

        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(probs, labels)

        self.logger.info("Isotonic Regression model fitted.")
        self.model.decoder.isotonic_regression_transform = iso_reg
        self.model.decoder.use_calibration = True
        self.logger.info("Calibration Complete!")

        return {"isotonic_model": iso_reg}



    def compute_nll_loss_ranking(self, model, params):
        """
        Compute NLL loss for calibration using ranking setup (same as MRR evaluation).
        """
        if params['type'] == 'mc_dropout':
            model.encoder.mc_dropout = True
            # Encode multiple times for MC dropout
            all_z = []
            for _ in range(params['mc_samples']):
                z = model.encode(self.data.edge_index, self.data.edge_type)
                all_z.append(z)
            model.encoder.mc_dropout = False
        else:
            z = model.encode(self.data.edge_index, self.data.edge_type)
        
        total_loss = torch.tensor(0.0, requires_grad=True)
        num_edges = min(64, self.data.valid_edge_type.numel())  # Subsample for efficiency
        
        for i in range(num_edges):
            src = self.data.valid_edge_index[0, i].item()
            dst = self.data.valid_edge_index[1, i].item()
            rel = self.data.valid_edge_type[i].item()
            
            # --- TAIL PREDICTION (same filtering as MRR) ---
            tail_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
            for (heads, tails), types in [
                (self.data.train_edge_index, self.data.train_edge_type),
                (self.data.valid_edge_index, self.data.valid_edge_type),
                (self.data.test_edge_index, self.data.test_edge_type),
            ]:
                tail_mask[tails[(heads == src) & (types == rel)]] = False
            
            tail = torch.arange(self.data.num_nodes)[tail_mask]
            tail = torch.cat([torch.tensor([dst]), tail])
            head = torch.full_like(tail, fill_value=src)
            eval_edge_index = torch.stack([head, tail], dim=0)
            eval_edge_type = torch.full_like(tail, fill_value=rel)
            
            # Decode
            if params['type'] == 'mc_dropout':
                predictions = []
                for z_sample in all_z:
                    pred = model.decode(z_sample, eval_edge_index, eval_edge_type)
                    predictions.append(pred)
                logits = torch.stack(predictions).mean(dim=0)
            else:
                logits = model.decode(z, eval_edge_index, eval_edge_type)
            k=10

            pos_logit = logits[0]
            neg_logits = logits[1:]

            if len(neg_logits) > k:
                neg_logits, _ = torch.sort(neg_logits, descending=True)

                # indices = torch.linspace(0, len(neg_logits) - 1, steps=k).long()
                target_values = torch.linspace(0, 1, steps=k,device=neg_logits.device)
                indices = torch.searchsorted(neg_logits, target_values)
                indices = torch.clamp(indices, max=len(neg_logits)-1)
                sampled_neg_logits = neg_logits[indices]
            else:
                sampled_neg_logits = neg_logits

            # 4. Binary Cross Entropy style loss (Contrastive)
            pos_loss = -F.logsigmoid(pos_logit)
            neg_loss = -F.logsigmoid(-sampled_neg_logits).sum()

            loss = (pos_loss + neg_loss) / (k + 1)
            total_loss = total_loss + loss
        
        return total_loss / num_edges