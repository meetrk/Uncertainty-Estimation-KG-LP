from pyexpat import model
import torch
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
from utils.utils import negative_sampling
from utils.evaluation import compute_mrr,compute_uncertainty,compute_mrr_mc_dropout
from utils.utils import dropout_edges

class Pipeline:

    def __init__(self, model, data, config, logger):
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.model_config = self.config.get_section('model')
        self.train_config = self.config.get_section('training')
        self.learning_rate = self.train_config['optimiser']['learning_rate']
        self.weight_decay = self.train_config['optimiser']['weight_decay']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate,
        )
        self.all_triples = torch.stack([
                    self.data.edge_index[0],
                    self.data.edge_type,
                    self.data.edge_index[1]
                ], dim=1)
        
        # Initialize TensorBoard writer
        log_dir = Path('runs') / f"experiment_{self.config.get_section('dataset')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")
        
        # Log hyperparameters
        self.log_hyperparameters()
        
        # Training state
        self.epoch = 0
        self.training_history = {'train_loss': [], 'val_loss': [], 'eval_metrics': []}

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

        for epoch in tqdm_range:
           
            loss = self.train()
            # val_loss = self.validate()
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

        
        # Close TensorBoard writer
        self.writer.close()
        self.logger.info("Training completed!")
        return self.training_history

    def load_pipeline(self, checkpoint_path, method,uncertainty_samples=5):


        self.load_checkpoint(checkpoint_path)
        self.logger.info("Pipeline loaded from checkpoint.")
        
        scores = self.test_link_pred(
            method=method,
            model=self.model,
            valid_edge_index=self.data.valid_edge_index,
            valid_edge_type=self.data.valid_edge_type,
            mc_samples=uncertainty_samples)

        scores = self.test_uncertainty(
            self.model,self.data.edge_index,
            self.data.edge_type,
            self.data.valid_edge_index,
            self.data.valid_edge_type)

        calibration_results = self.calibrate_pipeline(
            method=self.config.get_section('calibration')['method'],
            model=self.model,
            max_iters=self.config.get_section('calibration').get('max_iters', 50),
            lr=self.config.get_section('calibration').get('learning_rate', 0.01)
        )

        scores = self.test_link_pred(
            method=method,
            model=self.model,
            valid_edge_index=self.data.valid_edge_index,
            valid_edge_type=self.data.valid_edge_type,
            mc_samples=uncertainty_samples)


        scores = self.test_uncertainty(self.model,self.data.edge_index, self.data.edge_type, self.data.test_edge_index, self.data.test_edge_type)
        self.logger.info(f"Brier_score: {scores['brier_score']}")
        self.logger.info(f"ECE: {scores['ece']}")
        self.logger.info(f"Prob_true: {scores['prob_true']}")
        self.logger.info(f"Prob_pred: {scores['prob_pred']}")

        self.save_checkpoint(self.epoch, name=f'calibrated_{Path(checkpoint_path).name}')
        

    @torch.no_grad()
    def inference_mc(self, edge_index, edge_type, mc_samples=10):

        self.model.eval()
        if mc_samples > 1:
            self.model.encoder.mc_dropout = True  

        neg_edge_index,neg_edge_type = negative_sampling(edge_index, edge_type, self.data.num_nodes,1)
      
        preds_list = []

        for _ in range(mc_samples):
            print("Inference MC Samples:", mc_samples, self.model.encoder.mc_dropout)
            z = self.model.encode(self.data.edge_index, self.data.edge_type)
            pos_out = self.model.decode(z, edge_index, edge_type)
            pos_out = torch.sigmoid(pos_out)

            neg_out = self.model.decode(z, neg_edge_index, neg_edge_type)
            neg_out = torch.sigmoid(neg_out)

            out = torch.cat([pos_out, neg_out])
            preds_list.append(out)

        self.model.encoder.mc_dropout = False
        labels = torch.cat([(torch.ones_like(pos_out)), (torch.zeros_like(neg_out))])
        preds_stack = torch.stack(preds_list)
        preds_mean = preds_stack.mean(dim=0)

        means = {
            'preds_mean': preds_mean,
            'labels': labels    
        }

        return means

    @torch.no_grad()
    def inference(self, model, edge_index, edge_type, test_edge_index, test_edge_type):

        model.eval()
        z = model.encoder(edge_index, edge_type)
        pos_scores = model.decode(z, test_edge_index, test_edge_type)
        neg_edge_index, neg_edge_type = negative_sampling(test_edge_index, test_edge_type, self.data.num_nodes,1)
        neg_scores = model.decode(z, neg_edge_index, neg_edge_type)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        scores = torch.sigmoid(scores)

        return scores, labels

    @torch.no_grad()
    def test_link_pred(self, method, model, valid_edge_index, valid_edge_type,  mc_samples=10):
        model.eval()
        if method == 'standard':
            
            scores = compute_mrr(valid_edge_index, valid_edge_type ,self.data, model)   
            self.logger.info(f"MRR: {scores['mrr']:.4f}")
            self.logger.info(f"Mean Rank: {scores['mean_rank']:.4f}")
            self.logger.info(f"Hits@1: {scores['hits@1']:.4f}")
            self.logger.info(f"Hits@3: {scores['hits@3']:.4f}")
            self.logger.info(f"Hits@10: {scores['hits@10']:.4f}")

        elif method == 'mc_dropout':
            
            scores = compute_mrr_mc_dropout(self.data.edge_index, self.data.edge_type,
                                valid_edge_index, valid_edge_type,
                                self.data, model, mc_samples=mc_samples)
            
            self.logger.info(f"MRR = {scores['mrr']:.4f}")
            self.logger.info(f"Mean Rank = {scores['mean_rank']:.4f}")
            self.logger.info(f"Hits@1 = {scores['hits@1']:.4f}")
            self.logger.info(f"Hits@3 = {scores['hits@3']:.4f} ")
            self.logger.info(f"Hits@10 = {scores['hits@10']:.4f}")

        else:
            raise ValueError(f"Unsupported evaluation method: {method}")
        
        return scores

    def train(self):
        """
        Train the model on a single batch.
        """

        self.model.train()
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
        gt = torch.cat([torch.ones_like(pos_out) - 0.1, torch.zeros_like(neg_out) + 0.05])
        
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
    def test_uncertainty(self, model, edge_index, edge_type, test_edge_index, test_edge_type):

        scores, labels = self.inference(model, edge_index, edge_type, test_edge_index, test_edge_type)
        val_scores = compute_uncertainty(labels,scores)

        self.logger.info(f"Brier_score: {val_scores['brier_score']}")
        self.logger.info(f"ECE: {val_scores['ece']}")
        self.logger.info(f"Prob_true: {val_scores['prob_true']}")
        self.logger.info(f"Prob_pred: {val_scores['prob_pred']}")

        self.logger.info(f" {val_scores}")

        return val_scores

    def calibrate_pipeline(self, method, model, max_iters=50, lr=0.01):
        """Main entry point for calibration."""

        self.logger.info("Starting calibration process...")
        
        if method == 'scalar':
            return self.calibrate_scalar_temperature(model, max_iters, lr)
        elif method == 'input_scaling':
            return self.calibrate_input_dependent_temperature(model, max_iters, lr)
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
    
    def _log_temperature_stats(self, stats, prefix="", level="info"):
        """Log temperature statistics in a consistent format."""
        msg = f"{prefix}Temperature - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f}"
        if level == "info":
            self.logger.info(msg)
        else:
            self.logger.debug(msg)
    
    def _freeze_non_temperature_params(self, model, param_filter):
        """Freeze all parameters except those matching the filter.
        
        Args:
            model: The model to freeze parameters in
            param_filter: Function that returns True if parameter should be trainable
        
        Returns:
            list: Trainable parameters
        """
        trainable_params = []
        for name, param in model.named_parameters():
            if param_filter(name):
                param.requires_grad = True
                trainable_params.append(param)
                self.logger.info(f"Trainable: {name}, Shape: {param.shape}")
            else:
                param.requires_grad = False
        
        if not trainable_params:
            self.logger.warning("No trainable parameters found for calibration!")
        
        return trainable_params

    def compute_nll_loss(self,model):
        """
        Compute Negative Log-Likelihood loss for calibration.
        
        Args:
            model: The GAE model
            data: Graph data
            edge_index: Edge indices to evaluate
            edge_type: Edge types
            
        Returns:
            NLL loss value
        """
        model.eval()
        with torch.no_grad():
            z = model.encode(self.data.edge_index, self.data.edge_type)
        
        # Get positive logits (temperature is applied inside decoder)
        pos_logits = model.decode(z, self.data.valid_edge_index, self.data.valid_edge_type)
        
        # Sample negative edges for balanced calibration
        neg_edge_index, neg_edge_type = negative_sampling(
            self.data.valid_edge_index, 
            self.data.valid_edge_type, 
            self.data.num_nodes, 
            1  
        )
        neg_logits = model.decode(z, neg_edge_index, neg_edge_type)
        
        # Concatenate positive and negative samples
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        
        nll_loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return nll_loss

    def calibrate_input_dependent_temperature(self, model, max_iters=50, lr=0.01):
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
            loss = self.compute_nll_loss(model)
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

    def calibrate_scalar_temperature(self, model, max_iters=50, lr=0.01):
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
            loss = self.compute_nll_loss(model)
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


    def log_model_gradients(self, epoch):
        """Log gradient norms to TensorBoard for monitoring."""
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                # Log individual parameter gradients
                self.writer.add_scalar(f'Gradients/{name}', param_norm, epoch)
        
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Gradients/Total_Norm', total_norm, epoch)

    def log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        hparams = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.train_config['epochs'],
            'negative_sampling_ratio': self.train_config['sampling']['negative_sampling_ratio'],
            'embedding_dim': self.model_config['encoder']['embedding_dim'],
            'hidden_layer_size': self.model_config['encoder']['hidden_layer_size'],
            'num_bases': self.model_config['encoder']['num_bases'],
            # 'sampling_method': self.train_config['sampling']['method']
        }
        
        # Add text summary of hyperparameters
        hparam_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
        self.writer.add_text('Hyperparameters', hparam_text, 0)
        
        # Log as scalars for easy comparison
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Hyperparameters/{key}', value, 0)


    def save_checkpoint(self, epoch, name = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        if name is None:
            name = f'{self.config.get_section("dataset")["name"]}_checkpoint_epoch_{epoch}.pth'
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / name
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {'train_loss': [], 'eval_metrics': []})
        self.epoch = checkpoint['epoch']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
    def __del__(self):
        """Cleanup TensorBoard writer when pipeline is destroyed."""
        if hasattr(self, 'writer'):
            self.writer.close()
