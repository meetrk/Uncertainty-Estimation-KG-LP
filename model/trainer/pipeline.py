from pyexpat import model
import torch
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
# from utils.utils import negative_sampling
from utils.utils import negative_sampling
from utils.evaluation import compute_mrr,compute_uncertainty
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
            # weight_decay=self.weight_decay
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
                    
                    self.logger.info(f"New best model found! MRR: {best_val_mrr:.4f} > Previous: {best_val_mrr:.4f}")
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

    def load_pipeline(self, checkpoint_path):

        self.load_checkpoint(checkpoint_path)
        self.logger.info("Pipeline loaded from checkpoint.")
        test_scores = self.test_uncertainty()
        test_mc_scores = self.test_uncertainty_mc(mc_samples=10)

        self.logger.info(f"Uncertainty Evaluation - Brier Score: {test_scores['brier_score']:.4f}")
        self.logger.info(f"Uncertainty Evaluation - Reliability Curve: {test_scores['reliability_curve']}")

        self.logger.info(f"MC Dropout Uncertainty Evaluation - Brier Score: {test_mc_scores['brier_score']:.4f}")
        self.logger.info(f"MC Dropout Uncertainty Evaluation - Reliability Curve: {test_mc_scores['reliability_curve']}")
        return test_scores,test_mc_scores

    @torch.no_grad()
    def inference_mc(self, mc_samples=10):

        self.model.eval()
        self.model.encoder.mc_dropout = True  
        neg_edge_index = negative_sampling(self.data.train_edge_index, self.data.num_nodes)

        preds_list = []

        for _ in range(mc_samples):
            z = self.model.encode(self.data.edge_index, self.data.edge_type)
            pos_out = self.model.decode(z, self.data.train_edge_index, self.data.train_edge_type)
            pos_out = torch.sigmoid(pos_out)
            neg_out = self.model.decode(z, neg_edge_index, self.data.train_edge_type)
            neg_out = torch.sigmoid(neg_out)
            self.logger.info(f"MC Sample {_+1}: Pos Out Mean: {pos_out.mean():.4f}, Neg Out Mean: {neg_out.mean():.4f}")
            self.logger.info(f"MC Sample {_+1}: Pos Out Std: {pos_out.std():.4f}, Neg Out Std: {neg_out.std():.4f}")
            out = torch.cat([pos_out, neg_out])
            preds_list.append(out)

        self.model.encoder.mc_dropout = False
        labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        preds_stack = torch.stack(preds_list)
        preds_mean = preds_stack.mean(dim=0)
        preds_std = preds_stack.std(dim=0)

        return preds_mean, preds_std, labels


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

        neg_edge_index = negative_sampling(self.data.train_edge_index, self.data.num_nodes)
        neg_out = self.model.decode(z, neg_edge_index, self.data.train_edge_type)

        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()

        return float(loss)

    @torch.no_grad()
    def validate(self):
        """
        Validate the model on the validation set.
        """
        self.model.eval()
        self.optimizer.zero_grad()

        z = self.model.encode(self.data.edge_index, self.data.edge_type)

        pos_out = self.model.decode(z, self.data.valid_edge_index, self.data.valid_edge_type)

        neg_edge_index = negative_sampling(self.data.valid_edge_index, self.data.num_nodes)
        neg_out = self.model.decode(z, neg_edge_index, self.data.valid_edge_type)

        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + self.model_config['decoder']['l2_penalty'] * reg_loss

        return float(loss)
    
    @torch.no_grad()
    def test(self, test = True):

        self.model.eval()
        z = self.model.encode(self.data.edge_index, self.data.edge_type)
        valid_scores = compute_mrr(z, self.data.valid_edge_index, self.data.valid_edge_type,self.data, self.model)
        if test:
            test_scores = compute_mrr(z, self.data.test_edge_index, self.data.test_edge_type,self.data, self.model)
            return valid_scores, test_scores

        return valid_scores, None

    @torch.no_grad()
    def test_uncertainty(self):

        self.model.eval()
        z = self.model.encode(self.data.edge_index, self.data.edge_type)
        pos_out = self.model.decode(z, self.data.valid_edge_index, self.data.valid_edge_type)
        pos_out = torch.sigmoid(pos_out)
        print(torch.mean(pos_out))
        neg_edge_index = negative_sampling(self.data.valid_edge_index, self.data.num_nodes)
        neg_out = self.model.decode(z, neg_edge_index, self.data.valid_edge_type)
        neg_out = torch.sigmoid(neg_out)
        print(torch.mean(neg_out))
        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        valid_scores = compute_uncertainty(gt, out)

        return valid_scores

    @torch.no_grad()
    def test_uncertainty_mc(self, mc_samples=10):

        self.model.eval()
        mean_pred, var_pred, labels = self.inference_mc(mc_samples=mc_samples)
        # self.logger.info(f"Mean Prediction Stats - Mean: {mean_pred.mean():.4f}, Std: {mean_pred.std():.4f}")
        val_scores = compute_uncertainty(labels, mean_pred)

        return val_scores



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
            # 'batch_size': self.train_config['sampling']['batch_size'],
            'negative_sampling_ratio': self.train_config['sampling']['negative_sampling_ratio'],
            'embedding_dim': self.model_config['encoder']['embedding_dim'],
            'hidden_layer_size': self.model_config['encoder']['hidden_layer_size'],
            'num_bases': self.model_config['encoder']['num_bases'],
            'b_init': self.model_config['decoder']['b_init'],
            'w_gain': self.model_config['decoder']['w_gain'],
            # 'sampling_method': self.train_config['sampling']['method']
        }
        
        # Add text summary of hyperparameters
        hparam_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
        self.writer.add_text('Hyperparameters', hparam_text, 0)
        
        # Log as scalars for easy comparison
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Hyperparameters/{key}', value, 0)

    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f'{self.config.get_section("dataset")["name"]}_checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {'train_loss': [], 'eval_metrics': []})
        self.epoch = checkpoint['epoch']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    def __del__(self):
        """Cleanup TensorBoard writer when pipeline is destroyed."""
        if hasattr(self, 'writer'):
            self.writer.close()
