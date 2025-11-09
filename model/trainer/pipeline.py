import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import generate_batch_triples
from datetime import datetime
import numpy as np
from utils.utils import get_edges

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
        self.device = next(model.parameters()).device


        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Initialize TensorBoard writer
        log_dir = Path('runs') / f"experiment_{self.config.get_section('dataset')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")
        
        # Log hyperparameters
        self.log_hyperparameters()
        
        # Training state
        self.epoch = 0
        self.training_history = {'train_loss': [], 'eval_metrics': []}

    def start_pipeline(self):
        max_epochs = self.train_config['epochs']
        eval_frequency = self.train_config.get('evaluation_frequency', 10)
        save_frequency = self.train_config.get('save_frequency', 20)
        
    

        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        tqdm_range = range(1, max_epochs + 1)
        tqdm_range = tqdm(tqdm_range, desc="Training", unit="batch")        

        for epoch in tqdm_range:
            self.epoch = epoch
            
            # Training
            epoch_loss = 0.0

            triple_batch = generate_batch_triples(self.data.train_triplets, self.data.num_nodes, self.train_config, mode="train", sampling=self.train_config['sampling']['method'])
        
            loss, auc_score = self.train(
                triples=triple_batch
            )
            epoch_loss += loss.item()

            self.training_history['train_loss'].append({"epoch": epoch, "epoch_loss": epoch_loss, "auc_score": auc_score})

            
            # Log training loss to TensorBoard
            self.writer.add_scalar('Loss/Train', epoch_loss, epoch)
            self.writer.add_scalar('AUC SCORE/Train', auc_score, epoch)
            
            # Log gradients periodically
            if epoch % 10 == 0:  # Log gradients every 10 epochs
                self.log_model_gradients(epoch)

            self.logger.info(f"Epoch {epoch} completed. Loss: {epoch_loss:.4f}")
            self.logger.info(f"Epoch {epoch} completed. AUC Score: {auc_score:.4f}")

            # Evaluation
            if epoch % eval_frequency == 0:

                mean_rank, mrr, hits_at_k = self.model.test(
                    head_index=self.data.train_triplets[:,0],
                    rel_type=self.data.train_triplets[:,1],
                    tail_index=self.data.train_triplets[:,2],
                    batch_size=256,
                    k=10
                )
                self.logger.info(f"Evaluation metrics at epoch {epoch}: 'Mean Rank': {mean_rank}, 'MRR': {mrr}, 'Hits@10': {hits_at_k}")

                self.writer.add_scalar("LP/MRR", float(mrr), epoch)
                self.writer.add_scalar("LP/Hits@10", float(hits_at_k), epoch)
                self.writer.add_scalar("LP/Mean_Rank", float(mean_rank), epoch)
                # self.training_history['eval_metrics'].append({"epoch": epoch, "metrics": {'Mean Rank': mean_rank, 'MRR': mrr, 'Hits@10': hits_at_k}, "eval_loss": eval_loss_value, "eval_auc_score": eval_auc_score})


                eval_loss_value, eval_auc_score = self.validation_loss()

                self.training_history['eval_metrics'].append({"epoch": epoch, "metrics": {'Mean Rank': mean_rank, 'MRR': mrr, 'Hits@10': hits_at_k}, "eval_loss": eval_loss_value, "eval_auc_score": eval_auc_score})
                
                # Log evaluation loss to TensorBoard
                self.writer.add_scalar('Loss/Validation', eval_loss_value, epoch)
                self.writer.add_scalar('AUC SCORE/Validation', eval_auc_score, epoch)
                self.logger.info(f"Evaluation Loss at epoch {epoch}: {eval_loss_value}")
                self.logger.info(f"Evaluation AUC Score at epoch {epoch}: {eval_auc_score}")
                
            
            # Save checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)
        
        # Close TensorBoard writer
        self.writer.close()
        self.logger.info("Training completed!")
        return self.training_history



    def train(self, triples):
        """
        Train the model on a single batch.
        """

        self.model.train()
        self.optimizer.zero_grad()

        edge_label_index, edge_label_type = get_edges(triples)
       
        # Move data to device
        edge_label_index = edge_label_index.to(self.device)
        edge_label_type = edge_label_type.to(self.device)


        pred_logits, loss, roc_auc_score = self.model(edge_label_index, edge_label_type) 

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss, roc_auc_score

    
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
            'batch_size': self.train_config['sampling']['batch_size'],
            'negative_sampling_ratio': self.train_config['sampling']['negative_sampling_ratio'],
            'embedding_dim': self.model_config['encoder']['embedding_dim'],
            'hidden_layer_size': self.model_config['encoder']['hidden_layer_size'],
            'num_bases': self.model_config['encoder']['num_bases'],
            'b_init': self.model_config['decoder']['b_init'],
            'w_gain': self.model_config['decoder']['w_gain'],
            'sampling_method': self.train_config['sampling']['method']
        }
        
        # Add text summary of hyperparameters
        hparam_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
        self.writer.add_text('Hyperparameters', hparam_text, 0)
        
        # Log as scalars for easy comparison
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Hyperparameters/{key}', value, 0)

    
    def validation_loss(self):
        """
        Evaluate average loss on a validation/test split produced by RandomLinkSplit.
        """
        self.model.eval()
        with torch.no_grad():

            val_triplets = generate_batch_triples(self.data.valid_triplets, self.data.num_nodes, self.train_config,mode="eval", sampling=self.train_config['sampling']['method'])

            val_edge_index,val_edge_labels = get_edges(triplets=val_triplets)

            score,val_loss,val_roc_auc_score = self.model(val_edge_index, val_edge_labels)

            return val_loss,val_roc_auc_score
        
    
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
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {'train_loss': [], 'eval_metrics': []})
        self.epoch = checkpoint['epoch']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def __del__(self):
        """Cleanup TensorBoard writer when pipeline is destroyed."""
        if hasattr(self, 'writer'):
            self.writer.close()
