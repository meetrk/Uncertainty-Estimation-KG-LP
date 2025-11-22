import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.DataModel import LinkSplitter
from utils.utils import generate_batch_triples
from datetime import datetime
import numpy as np
from utils.utils import get_edges
from torch_geometric.loader import LinkNeighborLoader
from types import SimpleNamespace
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
        self.splitter = LinkSplitter(data, disjoint_train_ratio=0.4)
        


        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
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
            

            batch = self.splitter.generate_batch_triples(
                num_nodes=self.data.num_nodes,
                config=self.train_config,
                mode="train",
                sampling=self.train_config['sampling']['type'],
            )
            print(batch)
            loss,scores = self.train(
                batch=batch,
                all_triples=self.all_triples,
                entity_count=self.data.num_nodes,
                head_corrupt_prob=self.train_config['sampling']['head_corrupt_prob'],
                negative_sampling_ratio=self.train_config['sampling']['negative_sampling_ratio'],
            )
            

            self.training_history['train_loss'].append({"epoch": epoch, "epoch_loss": loss, "auc_score": scores['auc'], "precision": scores['precision'], "recall": scores['recall'], "f1": scores['f1']})

            
            # Log training loss to TensorBoard
            self.writer.add_scalar('Loss/Train', loss, epoch)
            self.writer.add_scalar('AUC SCORE/Train', scores['auc'], epoch)
            self.writer.add_scalar('Precision/Train', scores['precision'], epoch)
            self.writer.add_scalar('Recall/Train', scores['recall'], epoch)
            self.writer.add_scalar('F1 Score/Train', scores['f1'], epoch)

            # Log gradients periodically
            if epoch % 10 == 0:  # Log gradients every 10 epochs
                self.log_model_gradients(epoch)

            self.logger.info(f"Epoch {epoch} completed. Loss: {loss:.4f}")
            self.logger.info(f"Epoch {epoch} completed. AUC Score: {scores['auc']:.4f}")
            self.logger.info(f"Epoch {epoch} completed. Precision: {scores['precision']:.4f}")
            self.logger.info(f"Epoch {epoch} completed. Recall: {scores['recall']:.4f}")
            self.logger.info(f"Epoch {epoch} completed. F1 Score: {scores['f1']:.4f}")

            # Evaluation
            if epoch % eval_frequency == 0:

                batch = self.splitter.generate_batch_triples(
                    num_nodes=self.data.num_nodes,
                    config=self.train_config,
                    mode="train",
                    sampling=self.train_config['sampling']['type'],
                )

                mean_rank, mrr, hits_at_k = self.model.test(
                    batch=batch,
                    all_triples=self.all_triples,
                    batch_size=self.train_config['evaluation']['batch_size'],
                    k=self.train_config['evaluation']['hits_at_k'],
                )
                self.logger.info(f"Evaluation metrics at epoch {epoch}: 'Mean Rank': {mean_rank}, 'MRR': {mrr}, 'Hits@10': {hits_at_k}")

                self.writer.add_scalar("LP/MRR", float(mrr), epoch)
                self.writer.add_scalar("LP/Hits@10", float(hits_at_k), epoch)
                self.writer.add_scalar("LP/Mean_Rank", float(mean_rank), epoch)
                # self.training_history['eval_metrics'].append({"epoch": epoch, "metrics": {'Mean Rank': mean_rank, 'MRR': mrr, 'Hits@10': hits_at_k}, "eval_loss": eval_loss_value, "eval_auc_score": eval_auc_score})


                eval_loss_value, eval_scores = self.validation_loss()

                self.training_history['eval_metrics'].append({"epoch": epoch, "metrics": {'Mean Rank': mean_rank, 'MRR': mrr, 'Hits@10': hits_at_k}, "eval_loss": eval_loss_value, "eval_auc_score": eval_scores['auc'], "eval_precision": eval_scores['precision'], "eval_recall": eval_scores['recall'], "eval_f1": eval_scores['f1']})

                # Log evaluation loss to TensorBoard
                self.writer.add_scalar('Loss/Validation', eval_loss_value, epoch)
                self.writer.add_scalar('AUC SCORE/Validation', eval_scores['auc'], epoch)
                self.writer.add_scalar('Precision/Validation', eval_scores['precision'], epoch)
                self.writer.add_scalar('Recall/Validation', eval_scores['recall'], epoch)
                self.writer.add_scalar('F1 Score/Validation', eval_scores['f1'], epoch)
                
                self.logger.info(f"Evaluation Loss at epoch {epoch}: {eval_loss_value}")
                self.logger.info(f"Evaluation AUC Score at epoch {epoch}: {eval_scores['auc']}")
                self.logger.info(f"Evaluation Precision at epoch {epoch}: {eval_scores['precision']}")
                self.logger.info(f"Evaluation Recall at epoch {epoch}: {eval_scores['recall']}")
                self.logger.info(f"Evaluation F1 Score at epoch {epoch}: {eval_scores['f1']}")

            # Save checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)
        
        # Close TensorBoard writer
        self.writer.close()
        self.logger.info("Training completed!")
        return self.training_history



    def train(self, batch,all_triples, entity_count, head_corrupt_prob,negative_sampling_ratio):
        """
        Train the model on a single batch.
        """

        self.model.train()
        self.optimizer.zero_grad()
        loss, scores = self.model(batch, all_triples, entity_count, head_corrupt_prob,negative_sampling_ratio) # Forward pass
        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss, scores

    
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

    
    def validation_loss(self):
        """
        Evaluate average loss on a validation/test split produced by RandomLinkSplit.
        """
        self.model.eval()
        with torch.no_grad():

            batch = self.splitter.generate_batch_triples(
                num_nodes=self.data.num_nodes,
                config=self.train_config,
                mode="val",
                sampling=self.train_config['sampling']['type'],
            )

            val_loss,val_scores = self.model(batch,self.all_triples,self.data.num_nodes, head_corrupt_prob=self.train_config['sampling']['head_corrupt_prob'],negative_sampling_ratio=self.train_config['sampling']['negative_sampling_ratio'],)
            return val_loss,val_scores


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
