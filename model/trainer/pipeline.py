import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_triples,generate_batch_triples
from datetime import datetime

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
            positives, negatives, batch_idx = generate_batch_triples(self.data.train_triplets, self.data.num_nodes, self.train_config, self.device)

            loss = self.train(
                edge_label_index=batch_idx[:, :2].T,
                edge_label_type=batch_idx[:, 1],
                edge_label=torch.cat([torch.ones(positives.size(0), device=self.device), 
                                      torch.zeros(negatives.size(0), device=self.device)])
            )
            epoch_loss += loss.item()
            self.training_history['train_loss'].append(epoch_loss)
            
            # Log training loss to TensorBoard
            self.writer.add_scalar('Loss/Train', epoch_loss, epoch)
            
            # Log gradients periodically
            if epoch % 10 == 0:  # Log gradients every 10 epochs
                self.log_model_gradients(epoch)
            
            self.logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")

            # Evaluation
            if epoch % eval_frequency == 0:
                eval_metrics = self.evaluate_loss()
                self.training_history['eval_metrics'].append(eval_metrics)
                
                # Log evaluation loss to TensorBoard
                eval_loss_value = eval_metrics.item() if isinstance(eval_metrics, torch.Tensor) else eval_metrics
                self.writer.add_scalar('Loss/Validation', eval_loss_value, epoch)
                
                self.logger.info(f"Evaluation metrics at epoch {epoch}: {eval_metrics}")
            
            # Save checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)
        
        # Close TensorBoard writer
        self.writer.close()
        self.logger.info("Training completed!")
        return self.training_history



    def train(self, edge_label_index, edge_label_type, edge_label):
        """
        Train the model on a single batch.
        """

        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        edge_label_index = edge_label_index.to(self.device)
        edge_label_type = edge_label_type.to(self.device)
        edge_label = edge_label.to(self.device)

        entities = torch.arange(self.data.num_nodes, device=self.device)
        entity_embeddings = self.model(entities, edge_label_index, edge_label_type) ## generating embedding only for nodes in the batch

        triplets = get_triples(edge_label_index, edge_label_type)
        # Compute loss
        loss = self.model.score_loss(entity_embeddings, triplets, edge_label.float()) ## calculate the loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        
        return loss

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
            'w_init': self.model_config['decoder']['w_init'],
            'w_gain': self.model_config['decoder']['w_gain']
        }
        
        # Add text summary of hyperparameters
        hparam_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
        self.writer.add_text('Hyperparameters', hparam_text, 0)
        
        # Log as scalars for easy comparison
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Hyperparameters/{key}', value, 0)

    
    def evaluate_loss(self):
        """
        Evaluate average loss on a validation/test split produced by RandomLinkSplit.
        """
        self.model.eval()
        with torch.no_grad():
            entities = torch.arange(self.data.num_nodes, device=self.device)
            entity_embedding = self.model(entities, self.data.val_graph.edge_index, self.data.val_graph.edge_type)
            positives, negatives, val_triplets = generate_batch_triples(self.data.valid_triplets, self.data.num_nodes, self.train_config, self.device)
            val_edge_labels=torch.cat([torch.ones(positives.size(0), device=self.device), 
                                      torch.zeros(negatives.size(0), device=self.device)])
            val_loss = self.model.score_loss(entity_embedding, val_triplets, val_edge_labels)
            
            return val_loss
        



    # def plot_training_history(self):
    #     """Plot training loss and evaluation loss over epochs."""
    #     import matplotlib.pyplot as plt

    #     epochs = range(1, len(self.training_history['train_loss']) + 1)
        
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(epochs, self.training_history['train_loss'], label='Training Loss', marker='o', alpha=0.7)
        
    #     # Plot evaluation metrics on the same graph
    #     if self.training_history['eval_metrics']:
    #         eval_frequency = self.train_config.get('evaluation_frequency', 10)
    #         eval_loss_values = [metrics.item() if isinstance(metrics, torch.Tensor) else metrics 
    #                            for metrics in self.training_history['eval_metrics']]
    #         eval_epochs = [i * eval_frequency for i in range(1, len(eval_loss_values) + 1)]
            
    #         plt.plot(eval_epochs, eval_loss_values, label='Validation Loss', marker='s', alpha=0.7)
        
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss over Epochs')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()
    
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

