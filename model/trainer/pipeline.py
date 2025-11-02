import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
from torch_geometric.loader import LinkNeighborLoader, LinkLoader
from torch_geometric.sampler import NegativeSampling
import tqdm

class Pipeline:

    def __init__(self, model, data, config, logger):
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.learning_rate = self.config['optimiser']['learning_rate']
        self.weight_decay = self.config['optimiser']['weight_decay']
        self.device = next(model.parameters()).device

        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Training state
        self.epoch = 0
        self.training_history = {'train_loss': [], 'eval_metrics': []}

    def start_pipeline(self):
        max_epochs = self.config['epochs']
        eval_frequency = self.config.get('evaluation_frequency', 10)
        save_frequency = self.config.get('save_frequency', 20)
        
        loader = LinkNeighborLoader(
            self.data,
            num_neighbors=[30] * 2,
            batch_size=self.config['sampling']['batch_size'],
            edge_label_index=self.data.edge_index,
            neg_sampling=NegativeSampling(
                mode='binary',
                amount=self.config['sampling']['negative_sampling_ratio'],
            ),
            shuffle=True
        )

        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(1, max_epochs + 1):
            self.epoch = epoch
            
            # Training
            epoch_loss = 0.0
            num_batches = 0
            
            tqdm_loader = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
            
            for batch_idx, batch_data in enumerate(tqdm_loader):
                loss = self.train(batch_data)
                epoch_loss += loss
                num_batches += 1
                
                tqdm_loader.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = epoch_loss / num_batches
            self.training_history['train_loss'].append(avg_loss)
            
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Evaluation
            if epoch % eval_frequency == 0:
                eval_metrics = self.evaluate(self.data)
                self.training_history['eval_metrics'].append(eval_metrics)
                self.logger.info(f"Evaluation - Accuracy: {eval_metrics['accuracy']:.4f}, "
                               f"Loss: {eval_metrics['loss']:.4f}")
            
            # Save checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info("Training completed!")
        return self.training_history

    def train(self, batch_data):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        batch_data = batch_data.to(self.device)
        
        # Get entity indices for the batch
        # For knowledge graphs, we typically use all entities
        if hasattr(batch_data, 'n_id'):
            entity_indices = batch_data.n_id
        else:
            # Fallback: use all entities in the subgraph
            max_node = max(batch_data.edge_index.max(), batch_data.edge_label_index.max())
            entity_indices = torch.arange(max_node + 1, device=self.device)
        
        # Forward pass through encoder
        entity_embeddings = self.model(entity_indices, batch_data.edge_index, batch_data.edge_type)
        
        # Create triplets from edge_label_index
        if batch_data.edge_label_index.size(0) == 2:
            # Convert edge format to triplet format
            heads = batch_data.edge_label_index[0]
            tails = batch_data.edge_label_index[1]
            # Assume relation type 0 for simplicity, or extract from batch_data if available
            if hasattr(batch_data, 'edge_label_type'):
                relations = batch_data.edge_label_type
            else:
                relations = torch.zeros_like(heads)  # Default relation
            
            triplets = torch.stack([heads, relations, tails], dim=1)
        else:
            triplets = batch_data.edge_label_index.t()
        
        # Compute loss
        if hasattr(batch_data, 'edge_label'):
            targets = batch_data.edge_label.float()
        else:
            # Create binary targets: 1 for positive edges, 0 for negative
            targets = torch.ones(triplets.size(0), device=self.device)
        
        loss = self.model.score_loss(entity_embeddings, triplets, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, data):
        """
        Evaluate the model on given data.
        
        Args:
            data: PyG data object with evaluation edges
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Get all entity indices
            if hasattr(data, 'num_nodes'):
                num_nodes = data.num_nodes
            else:
                num_nodes = data.edge_index.max().item() + 1
            
            entity_indices = torch.arange(num_nodes, device=self.device)
            
            # Forward pass through the model
            entity_embeddings = self.model(entity_indices, data.edge_index, data.edge_type)
            
            # Use test triplets if available, otherwise use a subset of training data
            if hasattr(data, 'test_triplets') and data.test_triplets is not None:
                test_triplets = data.test_triplets[:1000]  # Limit for memory
                targets = torch.ones(len(test_triplets), device=self.device)
            elif hasattr(data, 'edge_label_index') and hasattr(data, 'edge_label'):
                # Use provided evaluation edges
                heads = data.edge_label_index[0]
                tails = data.edge_label_index[1]
                relations = torch.zeros_like(heads)  # Default relation
                test_triplets = torch.stack([heads, relations, tails], dim=1)
                targets = data.edge_label.float()
            else:
                # Fallback: create some test triplets from existing edges
                edge_sample = data.edge_index[:, :min(1000, data.edge_index.size(1))]
                test_triplets = torch.stack([
                    edge_sample[0], 
                    torch.zeros_like(edge_sample[0]), 
                    edge_sample[1]
                ], dim=1)
                targets = torch.ones(test_triplets.size(0), device=self.device)
            
            # Score the triplets using the decoder
            scores = self.model.decoder(entity_embeddings, test_triplets)
            
            # Convert scores to predictions (sigmoid + threshold)
            predictions = torch.sigmoid(scores) > 0.5
            
            # Calculate accuracy
            correct = (predictions.squeeze() == targets).float()
            accuracy = correct.mean().item()
            
            # Calculate AUC if possible
            try:
                from sklearn.metrics import roc_auc_score
                scores_np = torch.sigmoid(scores).cpu().numpy()
                targets_np = targets.cpu().numpy()
                auc = roc_auc_score(targets_np, scores_np)
            except (ImportError, ValueError):
                auc = None
            
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(scores.squeeze(), targets)
            
            metrics = {
                'accuracy': accuracy,
                'loss': loss.item(),
                'num_samples': len(targets)
            }
            
            if auc is not None:
                metrics['auc'] = auc
        
        return metrics
    
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