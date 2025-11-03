import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.loader import NeighborSampler, LinkLoader
from torch_geometric.sampler import NegativeSampling
from random import sample
from utils.evaluation import mean_reciprocal_rank, hits_at_k
from utils.utils import get_triples,negative_sampling
from tqdm import tqdm

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
        
    

        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        tqdm_range = range(1, max_epochs + 1)
        tqdm_range = tqdm(tqdm_range, desc="Evaluating", unit="batch")

        for epoch in tqdm_range:
            self.epoch = epoch
            
            # Training
            epoch_loss = 0.0
            positives = sample(range(self.data.train_triplets.size(0)), k=self.config['sampling']['batch_size'])
            positives = self.data.train_triplets[positives].to(self.device)
            negatives = positives.clone()[:, None, :].expand(self.config['sampling']['batch_size'], self.config['sampling']['negative_sampling_ratio'], 3).contiguous()
            negatives = negative_sampling(negatives, self.data.num_nodes, self.config['sampling']['head_corrupt_prob'], device=self.device)
            batch_idx = torch.cat([positives, negatives], dim=0)

            loss = self.train(
                edge_label_index=batch_idx[:, :2].T,
                edge_label_type=batch_idx[:, 1],
                edge_label=torch.cat([torch.ones(positives.size(0), device=self.device), 
                                      torch.zeros(negatives.size(0), device=self.device)])
            )
            epoch_loss += loss.item()
            self.training_history['train_loss'].append(epoch_loss)
            self.logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")

            # Evaluation
            if epoch % eval_frequency == 0:
                eval_metrics = self.evaluate(self.data)
                self.training_history['eval_metrics'].append(eval_metrics)
                self.logger.info(f"Evaluation metrics at epoch {epoch}: {eval_metrics}")
            
            # Save checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)
        
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
            self.logger.info("Starting evaluation...")
            
            # Use test triplets for evaluation
            test_triplets = data.test_triplets.to(self.device)
            all_triplets = data.all_triplets.to(self.device)
            
            # Get all entities and generate embeddings using the full graph
            entities = torch.arange(data.num_nodes, device=self.device)
            entity_embeddings = self.model(entities, data.edge_index.to(self.device), data.edge_type.to(self.device))
            
            # Evaluate in batches to avoid memory issues
            batch_size = self.config['evaluation']['batch_size']
            num_test = test_triplets.size(0)
            all_ranks = []

            tqdm_range = range(0, num_test, batch_size)
            if self.config.get('evaluation', {}).get('verbose', False):
                
                tqdm_range = tqdm(tqdm_range, desc="Evaluating", unit="batch")

            for i in tqdm_range:
                batch_end = min(i + batch_size, num_test)
                batch_triplets = test_triplets[i:batch_end]
                
                # For each test triplet, compute ranks for head and tail prediction
                for triplet in batch_triplets:
                    h, r, t = triplet[0].item(), triplet[1].item(), triplet[2].item()
                    
                    # Head prediction: corrupt head, keep relation and tail fixed
                    head_scores = []
                    for candidate_h in range(data.num_nodes):
                        candidate_triplet = torch.tensor([[candidate_h, r, t]], device=self.device)
                        score = self.model.decoder(entity_embeddings, candidate_triplet)
                        head_scores.append(score.item())
                    
                    head_scores = torch.tensor(head_scores, device=self.device)
                    
                    # Filter out known true triplets (except the target)
                    for known_triplet in all_triplets:
                        kh, kr, kt = known_triplet[0].item(), known_triplet[1].item(), known_triplet[2].item()
                        if kr == r and kt == t and kh != h:
                            head_scores[kh] = float('-inf')
                    
                    # Calculate rank for head prediction
                    target_score = head_scores[h]
                    head_rank = (head_scores >= target_score).sum().item()
                    all_ranks.append(head_rank)
                    
                    # Tail prediction: corrupt tail, keep head and relation fixed
                    tail_scores = []
                    for candidate_t in range(data.num_nodes):
                        candidate_triplet = torch.tensor([[h, r, candidate_t]], device=self.device)
                        score = self.model.decoder(entity_embeddings, candidate_triplet)
                        tail_scores.append(score.item())
                    
                    tail_scores = torch.tensor(tail_scores, device=self.device)
                    
                    # Filter out known true triplets (except the target)
                    for known_triplet in all_triplets:
                        kh, kr, kt = known_triplet[0].item(), known_triplet[1].item(), known_triplet[2].item()
                        if kh == h and kr == r and kt != t:
                            tail_scores[kt] = float('-inf')
                    
                    # Calculate rank for tail prediction
                    target_score = tail_scores[t]
                    tail_rank = (tail_scores >= target_score).sum().item()
                    all_ranks.append(tail_rank)
            
            # Calculate metrics
            mrr = mean_reciprocal_rank(all_ranks)
            hits_1 = hits_at_k(all_ranks, 1)
            hits_3 = hits_at_k(all_ranks, 3)
            hits_10 = hits_at_k(all_ranks, 10)
            
            self.logger.info(f"Evaluation completed. MRR: {mrr:.4f}, Hits@1: {hits_1:.4f}, Hits@3: {hits_3:.4f}, Hits@10: {hits_10:.4f}")

        return {
            'MRR': mrr,
            'Hits@1': hits_1,
            'Hits@3': hits_3,
            'Hits@10': hits_10
        }
    
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