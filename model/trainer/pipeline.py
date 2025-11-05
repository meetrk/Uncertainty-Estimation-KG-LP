import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import generate_batch_triples
import torch.nn.functional as F 
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
            positives, negatives, batch_idx = generate_batch_triples(self.data.train_triplets, self.data.num_nodes, self.train_config, self.device, sampling=self.train_config['sampling']['method'])

            loss = self.train(
                edge_label_index=batch_idx[:, :2].T,
                edge_label_type=batch_idx[:, 1],
                edge_label= torch.cat([
                    torch.ones(positives.size(0), dtype=torch.float, device=self.device), 
                    torch.zeros(negatives.size(0), dtype=torch.float, device=self.device)
                ], dim=0)
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
    
                # metrics = evaluate_distmult(
                #     triples=self.data.valid_triplets,
                #     entity_embeddings=self.model.entity_embedding.detach().cpu().numpy(),
                #     relation_embeddings=self.model.decoder.relations_embedding.detach().cpu().numpy(),
                #     known_triples= torch.cat([self.data.train_triplets, self.data.valid_triplets, self.data.test_triplets],dim=0),
                #     ks=[1,3,10]
                # )
                # self.writer.add_scalar("LP/MRR", float(metrics["MRR"]), epoch)
                # self.writer.add_scalar("LP/Hits@1", float(metrics["Hits@K"][1]), epoch)
                # self.writer.add_scalar("LP/Hits@3", float(metrics["Hits@K"][3]), epoch)
                # self.writer.add_scalar("LP/Hits@10", float(metrics["Hits@K"][10]), epoch)
                # self.logger.info(f"Evaluation metrics at epoch {epoch}: {metrics}")

                
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

        score, penalty = self.model(edge_label_index, edge_label_type,self.model_config) ## generating embedding only for nodes in the batch

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(score, edge_label)
        loss  = loss + (self.model_config['decoder']['l2_penalty'] * penalty)

        # Backward pass
        loss.backward()

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

    
    def evaluate_loss(self):
        """
        Evaluate average loss on a validation/test split produced by RandomLinkSplit.
        """
        self.model.eval()
        with torch.no_grad():

            positives, negatives, val_triplets = generate_batch_triples(self.data.valid_triplets, self.data.num_nodes, self.train_config, self.device, sampling=self.train_config['sampling']['method'])

            val_edge_index,val_edge_labels = get_edges(triplets=val_triplets)

            score, _ = self.model(val_edge_index, val_edge_labels,self.model_config)

            val_edge_labels=torch.cat([torch.ones(positives.size(0), device=self.device), 
                                      torch.zeros(negatives.size(0), device=self.device)])
            # Compute loss
            val_loss = F.binary_cross_entropy_with_logits(score, val_edge_labels)

            return val_loss
        
    
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

    

import numpy as np
from collections import defaultdict
from contextlib import nullcontext
from tqdm import tqdm

def evaluate_distmult(
    triples,
    entity_embeddings,
    relation_embeddings,
    known_triples=None,
    ks=(1, 3, 10),
    filtered_index=None,
):
    """
    Filtered MRR and Hits@K for link prediction with DistMult.

    Parameters
    ----------
    triples : iterable[tuple[int,int,int]]
    entity_embeddings : (num_entities, d) array-like (NumPy recommended)
    relation_embeddings : (num_relations, d) array-like
    known_triples : set[(h,r,t)] | None
    ks : iterable[int]
    filtered_index : optional dict with two maps to speed filtering:
        {
          'hr_to_t': dict[(h,r)] -> np.ndarray of true tails (including the eval t),
          'rt_to_h': dict[(r,t)] -> np.ndarray of true heads (including the eval h),
        }
        If not provided but known_triples is, it will be built.

    Returns
    -------
    {"MRR": float, "Hits@K": {K: float}}
    """
    # Ensure NumPy arrays
    E = np.asarray(entity_embeddings, dtype=np.float32)
    R = np.asarray(relation_embeddings, dtype=np.float32)
    N, d = E.shape
    assert R.shape[1] == d, "Entity and relation embeddings must share dimension d"

    # Precompute ET for fast tail scoring
    ET = E.T  # (d, N)

    # Build filtered index if needed
    if known_triples is not None and filtered_index is None:
        hr_to_t = defaultdict(list)
        rt_to_h = defaultdict(list)
        for (hh, rr, tt) in known_triples:
            hr_to_t[(hh, rr)].append(tt)
            rt_to_h[(rr, tt)].append(hh)
        # Convert lists to arrays for fast masking
        filtered_index = {
            "hr_to_t": {k: np.array(v, dtype=np.int64) for k, v in hr_to_t.items()},
            "rt_to_h": {k: np.array(v, dtype=np.int64) for k, v in rt_to_h.items()},
        }

    ks = tuple(int(k) for k in ks)
    rr_sum = 0.0
    hits_counts = {K: 0.0 for K in ks}

    # Iterate triples
    for (h, r, t) in tqdm(triples, desc="Evaluating", unit="triple"):
        # Tail ranking (h, r, ?): scores for all t'
        # DistMult => (E[h] * R[r]) @ E.T
        hr = E[h] * R[r]                    # (d,)
        scores_tail = hr @ ET               # (N,)

        # Filter other true tails for (h,r)
        if known_triples is not None:
            idx = filtered_index["hr_to_t"].get((h, r), None)
            if idx is not None and idx.size > 0:
                # mask out all true tails except the target t
                # set to -inf so they don't affect ranking
                mask_idx = idx[idx != t]
                if mask_idx.size > 0:
                    scores_tail[mask_idx] = -np.inf

        # Rank of true t (higher is better); use competition ranking
        # Rank = 1 + count of strictly greater scores
        st = scores_tail[t]
        # Protect against NaN/inf
        if not np.isfinite(st):
            st = -np.inf
        rank_t = int(np.sum(scores_tail > st)) + 1

        # Head ranking (?, r, t): scores for all h'
        rt = R[r] * E[t]                    # (d,)
        scores_head = E @ rt                # (N,)

        if known_triples is not None:
            idx = filtered_index["rt_to_h"].get((r, t), None)
            if idx is not None and idx.size > 0:
                mask_idx = idx[idx != h]
                if mask_idx.size > 0:
                    scores_head[mask_idx] = -np.inf

        sh = scores_head[h]
        if not np.isfinite(sh):
            sh = -np.inf
        rank_h = int(np.sum(scores_head > sh)) + 1

        rr = 0.5 * (1.0 / rank_t + 1.0 / rank_h)
        rr_sum += rr

        for K in ks:
            hits_counts[K] += 0.5 * ((rank_t <= K) + (rank_h <= K))

    n = len(triples) if hasattr(triples, "__len__") else int(getattr(triples, "shape", [0])[0])
    if n == 0:
        return {"MRR": 0.0, "Hits@K": {K: 0.0 for K in ks}}

    mrr = rr_sum / n
    hits_at_k = {K: hits_counts[K] / n for K in ks}
    return {"MRR": float(mrr), "Hits@K": {K: float(v) for K, v in hits_at_k.items()}}
