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

            triple_batch = generate_batch_triples(self.data.train_triplets, self.data.num_nodes, self.train_config, self.device, mode="train", sampling=self.train_config['sampling']['method'])
        
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
    
                metrics = evaluate_mrr_hits(
                    entity_emb=self.model.entity_embedding.detach().cpu().numpy(),
                    relation_emb=self.model.decoder.rel_emb.weight.detach().cpu().numpy(),
                    test_triples=self.data.valid_triplets,
                    known_triples= self.data.train_triplets
                )
                self.writer.add_scalar("LP/MRR", float(metrics["mrr"]), epoch)
                self.writer.add_scalar("LP/Hits@1", float(metrics["hits@1"]), epoch)
                self.writer.add_scalar("LP/Hits@3", float(metrics["hits@3"]), epoch)
                self.writer.add_scalar("LP/Hits@10", float(metrics["hits@10"]), epoch)
                self.logger.info(f"Evaluation metrics at epoch {epoch}: {metrics}")

                
                eval_loss_value,eval_auc_score = self.evaluate_loss()

                self.training_history['eval_metrics'].append({"epoch":epoch,"metrics":metrics,"eval_loss":eval_loss_value,"eval_auc_score":eval_auc_score})
                
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

    
    def evaluate_loss(self):
        """
        Evaluate average loss on a validation/test split produced by RandomLinkSplit.
        """
        self.model.eval()
        with torch.no_grad():

            val_triplets = generate_batch_triples(self.data.valid_triplets, self.data.num_nodes, self.train_config, self.device,mode="eval", sampling=self.train_config['sampling']['method'])

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


import numpy as np
import torch
from typing import Iterable, Tuple, Set, List, Dict, Union
from tqdm import tqdm

Triple = Tuple[int, int, int]
ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_triple_list(triples: Union[ArrayLike, Iterable[Triple]]) -> List[Triple]:
    if isinstance(triples, torch.Tensor):
        triples = triples.detach().cpu().numpy()
    if isinstance(triples, np.ndarray):
        return [tuple(map(int, t)) for t in triples]
    return [tuple(map(int, t)) for t in triples]


def _distmult_scores_replace_tail(entity_emb: np.ndarray, relation_emb: np.ndarray,
                                   s: int, p: int) -> np.ndarray:
    v = entity_emb[s] * relation_emb[p]
    return entity_emb @ v


def _distmult_scores_replace_head(entity_emb: np.ndarray, relation_emb: np.ndarray,
                                   p: int, o: int) -> np.ndarray:
    v = entity_emb[o] * relation_emb[p]
    return entity_emb @ v


def evaluate_mrr_hits(entity_emb: ArrayLike,
                      relation_emb: ArrayLike,
                      test_triples: Union[ArrayLike, Iterable[Triple]],
                      known_triples: Union[ArrayLike, Iterable[Triple]],
                      hits_k: Tuple[int, ...] = (1, 3, 10),
                      filter_true: bool = True,
                      batch_size: int = 128,
                      verbose: bool = True) -> Dict[str, float]:
    """Evaluate MRR and Hits@k for DistMult link prediction (filtered setting).

    - Compatible with both PyTorch tensors and NumPy arrays.
    - Uses tqdm progress bar for tracking progress.
    - More efficient filtering via precomputed mapping.
    """


    entity_emb = _to_numpy(entity_emb)
    relation_emb = _to_numpy(relation_emb)
    n_entities = entity_emb.shape[0]

    known_triples_list = _to_triple_list(known_triples)
    test_triples_list = _to_triple_list(test_triples)

    # Precompute lookup maps for filtering
    known_sp = {}
    known_po = {}
    if filter_true:
        for (s, p, o) in known_triples_list:
            known_sp.setdefault((s, p), set()).add(o)
            known_po.setdefault((p, o), set()).add(s)

    reciprocal_ranks = []
    hits_counts = {k: 0 for k in hits_k}
    total = 0

    # tqdm progress bar
    iterator = tqdm(range(0, len(test_triples_list), batch_size),
                    disable=not verbose,
                    desc="Evaluating",
                    unit="batch")

    for i in iterator:
        batch = test_triples_list[i:i + batch_size]

        for (s, p, o) in batch:
            # (s, p, ?)
            v_tail = entity_emb[s] * relation_emb[p]
            scores_tail = entity_emb @ v_tail

            if filter_true and (s, p) in known_sp:
                for cand in known_sp[(s, p)]:
                    if cand != o:
                        scores_tail[cand] = -np.inf

            true_score = scores_tail[o]
            better = np.sum(scores_tail > true_score)
            equal = np.sum(scores_tail == true_score)
            rank_tail = 1 + better + (equal - 1) / 2.0
            reciprocal_ranks.append(1.0 / rank_tail)
            for k in hits_k:
                hits_counts[k] += rank_tail <= k

            total += 1

            # (?, p, o)
        
            v_head = entity_emb[o] * relation_emb[p]
            scores_head = entity_emb @ v_head

            if filter_true and (p, o) in known_po:
                for cand in known_po[(p, o)]:
                    if cand != s:
                        scores_head[cand] = -np.inf

            true_score = scores_head[s]
            better = np.sum(scores_head > true_score)
            equal = np.sum(scores_head == true_score)
            rank_head = 1 + better + (equal - 1) / 2.0

            for k in hits_k:
                hits_counts[k] += rank_head <= k

            total += 1

    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics = {'mrr': mrr}
    for k in hits_k:
        metrics[f'hits@{k}'] = float(hits_counts[k] / total) if total > 0 else 0.0

    return metrics
