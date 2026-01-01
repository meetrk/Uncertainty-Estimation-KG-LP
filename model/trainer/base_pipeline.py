"""
Base pipeline class with common functionality for training and evaluation.

This abstract base class provides shared functionality for different pipeline
implementations (e.g., standard training, ensemble training) to reduce code
duplication and improve maintainability.
"""
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class BasePipeline(ABC):
    """
    Abstract base class for training pipelines.
    
    Provides common functionality for:
    - TensorBoard logging
    - Checkpoint management
    - Hyperparameter tracking
    - Training history
    """
    
    def __init__(self, model, data, config, logger):
        """
        Initialize base pipeline.
        
        Args:
            model: The model or ensemble to train
            data: Dataset containing train/val/test splits
            config: Configuration object
            logger: Logger instance
        """
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        
        # Extract common configurations
        self.model_config = self.config.get_section('model')
        self.train_config = self.config.get_section('training')
        self.learning_rate = self.train_config['optimiser']['learning_rate']
        self.weight_decay = self.train_config['optimiser']['weight_decay']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'eval_metrics': []
        }
        
        # Setup TensorBoard
        self._setup_tensorboard()
        self.log_hyperparameters()
    
    def _setup_tensorboard(self):
        """Initialize TensorBoard writer with appropriate log directory."""
        experiment_name = self._get_experiment_name()
        log_dir = Path('runs') / f"{experiment_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    @abstractmethod
    def _get_experiment_name(self) -> str:
        """Get experiment name for TensorBoard logs."""
        pass
    
    @abstractmethod
    def start_pipeline(self):
        """Start the training pipeline. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train(self) -> float:
        """Execute one training epoch. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def test(self) -> tuple:
        """Evaluate model on validation/test sets. Must be implemented by subclasses."""
        pass
    
    def _early_stopping_check(self, current_metric, best_metric, delta):
        """
        Check if current metric shows improvement over best metric.
        
        Args:
            current_metric: Current validation metric
            best_metric: Best validation metric so far
            delta: Minimum change to qualify as improvement
            
        Returns:
            bool: True if improvement detected
        """
        return current_metric - best_metric > delta
    
    def log_model_gradients(self, epoch):
        """
        Log gradient norms to TensorBoard for monitoring.
        
        Args:
            epoch: Current epoch number
        """
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                self.writer.add_scalar(f'Gradients/{name}', param_norm, epoch)
        
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Gradients/Total_Norm', total_norm, epoch)
    
    def log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        hparams = self._get_hyperparameters()
        
        # Add text summary
        hparam_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
        self.writer.add_text('Hyperparameters', hparam_text, 0)
        
        # Log as scalars for comparison
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Hyperparameters/{key}', value, 0)
    
    def _get_hyperparameters(self):
        """
        Get dictionary of hyperparameters to log.
        Subclasses can override to add specific hyperparameters.
        
        Returns:
            dict: Hyperparameters dictionary
        """
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.train_config['epochs'],
            'negative_sampling_ratio': self.train_config['sampling']['negative_sampling_ratio'],
            'embedding_dim': self.model_config['encoder']['embedding_dim'],
            'hidden_layer_size': self.model_config['encoder']['hidden_layer_size'],
            'num_bases': self.model_config['encoder'].get('num_bases', 'N/A'),
        }
    
    def save_checkpoint(self, epoch, name=None):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            name: Optional custom checkpoint name
        """
        checkpoint = self._build_checkpoint(epoch)
        
        if name is None:
            name = self._get_default_checkpoint_name(epoch)
        
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / name
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    @abstractmethod
    def _build_checkpoint(self, epoch) -> dict:
        """
        Build checkpoint dictionary.
        Subclasses must implement to include model-specific state.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Checkpoint dictionary
        """
        pass
    
    def _get_default_checkpoint_name(self, epoch):
        """
        Get default checkpoint filename.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            str: Checkpoint filename
        """
        dataset_name = self.config.get_section("dataset")["name"]
        return f'{dataset_name}_checkpoint_epoch_{epoch}.pth'
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._restore_from_checkpoint(checkpoint)
        self.epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    @abstractmethod
    def _restore_from_checkpoint(self, checkpoint):
        """
        Restore model state from checkpoint.
        Subclasses must implement to restore model-specific state.
        
        Args:
            checkpoint: Checkpoint dictionary
        """
        pass
    
    def log_training_metrics(self, epoch, train_loss, val_loss=None):
        """
        Log training metrics to TensorBoard and history.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Optional validation loss value
        """
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.training_history['train_loss'].append({"epoch": epoch, "epoch_loss": train_loss})
        
        if val_loss is not None:
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.training_history['val_loss'].append({"epoch": epoch, "epoch_loss": val_loss})
    
    def log_evaluation_metrics(self, epoch, valid_scores, test_scores=None):
        """
        Log evaluation metrics to TensorBoard and history.
        
        Args:
            epoch: Current epoch number
            valid_scores: Validation metrics dictionary
            test_scores: Optional test metrics dictionary
        """
        # Ensure test_scores exists with default values
        if test_scores is None:
            test_scores = {
                "mrr": 0, "mean_rank": 0,
                "hits@1": 0, "hits@3": 0, "hits@10": 0
            }
        
        # Log to history
        metric_entry = {
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
        }
        self.training_history['eval_metrics'].append(metric_entry)
        
        # Log to TensorBoard
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
        
        # Log to console
        self.logger.info(f"Epoch {epoch}: Val MRR = {valid_scores['mrr']:.4f}, Test MRR = {test_scores['mrr']:.4f}")
        self.logger.info(f"Epoch {epoch}: Val Mean Rank = {valid_scores['mean_rank']:.4f}, Test Mean Rank = {test_scores['mean_rank']:.4f}")
        self.logger.info(f"Epoch {epoch}: Val Hits@1 = {valid_scores['hits@1']:.4f}, Test Hits@1 = {test_scores['hits@1']:.4f}")
        self.logger.info(f"Epoch {epoch}: Val Hits@3 = {valid_scores['hits@3']:.4f}, Test Hits@3 = {test_scores['hits@3']:.4f}")
        self.logger.info(f"Epoch {epoch}: Val Hits@10 = {valid_scores['hits@10']:.4f}, Test Hits@10 = {test_scores['hits@10']:.4f}")
    
    def __del__(self):
        """Cleanup TensorBoard writer when pipeline is destroyed."""
        if hasattr(self, 'writer'):
            self.writer.close()
