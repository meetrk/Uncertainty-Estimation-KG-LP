import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.evaluation import compute_mrr,compute_mrr_mc_dropout,compute_mrr_ensemble

class BasePipeline:

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
        
        # Initialize TensorBoard writer
        log_dir = Path('runs') / f"experiment_{self.config.get_section('dataset')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")
        
        # Log hyperparameters
        self.log_hyperparameters()
        
        # Training state
        self.epoch = 0
        self.training_history = {'train_loss': [], 'val_loss': [], 'eval_metrics': []}

    def start_pipeline(self) -> dict[str, list]:
        raise NotImplementedError

    def load_pipeline(self, checkpoint_path, method, save):
        raise NotImplementedError
    

    def _inference_helper(self, model, edge_index, edge_type, test_edge_index, test_edge_type, return_logits=False):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, model, edge_index, edge_type, test_edge_index, test_edge_type):
        raise NotImplementedError


    @torch.no_grad()
    def test_link_pred(self, method, model, valid_edge_index, valid_edge_type,  mc_samples=10):

        self.logger.info("Starting link prediction evaluation...")
        self.logger.info(f"Evaluation method: {method}")

        model.eval()
        if method == 'standard':
            scores = compute_mrr(valid_edge_index, valid_edge_type ,self.data, model)   

        elif method == 'mc_dropout':
            scores = compute_mrr_mc_dropout(self.data.edge_index, self.data.edge_type,
                                valid_edge_index, valid_edge_type,
                                self.data, model, mc_samples=mc_samples)
        elif method == 'ensemble':
            scores = compute_mrr_ensemble(self.data.edge_index, self.data.edge_type,
                                valid_edge_index, valid_edge_type, self.data, model)
        else:
            raise ValueError(f"Unsupported evaluation method: {method}")
        
        self.logger.info(f"MRR = {scores['mrr']:.4f}")
        self.logger.info(f"Mean Rank = {scores['mean_rank']:.4f}")
        self.logger.info(f"Hits@1 = {scores['hits@1']:.4f}")
        self.logger.info(f"Hits@3 = {scores['hits@3']:.4f} ")
        self.logger.info(f"Hits@10 = {scores['hits@10']:.4f}")
        
        return scores

    def train(self):
        """
        Train the model on a single batch.
        """
        raise NotImplementedError
        

    @torch.no_grad()
    def test(self, test = True):
        raise NotImplementedError
        

    @torch.no_grad()
    def test_uncertainty(self, model, method, edge_index, edge_type, test_edge_index, test_edge_type, uncertainty_samples):
        raise NotImplementedError


    def calibrate_pipeline(self, method, model, max_iters=50, lr=0.01):
        raise NotImplementedError

    def compute_nll_loss(self, model, params):
        raise NotImplementedError
    
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

