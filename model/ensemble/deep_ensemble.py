"""
Deep Ensemble wrapper for uncertainty estimation in knowledge graph link prediction.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GAE
from typing import Tuple
from torch.nn import Parameter



class DeepEnsemble(nn.Module):
    """
    Deep Ensemble model for uncertainty estimation.
    
    Trains multiple independent models with different initializations
    and aggregates their predictions for uncertainty quantification.
    
    Args:
        base_encoder_class: The encoder class (e.g., RGCN)
        base_decoder_class: The decoder class (e.g., DistMult)
        encoder_args: Arguments for encoder initialization
        decoder_args: Arguments for decoder initialization
        num_models: Number of ensemble members (default: 5)
        device: Device to use for computation
    """
    
    def __init__(
        self,
        base_encoder_class,
        base_decoder_class,
        encoder_args: dict,
        decoder_args: dict,
        num_models: int = 5,
        device: str = 'cuda',
        calibration: str = "none"
    ):
        super(DeepEnsemble, self).__init__()
        
        self.num_models = num_models
        self.device = device
        self.base_encoder_class = base_encoder_class
        self.base_decoder_class = base_decoder_class
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.calibration = calibration
        
        # Create ensemble of models
        self.models = nn.ModuleList()
        for i in range(num_models):
            encoder = base_encoder_class(**encoder_args)
            decoder = base_decoder_class(**decoder_args)
            model = GAE(encoder=encoder, decoder=decoder)
            self.models.append(model)
        
        self.to(device)

        if self.calibration == "scalar":
            self.temperature = Parameter(torch.ones(1), requires_grad=False)
            self.temperature.to(device)

        elif self.calibration == "input_dependent":
            self.temp_network = torch.nn.Sequential(
                torch.nn.Linear(2 * encoder_args['hidden_channels'], encoder_args['hidden_channels'] // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(encoder_args['hidden_channels'] // 2, 1),
                torch.nn.Softplus()  
            )
            self.temp_network.to(device)
            # Initialize to output ~1.0 (no scaling) initially
            for param in self.temp_network.parameters():
                param.data.normal_(0, 0.01)
        elif self.calibration == "isotonic_regression":
            self.isotonic_regression_transform = None  # To be set during calibration

        elif self.calibration == "none":
            self.temperature = Parameter(torch.ones(1), requires_grad=False)
            self.temperature.to(device)
        else:
            raise ValueError("Unsupported calibration method specified")
        
        self.use_calibration = False

    
    def get_model(self, idx: int):
        """Get a specific model from the ensemble."""
        return self.models[idx]
    
    def encode(self, edge_index, edge_type, model_idx: int = None):
        """
        Encode using a specific model or all models.
        
        Args:
            edge_index: Edge indices
            edge_type: Edge types
            model_idx: If specified, use only this model. Otherwise use all.
        """
        if model_idx is not None:
            return self.models[model_idx].encode(edge_index, edge_type)
        else:
            # Return list of encodings from all models
            return [model.encode(edge_index, edge_type) for model in self.models]
    
    def decode(self, z, edge_index, edge_type, model_idx: int = None):
        """
        Decode using a specific model or all models.
        
        Args:
            z: Node embeddings (can be list if from all models)
            edge_index: Edge indices
            edge_type: Edge types
            model_idx: If specified, use only this model
        """
        if model_idx is not None:
            return self.models[model_idx].decode(z, edge_index, edge_type)
        else:
            # z should be a list of embeddings
            if not isinstance(z, list):
                raise ValueError("z must be a list of embeddings when model_idx is None")
            return [self.models[i].decode(z[i], edge_index, edge_type) 
                    for i in range(self.num_models)]
    
    def forward_ensemble(self, edge_index, edge_type, pred_edge_index, pred_edge_type) -> list:
        """
        Forward pass through all ensemble members.
        
        Returns:
            predictions: List of predictions from each model
        """
        predictions = []
        for model in self.models:
            z = model.encode(edge_index, edge_type)
            logits = model.decode(z, pred_edge_index, pred_edge_type)
            pred = torch.sigmoid(logits)
            predictions.append(pred)
        
        return predictions
    
    def predict_with_uncertainty(
        self, 
        edge_index, 
        edge_type, 
        pred_edge_index, 
        pred_edge_type,
        return_logits: bool = False,
        enable_grad: bool = False  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty estimates."""
        self.eval()
        if enable_grad:
            predictions = self.forward_ensemble(
                edge_index, edge_type, pred_edge_index, pred_edge_type
            )
        else:
            with torch.no_grad():
                predictions = self.forward_ensemble(
                    edge_index, edge_type, pred_edge_index, pred_edge_type
                )
        
        # Rest stays the same
        preds_stack = torch.stack(predictions, dim=0)
        mean_pred = preds_stack.mean(dim=0)
        std_pred = preds_stack.std(dim=0)
        eps = 1e-6
        mean_prob_clamped = torch.clamp(mean_pred, min=eps, max=1-eps)
        ensemble_logit = torch.logit(mean_prob_clamped)

        if self.calibration == "scalar" and self.use_calibration:
            calibrated_logit = ensemble_logit / self.temperature.to(ensemble_logit.device)
        else:
            calibrated_logit = ensemble_logit

        if return_logits:
            return calibrated_logit, std_pred
        else:
            return torch.sigmoid(calibrated_logit), std_pred        
    
    def get_optimizers(self, lr: float, weight_decay: float = 0.0):
        """
        Create separate optimizers for each ensemble member.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay
            
        Returns:
            List of optimizers, one per model
        """
        optimizers = []
        for model in self.models:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            optimizers.append(optimizer)
        
        return optimizers
    
    def save_ensemble(self, path: str):
        """Save all ensemble models."""
        ensemble_state = {
            'num_models': self.num_models,
            'encoder_args': self.encoder_args,
            'decoder_args': self.decoder_args,
            'models': [model.state_dict() for model in self.models]
        }
        torch.save(ensemble_state, path)
    
    def load_ensemble(self, path: str):
        """Load all ensemble models."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        for i, state_dict in enumerate(checkpoint['models']):
            self.models[i].load_state_dict(state_dict)
