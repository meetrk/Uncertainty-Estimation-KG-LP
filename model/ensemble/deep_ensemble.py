"""
Deep Ensemble wrapper for uncertainty estimation in knowledge graph link prediction.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GAE
from typing import Tuple



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
        device: str = 'cuda'
    ):
        super(DeepEnsemble, self).__init__()
        
        self.num_models = num_models
        self.device = device
        self.base_encoder_class = base_encoder_class
        self.base_decoder_class = base_decoder_class
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        
        # Create ensemble of models
        self.models = nn.ModuleList()
        for i in range(num_models):
            encoder = base_encoder_class(**encoder_args)
            decoder = base_decoder_class(**decoder_args)
            model = GAE(encoder=encoder, decoder=decoder)
            self.models.append(model)
        
        self.to(device)
    
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
    
    def forward_ensemble(self, edge_index, edge_type, pred_edge_index, pred_edge_type):
        """
        Forward pass through all ensemble members.
        
        Returns:
            predictions: List of predictions from each model
        """
        predictions = []
        for model in self.models:
            z = model.encode(edge_index, edge_type)
            pred = model.decode(z, pred_edge_index, pred_edge_type)
            predictions.append(pred)
        
        return predictions
    
    def predict_with_uncertainty(
        self, 
        edge_index, 
        edge_type, 
        pred_edge_index, 
        pred_edge_type,
        return_all_preds: bool = False,
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
        
        if return_all_preds:
            return mean_pred, std_pred, preds_stack
        else:
            return mean_pred, std_pred
    
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
