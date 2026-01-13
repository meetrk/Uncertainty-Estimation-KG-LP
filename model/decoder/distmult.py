import torch
from torch import Tensor

from model.decoder.kgemodel import KGEModel
import torch.nn.functional as F
from utils.utils import negative_sampling



class DistMult(KGEModel):
    r"""The DistMult model from the `"Embedding Entities and Relations for
    Learning and Inference in Knowledge Bases"
    <https://arxiv.org/abs/1412.6575>`_ paper.

    :class:`DistMult` models relations as diagonal matrices, which simplifies
    the bi-linear interaction between the head and tail entities to the score
    function:

    .. math::
        d(h, r, t) = < \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t >

    .. note::

        For an example of using the :class:`DistMult` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
        calibration: str = "none",
        
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse, calibration)

        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # super().reset_parameters()
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(
        self,
        X: Tensor,
        edge_index, edge_type
    ) -> Tensor:

        head, tail = X[edge_index[0]], X[edge_index[1]]
        rel = self.rel_emb[edge_type]

        if self.calibration == "input_dependent":
            temperature = self.compute_temperature(head, rel) 

            scores = torch.sum(head * rel * tail, dim=1, keepdim=True)  
            
            scores = scores / temperature  

            return scores.squeeze(-1)  
            
        elif self.calibration == "scalar":
            
            scores = torch.sum(head * rel * tail, dim=1, keepdim=True)  
            scores = scores / self.temperature 
            
            return scores
        elif self.calibration == "isotonic_regression":

            logits = torch.sum(head * rel * tail, dim=1, keepdim=True)
            if self.use_calibration and self.isotonic_regression_transform is not None:
                probab = torch.sigmoid(logits)
                scores = self.isotonic_regression_transform.predict(probab.cpu().numpy())
                scores = torch.from_numpy(scores).to(logits.device).unsqueeze(-1)
                return scores.squeeze(-1)
            else:
                return logits.squeeze(-1)
        else:
            scores  = torch.sum(head * rel * tail, dim=1, keepdim=True)
            return scores

    def compute_nll_loss(self, X, edge_index, edge_type):
        pos_scores = self.forward(X, edge_index, edge_type)
        neg_edge_index, neg_edge_type = negative_sampling(edge_index, edge_type, self.num_nodes, 1)
        neg_scores = self.forward(X, neg_edge_index, neg_edge_type)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        nll_loss = F.binary_cross_entropy_with_logits(scores, labels)
        return nll_loss

    def calibrate(self, encoder_fn, train_edge_index, train_edge_type, val_edge_index, val_edge_type):
        """
        Calibrate input-dependent temperature network.
        
        Args:
            encoder_fn: Function that encodes the graph (callable that takes edge_index, edge_type)
            train_edge_index: Training edge index for encoding
            train_edge_type: Training edge type for encoding
            val_edge_index: Validation edge index for computing loss
            val_edge_type: Validation edge type for computing loss
        """
        print(f"Calibration method: {self.calibration}")
        param_filter = lambda name: 'temp_network' in name or 'temperature' in name
        for name, param in self.named_parameters():
            if param_filter(name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.use_input_dependent_temp = True
        optimizer = torch.optim.Adam(self.temp_network.parameters(), lr=0.01)
        self.temp_network.train()
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience, patience_counter = 5, 0

        for iter in range(1, 100):
            optimizer.zero_grad()
            
            # Re-encode the graph each iteration to maintain computation graph
            X = encoder_fn(train_edge_index, train_edge_type)
            
            loss = self.compute_nll_loss(X, val_edge_index, val_edge_type)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.temp_network.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break
        
        self.temp_network.eval()
        print("Calibration Complete!")
        print(f"Final NLL Loss: {best_loss:.4f}")
        
        return best_loss


    

    
        

