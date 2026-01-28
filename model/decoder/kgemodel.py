from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm
from torch.nn import Parameter


class KGEModel(torch.nn.Module):
    r"""An abstract base class for implementing custom KGE models.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        embedding_dim (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int,
        sparse: bool = False,
        calibration: str = "none"
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.rel_emb = Parameter(torch.FloatTensor(num_relations, embedding_dim))
        self.calibration = calibration

        if self.calibration == "scalar":
            self.temperature = Parameter(torch.ones(1), requires_grad=False)

        elif self.calibration == "input_dependent":
            self.use_input_dependent_temp = False 
            self.temp_network = torch.nn.Sequential(
                torch.nn.Linear(2 * embedding_dim, embedding_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim // 2, 1),
                torch.nn.Softplus()  
            )
            # Initialize to output ~1.0 (no scaling) initially
            for param in self.temp_network.parameters():
                param.data.normal_(0, 0.01)
        elif self.calibration == "isotonic_regression":
            self.use_calibration = False 
            self.isotonic_regression_transform = None  # To be set during calibration
        elif self.calibration == "none":
            self.temperature = Parameter(torch.ones(1), requires_grad=False)
        else:
            raise ValueError("Unsupported calibration method specified")





    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # self.rel_emb.reset_parameters()
    
    def compute_temperature(self, head_emb: Tensor, rel_emb: Tensor) -> Tensor:
        r"""Computes input-dependent temperature for each query (h, r).
        
        Args:
            head_emb: Head entity embeddings [batch_size, embedding_dim]
            rel_emb: Relation embeddings [batch_size, embedding_dim]
            
        Returns:
            Temperature scalar for each query [batch_size, 1]
        """
        if not self.use_input_dependent_temp:
            # Return fixed temperature of 1.0
            return torch.ones(head_emb.size(0), 1, device=head_emb.device)

        # Concatenate head and relation embeddings
        query_emb = torch.cat([head_emb, rel_emb], dim=-1)  # [batch_size, 2 * embedding_dim]
        
        # Predict temperature (add small epsilon to avoid division by zero)
        temperature = self.temp_network(query_emb) + 0.01  # [batch_size, 1]
        
        return temperature


    def forward(
        self,
        X: Tensor,
        edge_index, edge_type
    ) -> Tensor:
        r"""Returns the score for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    # def loader(
    #     self,
    #     head_index: Tensor,
    #     rel_type: Tensor,
    #     tail_index: Tensor,
    #     **kwargs,
    # ) -> Tensor:
    #     r"""Returns a mini-batch loader that samples a subset of triplets.

    #     Args:
    #         head_index (torch.Tensor): The head indices.
    #         rel_type (torch.Tensor): The relation type.
    #         tail_index (torch.Tensor): The tail indices.
    #         **kwargs (optional): Additional arguments of
    #             :class:`torch.utils.data.DataLoader`, such as
    #             :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last`
    #             or :obj:`num_workers`.
    #     """
    #     return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)


    @torch.no_grad()
    def test(
        self,
        X,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        all_triples: Tensor,  # Add this parameter
        k: int = 10,
        log: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Args:
            all_triples: Tensor of shape [num_triples, 3] containing 
                        [head, rel, tail] for all known triples 
                        (train + val + test) to filter
        """
        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange
        
        # Create a set of known triples for efficient filtering
        if all_triples is not None:
            known_triples = set(
                map(tuple, all_triples.cpu().numpy())
            )
        
        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]
            
            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(self(X, h.expand_as(ts), r.expand_as(ts), ts))
            
            all_scores = torch.cat(scores)
            
            # Filtered setting: Remove scores for known triples (except target)
            if all_triples is not None:
                for tail_idx in range(self.num_nodes):
                    if tail_idx == t.item():
                        continue  # Keep the target triple
                    if (h.item(), r.item(), tail_idx) in known_triples:
                        all_scores[tail_idx] = float('-inf')  # Filter out
            
            # Compute rank of the true tail
            rank = int((all_scores.argsort(descending=True) == t).nonzero().view(-1)[0])
            
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)
        
        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
        
        return mean_rank, mrr, hits_at_k

    @torch.no_grad()
    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index
    


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'embedding_dim={self.embedding_dim})')

    def s_penalty(self, head_index, rel_type, tail_index, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = head_index, rel_type, tail_index   

        s, p, o = nodes[s_index, :], self.rel_emb[p_index, :], nodes[o_index, :]

        return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()