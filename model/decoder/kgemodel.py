from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from tqdm import tqdm
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.nn.kge.loader import KGTripletLoader


class KGEModel(torch.nn.Module):
    r"""An abstract base class for implementing custom KGE models.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels
        self.rel_emb = Parameter(torch.FloatTensor(num_relations, hidden_channels))
        # self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # self.rel_emb.reset_parameters()


    def forward(
        self,
        X: Tensor,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the score for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    def loss(
        self,
        X: Tensor,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ):

        pos_score = self(X, head_index, rel_type, tail_index)
        neg_score = self(X, *self.random_sample(head_index, rel_type, tail_index))

        scores = torch.cat([pos_score,neg_score])
        labels = torch.cat([
                torch.ones(pos_score.size()),
                torch.zeros(neg_score.size())
            ])
        loss = F.binary_cross_entropy_with_logits(
            input= scores,
            target= labels)
        scores = torch.sigmoid(scores)
        auc_score = roc_auc_score(y_score=scores.detach().numpy(),y_true=labels.detach().numpy())
        return loss, auc_score

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
        num_neg_per_pos: int = 10,
        head_corruption_prob: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by corrupting either the head or
        the tail (but not both) using uniform random sampling.

        Args:
            head_index (torch.Tensor): The head indices of shape [batch_size].
            rel_type (torch.Tensor): The relation type of shape [batch_size].
            tail_index (torch.Tensor): The tail indices of shape [batch_size].
            num_neg_per_pos (int, optional): Number of negative samples to generate
                per positive sample. (default: :obj:`1`)
            head_corruption_prob (float, optional): Probability of corrupting the head
                entity vs tail entity. Value should be in [0, 1]. If 0.5, equal chance
                of head or tail corruption. If 0.0, only tail corruption. If 1.0, only
                head corruption. (default: :obj:`0.5`)
        
        Returns:
            Tuple of (neg_head_index, neg_rel_type, neg_tail_index) where each tensor
            has shape [batch_size * num_neg_per_pos].
        """
        batch_size = head_index.numel()
        device = head_index.device
        
        # Repeat each positive sample num_neg_per_pos times
        # Shape: [batch_size * num_neg_per_pos]
        neg_head = head_index.repeat_interleave(num_neg_per_pos)
        neg_rel = rel_type.repeat_interleave(num_neg_per_pos)
        neg_tail = tail_index.repeat_interleave(num_neg_per_pos)
        
        total_samples = batch_size * num_neg_per_pos
        
        # Decide which samples to corrupt head vs tail
        # Shape: [batch_size * num_neg_per_pos]
        corrupt_head_mask = torch.rand(total_samples, device=device) < head_corruption_prob
        
        # Generate random node indices for corruption
        # Shape: [batch_size * num_neg_per_pos]
        rnd_nodes = torch.randint(0, self.num_nodes, (total_samples,), device=device)
        
        # Apply corruption
        neg_head[corrupt_head_mask] = rnd_nodes[corrupt_head_mask]
        neg_tail[~corrupt_head_mask] = rnd_nodes[~corrupt_head_mask]
        
        return neg_head, neg_rel, neg_tail

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')

    def s_penalty(self, head_index, rel_type, tail_index, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = head_index, rel_type, tail_index   

        s, p, o = nodes[s_index, :], self.rel_emb[p_index, :], nodes[o_index, :]

        return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()