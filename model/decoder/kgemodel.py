from typing import Tuple

import torch
from torch import Tensor
from collections import defaultdict
from tqdm import tqdm
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

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
        all_triples, entity_count, head_corrupt_prob,negative_sampling_ratio
    ):

        pos_score = self(X, head_index, rel_type, tail_index)
        neg_score = self(X, *self.negative_sampling(head_index, rel_type, tail_index, all_triples, entity_count, head_corrupt_prob,negative_sampling_ratio))

        scores = torch.cat([pos_score,neg_score])
        labels = torch.cat([
                torch.ones(pos_score.size()),
                torch.zeros(neg_score.size())
            ])
        loss = F.binary_cross_entropy_with_logits(
            input= scores,
            target= labels)
        auc_score = roc_auc_score(y_score=scores.detach().numpy(),y_true=labels.detach().numpy())
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels.detach().numpy(), (scores.detach().numpy() > 0).astype(int), average='binary')
        scores = {
            "auc": auc_score,
            "precision": precision,
            "recall": recall,
            "f1": f1_score
        }


        return loss, scores

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
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        batch_size: int,
        all_triples: torch.Tensor, 
        k: int = 10,
        log: bool = True,
    ) -> tuple[float, float, float]:
        
        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange
        
        # 1. Pre-compute filter map: (h, r) -> list of known tails
        # This moves the heavy lifting outside the testing loop
        to_filter = defaultdict(list)
        if all_triples is not None:
            # Move to CPU for dictionary creation to save GPU memory/time
            all_triples_cpu = all_triples.cpu()
            for i in range(all_triples_cpu.size(0)):
                h_idx, r_idx, t_idx = all_triples_cpu[i].tolist()
                to_filter[(h_idx, r_idx)].append(t_idx)

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []

        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]
            
            # Calculate scores for all possible tails
            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(self(X, h.expand_as(ts), r.expand_as(ts), ts))
            all_scores = torch.cat(scores)
            
            # 2. Vectorized Filtering
            # Instead of looping num_nodes, we look up the specific indices to mask
            if all_triples is not None:
                filter_indices = to_filter[(h.item(), r.item())]
                
                # Convert to tensor for indexing
                # Note: Ensure the target 't' is NOT in this list, or explicitely unmask it
                filter_indices = torch.tensor(filter_indices, device=all_scores.device)
                
                # Apply mask
                all_scores.index_fill_(0, filter_indices, float('-inf'))
                
                # CRITICAL: Ensure the target triple itself is not filtered out
                # (In case the target triple was in 'all_triples', which it usually is)
                target_score = self(X, h.view(1), r.view(1), t.view(1)).squeeze()
                all_scores[t] = target_score

            # 3. Optimized Ranking (Avoiding full sort)
            # We don't need to sort the whole array to find the rank. 
            # Rank is simply: count(scores > target_score) + 1
            
            # Get the score of the true target
            target_score = all_scores[t]
            
            # Count how many scores are strictly greater than the target
            # (Using strictly greater handles ties optimistically, >= handles strictly)
            # Standard convention is usually "rank is count of scores >= target"
            # but masking makes them unique usually.
            # Below is the 'strict' rank calculation (best rank = 1)
            rank = (all_scores > target_score).sum().item() + 1
            
            mean_ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
            hits_at_k.append(rank <= k)

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
    
    @torch.no_grad()
    def negative_sampling(
        self, 
        head_index, 
        rel_type, 
        tail_index, 
        all_triples, 
        entity_count,
        head_corrupt_prob=0.5, 
        negative_sampling_ratio=1
    ):
        # Step 1: Decide number of negatives per positive
        batch_size = head_index.numel()
        num_negatives = batch_size * negative_sampling_ratio

        # Step 2: Repeat triples for negatives
        head_pos = head_index.repeat(negative_sampling_ratio)
        rel_pos = rel_type.repeat(negative_sampling_ratio)
        tail_pos = tail_index.repeat(negative_sampling_ratio)

        # Step 3: Randomly decide which to corrupt (head or tail)
        # True = corrupt head, False = corrupt tail
        corruption_mask = torch.rand(num_negatives, device=head_index.device) < head_corrupt_prob

        # Step 4: Draw random entity replacements
        random_entities = torch.randint(0, entity_count, (num_negatives,), device=head_index.device)

        neg_heads = head_pos.clone()
        neg_tails = tail_pos.clone()

        # Apply corruption
        neg_heads[corruption_mask] = random_entities[corruption_mask]
        neg_tails[~corruption_mask] = random_entities[~corruption_mask]

        # **Optional Step 5: Filter out any negatives that are actually positives**
        # Build set of all true triples for fast lookup
        # (for large graphs, skip or make approximate)
        triple_set = set(tuple(triple.tolist()) for triple in all_triples)
        negatives = []
        for h, r, t in zip(neg_heads.tolist(), rel_pos.tolist(), neg_tails.tolist()):
            if (h, r, t) not in triple_set:
                negatives.append((h, r, t))

        # Convert to tensor
        if negatives:
            negatives = torch.tensor(negatives, dtype=head_index.dtype, device=head_index.device).T
            neg_heads, rel_neg, neg_tails = negatives[0], negatives[1], negatives[2]
        else:
            neg_heads, rel_neg, neg_tails = neg_heads, rel_pos, neg_tails

        # Output as you need (stacked)
        return neg_heads, rel_neg, neg_tails


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')

    def s_penalty(self, head_index, rel_type, tail_index, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = head_index, rel_type, tail_index   

        s, p, o = nodes[s_index, :], self.rel_emb[p_index, :], nodes[o_index, :]

        return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()