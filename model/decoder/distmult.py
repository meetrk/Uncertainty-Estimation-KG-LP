import torch
from torch import Tensor
from model.decoder.kgemodel import KGEModel



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
        embedding_dim: int,
        margin: float = 1.0,
        sparse: bool = False,
        calibration: str = "none",
        
    ):
        super().__init__(num_nodes, num_relations, embedding_dim, sparse, calibration)

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
        scores = torch.sum(head * rel * tail, dim=1, keepdim=True)
        return scores.squeeze(-1)
