import torch
import torch.nn.functional as F
from torch import Tensor

from model.decoder.kgemodel import KGEModel
from sklearn.metrics import roc_auc_score


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
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self,
        X: Tensor,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:


        head = X[head_index]
        rel = self.rel_emb(rel_type)
        tail = X[tail_index]

        return (head * rel * tail).sum(dim=-1)

    