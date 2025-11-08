from model.encoder.layer import RGCNLayer
from torch.nn.modules import Module
from torch import nn
import torch.nn.functional as F
import torch
from model.decoder.distmult import DistMult
from utils.utils import get_triples
from sklearn.metrics import roc_auc_score
from torch_geometric.nn.conv import RGCNConv
from torch import Tensor


class RGCN(nn.Module):
    def __init__(self, num_nodes, hidden_layer_size, num_relations, num_bases, dropout, 
                 embedding_dim=100, decoder=None):
        """
        RGCN encoder with modular decoder.
        
        Args:
            num_nodes: Number of entities in the knowledge graph
            num_relations: Number of relation types
            hidden_layer_size: Size of the hidden layer in RGCN
            num_bases: Number of bases for basis decomposition
            dropout: Dropout probability
            embedding_dim: Dimension of entity embeddings
            decoder: Decoder module (if None, defaults to DistMultDecoder)
        """
        super(RGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Entity embeddings (encoder)
        self.entity_embedding = nn.Parameter(torch.FloatTensor(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.entity_embedding)
        self.entity_embedding_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        
        # RGCN layers
        self.conv1 = RGCNLayer(
            embedding_dim, hidden_layer_size, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNLayer(
            hidden_layer_size, embedding_dim, num_relations * 2, num_bases=num_bases)
        
        # self.conv1 = RGCNConv(
        #     in_channels=embedding_dim,
        #     out_channels=hidden_layer_size,
        #     num_relations=num_relations,
        #     num_bases=num_bases,
        #     aggr="add",
        #     bias=False
        # )
        # self.conv2 = RGCNConv(
        #     in_channels=embedding_dim,
        #     out_channels=hidden_layer_size,
        #     num_relations=num_relations,
        #     num_bases=num_bases,
        #     aggr="add",
        #     bias=False
        # )

        self.dropout_ratio = dropout
        
        # Decoder (modular component)
        if decoder is None:
            self.decoder = DistMult(num_nodes, num_relations, embedding_dim)
        else:
            self.decoder = decoder
    
    def forward(self, edge_index, edge_type):
        """
        Encode entities using RGCN layers.
        
        Args:
            entity: Entity indices [num_nodes]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
        
        Returns:
            Entity embeddings [num_nodes, embedding_dim]
        """

        x = self.entity_embedding + self.entity_embedding_bias
        x = torch.nn.functional.relu(x)
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        triples = get_triples(edge_index=edge_index,edge_type=edge_type)

        pred_logits = self.decoder(x, triples[:,0], triples[:,1], triples[:,2])

        loss,roc_auc_score = self.decoder.loss(x, triples[:,0], triples[:,1], triples[:,2])
        
        return pred_logits, loss, roc_auc_score
    
    @torch.no_grad()
    def test(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        k: int = 10,
        log: bool = True):

        self.eval()
        print("Starting evaluation...")
        print(f"Batch size: {batch_size}, Top-K: {k}")
        return self.decoder.test(
            self.entity_embedding + self.entity_embedding_bias,
            head_index,
            rel_type,
            tail_index,
            batch_size,
            k,
            log
        )

