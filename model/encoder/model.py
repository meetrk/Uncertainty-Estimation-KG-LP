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
    def __init__(self, num_nodes, num_relations, model_config, decoder=None,):
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

        self.embedding_dim = model_config['encoder']['embedding_dim']
        self.num_bases = model_config['encoder']['num_bases']
        self.dropout_ratio = model_config['encoder']['dropout']
        self.hidden_layer_size = model_config['encoder']['hidden_layer_size']
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.w_init = model_config['encoder'].get('w_init', None)
        self.w_gain = model_config['encoder'].get('w_gain', False)
        self.b_init = model_config['encoder'].get('b_init', False)
        self.decoder_l2_type =model_config['decoder'].get('l2_type', 'schlichtkrull-l2')
        self.decoder_l2_penalty = model_config['decoder'].get('l2_penalty', 0.0)

        
        # Entity embeddings (encoder)
        self.entity_embedding = nn.Parameter(torch.FloatTensor(num_nodes, self.embedding_dim))
        nn.init.xavier_uniform_(self.entity_embedding)
        self.entity_embedding_bias = nn.Parameter(torch.zeros(1, self.embedding_dim))
        
        # RGCN layers
        self.conv1 = RGCNLayer(
            self.embedding_dim, self.hidden_layer_size, self.num_relations * 2 + 1, num_bases=self.num_bases, w_init=self.w_init, w_gain=self.w_gain, b_init=self.b_init)
        self.conv2 = RGCNLayer(
            self.hidden_layer_size, self.embedding_dim, self.num_relations * 2 + 1, num_bases=self.num_bases, w_init=self.w_init, w_gain=self.w_gain, b_init=self.b_init)
        # self.conv1 = RGCNConv(
        #     in_channels=self.embedding_dim,
        #     out_channels=self.hidden_layer_size,
        #     num_relations=self.num_relations * 2 + 1,
        #     num_bases=self.num_bases,
        #     aggr="add",
        #     bias=False
        # )
        # self.conv2 = RGCNConv(
        #     in_channels=self.embedding_dim,
        #     out_channels=self.hidden_layer_size,
        #     num_relations=self.num_relations * 2 + 1,
        #     num_bases=self.num_bases,
        #     aggr="add",
        #     bias=False
        # )
        
        # Decoder (modular component)
        if decoder is None:
            self.decoder = DistMult(num_nodes, num_relations, self.embedding_dim)
        else:
            self.decoder = decoder
    
    def forward(self, batch, all_triples, entity_count, head_corrupt_prob,negative_sampling_ratio,X=None):
        """
        Encode entities using RGCN layers.
        
        Args:
            entity: Entity indices [num_nodes]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
        
        Returns:
            Entity embeddings [num_nodes, embedding_dim]
        """
        if X is None:
            x = self.entity_embedding + self.entity_embedding_bias
        else:
            x = X
        x = torch.nn.functional.relu(x)
        x = self.conv1(x, batch.edge_index, batch.edge_type)
        x = F.relu(x)
        x = self.conv2(x, batch.edge_index, batch.edge_type)

        head_index = batch.edge_label_index[0, :]
        rel_type = batch.edge_label_type
        tail_index = batch.edge_label_index[1, :]

        assert head_index.size() == rel_type.size() == tail_index.size()

        loss,scores = self.decoder.loss(x, head_index, rel_type, tail_index,all_triples, entity_count, head_corrupt_prob,negative_sampling_ratio)

        loss += self.compute_penalty(head_index, rel_type, tail_index, x)  * self.decoder_l2_penalty

        return loss, scores

    def compute_penalty(self, head_index, rel_type, tail_index, x):
        """ Compute L2 penalty for decoder """
        if self.decoder_l2_penalty == 0.0:
            return 0

        if self.decoder_l2_type == 'schlichtkrull-l2':
            return self.decoder.s_penalty(head_index, rel_type, tail_index, x)
        else:
            return self.decoder.rel_emb.pow(2).sum()

    @torch.no_grad()
    def test(
        self,
        batch,
        batch_size: int,
        all_triples,
        k: int = 10,
        log: bool = True,
        ):

        head_index = batch.edge_label_index[:, 0]
        rel_type = batch.edge_label_type
        tail_index = batch.edge_label_index[:, 1]

        self.eval()
        print("Starting evaluation...")
        return self.decoder.test(
            self.entity_embedding + self.entity_embedding_bias,
            head_index,
            rel_type,
            tail_index,
            batch_size,
            all_triples,
            k,
            log
        )

