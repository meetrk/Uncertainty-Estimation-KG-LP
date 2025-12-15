from model.encoder.layer import RGCNLayer
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn.conv import RGCNConv


class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, model_config,):
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
        self.mc_dropout = False
        
        # Entity embeddings (encoder)
        self.entity_embedding = nn.Parameter(torch.FloatTensor(num_nodes, self.embedding_dim))
        self.entity_embedding_bias = nn.Parameter(torch.zeros(1, self.embedding_dim))
        
        # RGCN layers
        # self.conv1 = RGCNLayer(
        #     self.embedding_dim, self.hidden_layer_size, self.num_relations, num_bases=self.num_bases, w_init=self.w_init, w_gain=self.w_gain, b_init=self.b_init)
        # self.conv2 = RGCNLayer(
        #     self.hidden_layer_size, self.embedding_dim, self.num_relations, num_bases=self.num_bases, w_init=self.w_init, w_gain=self.w_gain, b_init=self.b_init)
        self.conv1 = RGCNConv(self.embedding_dim, self.hidden_layer_size, self.num_relations,
                              num_blocks=5)
        self.conv2 = RGCNConv(self.hidden_layer_size, self.embedding_dim, self.num_relations,
                              num_blocks=5)

        self.reset_parameters()


    def forward(self, edge_index, edge_type):
        x = self.entity_embedding + self.entity_embedding_bias
        x = self.conv1(x, edge_index, edge_type).relu_()
        dropout = self.training or self.mc_dropout
        x = F.dropout(x, p=self.dropout_ratio, training=dropout)
        x = self.conv2(x, edge_index, edge_type)
        return x
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_embedding)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()