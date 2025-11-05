from model.encoder.layer import RGCNLayer
from torch.nn.modules import Module
from torch import nn
import torch.nn.functional as F
import torch
from model.decoder.distmult import DistMultDecoder
from utils.utils import get_triples


class RGCN(nn.Module):
    def __init__(self, num_entities, hidden_layer_size, num_relations, num_bases, dropout, 
                 embedding_dim=100, decoder=None):
        """
        RGCN encoder with modular decoder.
        
        Args:
            num_entities: Number of entities in the knowledge graph
            num_relations: Number of relation types
            hidden_layer_size: Size of the hidden layer in RGCN
            num_bases: Number of bases for basis decomposition
            dropout: Dropout probability
            embedding_dim: Dimension of entity embeddings
            decoder: Decoder module (if None, defaults to DistMultDecoder)
        """
        super(RGCN, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Entity embeddings (encoder)
        self.entity_embedding = nn.Parameter(torch.FloatTensor(num_entities, embedding_dim))
        self.entity_embedding_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        
        # RGCN layers
        self.conv1 = RGCNLayer(
            embedding_dim, hidden_layer_size, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNLayer(
            hidden_layer_size, embedding_dim, num_relations * 2, num_bases=num_bases)
        
        self.dropout_ratio = dropout
        
        # Decoder (modular component)
        if decoder is None:
            self.decoder = DistMultDecoder(num_relations, embedding_dim, num_entities)
        else:
            self.decoder = decoder
    
    def forward(self, edge_index, edge_type,config):
        """
        Encode entities using RGCN layers.
        
        Args:
            entity: Entity indices [num_entities]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
        
        Returns:
            Entity embeddings [num_entities, embedding_dim]
        """

        x = self.entity_embedding + self.entity_embedding_bias
        x = torch.nn.functional.relu(x)
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)

        triples = get_triples(edge_index=edge_index,edge_type=edge_type)

        score = self.decoder(x, triples)
        penalty = self.compute_penalty(triples,x,config)
        
        return score,penalty

    def compute_penalty(self, batch, x, config):
        """ Compute L2 penalty for decoder """
        if config['decoder']['l2_penalty'] == 0.0:
            return 0

        if config['decoder']['l2_penalty_type'] == 'schlichtkrull-l2':
            return self.decoder.s_penalty(batch, x)
        else:
            return self.decoder.relations_embedding.pow(2).sum()
    