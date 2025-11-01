from model.encoder.layer import RGCNLayer
from torch.nn.modules import Module
from torch import nn
import torch.nn.functional as F
import torch
from model.decoder.distmult import DistMultDecoder

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
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        
        # RGCN layers
        self.conv1 = RGCNLayer(
            embedding_dim, hidden_layer_size, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNLayer(
            hidden_layer_size, embedding_dim, num_relations * 2, num_bases=num_bases)
        
        self.dropout_ratio = dropout
        
        # Decoder (modular component)
        if decoder is None:
            self.decoder = DistMultDecoder(num_relations, embedding_dim, num_entities, num_relations)
        else:
            self.decoder = decoder
    
    def forward(self, entity, edge_index, edge_type):
        """
        Encode entities using RGCN layers.
        
        Args:
            entity: Entity indices [num_entities]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
        
        Returns:
            Entity embeddings [num_entities, embedding_dim]
        """
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        
        return x
    
    def score_loss(self, embedding, triplets, target):
        """
        Compute prediction loss using the decoder.
        
        Args:
            embedding: Entity embeddings from forward pass
            triplets: Triplet indices [batch_size, 3]
            target: Binary labels [batch_size]
        
        Returns:
            Binary cross entropy loss
        """
        score = self.decoder(embedding, triplets)
        return F.binary_cross_entropy_with_logits(score, target)
    
    def reg_loss(self, embedding):
        """
        Compute regularization loss for both encoder and decoder.
        
        Args:
            embedding: Entity embeddings from forward pass
        
        Returns:
            Combined L2 regularization loss
        """
        entity_reg = torch.mean(embedding.pow(2))
        decoder_reg = self.decoder.regularization_loss()
        return entity_reg + decoder_reg