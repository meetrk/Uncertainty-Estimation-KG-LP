from torch import nn
import torch
from torch.nn import Parameter
from torch.nn.init import uniform_  
from utils.initialiser import schlichtkrull_normal_


# Base Decoder Class
class Decoder(nn.Module):
    """Base class for knowledge graph decoders."""
    
    def __init__(self, num_relations, embedding_dim):
        super(Decoder, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
    
    def forward(self, embedding, triplets):
        """
        Score triplets given entity embeddings.
        
        Args:
            embedding: Entity embeddings [num_entities, embedding_dim]
            triplets: Triplet indices [batch_size, 3] (head, relation, tail)
        
        Returns:
            Scores for each triplet [batch_size]
        """
        raise NotImplementedError
    
    def regularization_loss(self):
        """Return L2 regularization loss for decoder parameters."""
        raise NotImplementedError

class DistMultDecoder(Decoder):
    """
    DistMult decoder for knowledge graph link prediction.
    
    Implements the scoring function: score = sum(head * relation * tail)
    """
    
    def __init__(self, num_relations,
                embedding_dim,
                num_nodes,
                w_init='standard-normal',
                w_gain=False,
                b_init=True,
                ):
        super(DistMultDecoder, self).__init__(num_relations, embedding_dim)
        self.w_init = w_init
        self.w_gain = w_gain
        self.b_init = b_init

        # Create weights & biases
        self.relations_embedding = nn.Parameter(torch.FloatTensor(num_relations, embedding_dim))
        if b_init:
            self.sbias = Parameter(torch.FloatTensor(num_nodes))
            self.obias = Parameter(torch.FloatTensor(num_nodes))
            self.pbias = Parameter(torch.FloatTensor(num_relations))
        else:
            self.register_parameter('sbias', None)
            self.register_parameter('obias', None)
            self.register_parameter('pbias', None)


        self.reset_parameters()
    
    def reset_parameters(self):
        
        if self.w_gain:
            gain = nn.init.calculate_gain('relu')
            schlichtkrull_normal_(self.relations_embedding, shape=self.relations_embedding.shape, gain=gain)
        else:
            schlichtkrull_normal_(self.relations_embedding, shape=self.relations_embedding.shape)

        # Biases
        if self.b_init:
            init = uniform_
            init(self.sbias)
            init(self.pbias)
            init(self.obias)
        nn.init.xavier_uniform_(self.relations_embedding, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, embedding, triplets):
        """
        Compute DistMult scores.
        
        Args:
            embedding: Entity embeddings from encoder [num_entities, embedding_dim]
            triplets: [batch_size, 3] containing (head_idx, relation_idx, tail_idx)
        
        Returns:
            Scores [batch_size]
        """
        s = embedding[triplets[:, 0]]  # Head entities
        r = self.relations_embedding[triplets[:, 1]]  # Relations
        o = embedding[triplets[:, 2]]  # Tail entities
        scores = torch.sum(s * r * o, dim=1)

        if self.b_init:
            scores = scores + (self.sbias[triplets[:, 0]] + self.pbias[triplets[:, 1]] + self.obias[triplets[:, 2]])

        return scores
    
    def regularization_loss(self):
        """L2 regularization on relation embeddings."""
        return torch.mean(self.relations_embedding.pow(2))
    