from model.decoder.distmult import Decoder
import torch 
from torch import nn
from torch.nn import Parameter
from utils.initialiser import schlichtkrull_normal_
from torch.nn.init import uniform_  
from torch.nn import functional as F




class TransE(Decoder):



    def __init__(self, num_relations,
                embedding_dim,
                num_nodes,
                w_init='standard-normal',
                w_gain=False,
                b_init=True,
                ):
        super(TransE, self).__init__(num_relations, embedding_dim)

        self.relations_embedding = Parameter(torch.FloatTensor(num_relations, embedding_dim))
        self.b_init = b_init
        if b_init:
            self.sbias = Parameter(torch.FloatTensor(num_nodes))
            self.obias = Parameter(torch.FloatTensor(num_nodes))
            self.pbias = Parameter(torch.FloatTensor(num_relations))
        else:
            self.register_parameter('sbias', None)
            self.register_parameter('obias', None)
            self.register_parameter('pbias', None)


    
    def reset_parameter(self):

        nn.init.xavier_uniform_(self.relations_embedding, gain=1.0)
        
        # Normalize after initialization
        with torch.no_grad():
            self.relations_embedding.data = F.normalize(self.relations_embedding.data, p=2, dim=1)

        # Biases - initialize to small values
        if self.b_init:
            nn.init.zeros_(self.sbias)
            nn.init.zeros_(self.pbias)
            nn.init.zeros_(self.obias)


    def forward(self,embedding, triplets):

        s = embedding[triplets[:, 0]]  # Head entities
        r = self.relations_embedding[triplets[:, 1]]  # Relations
        o = embedding[triplets[:, 2]]  # Tail entities

        s = torch.nn.functional.normalize(s, p=2, dim=1)
        r = torch.nn.functional.normalize(r, p=2, dim=1)
        o = torch.nn.functional.normalize(o, p=2, dim=1)

        scores = torch.linalg.vector_norm(s + r - o, dim=1, ord=2)
        
        if self.b_init:
            scores = scores + (self.sbias[triplets[:, 0]] + self.pbias[triplets[:, 1]] + self.obias[triplets[:, 2]])
        scores = -scores  # Negate the scores for compatibility with loss functions
        return scores
    
    def s_penalty(self, triples, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = triples[:,0], triples[:,1], triples[:,2]

        s, p, o = nodes[s_index, :], self.relations_embedding[p_index, :], nodes[o_index, :]

        # Return mean of squared norms (not individual element squares)
        # This is the standard L2 regularization
        s_norm = torch.norm(s, p=2, dim=1).pow(2).mean()
        p_norm = torch.norm(p, p=2, dim=1).pow(2).mean()
        o_norm = torch.norm(o, p=2, dim=1).pow(2).mean()
        
        return (s_norm + p_norm + o_norm) / 3.0


        
