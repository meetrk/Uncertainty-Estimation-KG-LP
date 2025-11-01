import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
import math
from torch_geometric.utils import degree, add_self_loops

class RGCNLayer(MessagePassing):
    
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, aggr='add', **kwargs):
        super(RGCNLayer, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        
        # Basis decomposition parameters
        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        
        # Optional root transformation
        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)
        
        # Optional bias
        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Glorot/Xavier-style initialization for basis and attention
        bound = math.sqrt(6.0 / (self.in_channels + self.out_channels))
        torch.nn.init.uniform_(self.basis, -bound, bound)
        torch.nn.init.uniform_(self.att, -bound, bound)
        
        # Initialize root weight
        if self.root is not None:
            bound_root = math.sqrt(6.0 / (self.in_channels + self.out_channels))
            torch.nn.init.uniform_(self.root, -bound_root, bound_root)
        
        # Initialize bias
        if self.bias is not None:
            bound_bias = 1.0 / math.sqrt(self.out_channels)
            torch.nn.init.uniform_(self.bias, -bound_bias, bound_bias)
    
    def forward(self, x, edge_index, edge_type, size=None):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        
        # Iterate over each relation type
        for i in range(self.num_relations):
            mask = edge_type == i
            edge_index_i = edge_index[:, mask]
            
            # Propagate WITHOUT calling update
            h = self.propagate(edge_index_i, x=x, relation=i, size=None)
            out = out + h
        
        # Add root transformation ONCE after all relations
        if self.root is not None:
            out = out + torch.matmul(x, self.root)
        
        # Add bias ONCE
        if self.bias is not None:
            out = out + self.bias
        
        return out

    def message(self, x_j, relation):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        return torch.matmul(x_j, w[relation])
        
    
    def __repr__(self):
        return '{}({}, {}, num_relations={}, num_bases={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations, self.num_bases)
