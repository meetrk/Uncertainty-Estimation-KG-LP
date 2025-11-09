import torch
import torch.nn as nn
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
import math
from utils.initialiser import schlichtkrull_normal_,schlichtkrull_uniform_
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_geometric.typing import Adj, OptTensor, Tensor,pyg_lib,SparseTensor
from torch_geometric.utils import spmm
import torch_geometric.typing
from torch_geometric import is_compiling


class RGCNLayer(MessagePassing):
    
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, aggr='add', w_init="xavier_uniform", w_gain=False, b_init="zeros", **kwargs):
        super(RGCNLayer, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.w_init = kwargs.get('w_init', "xavier_uniform")
        self.w_gain = kwargs.get('w_gain', False)
        self.b_init = kwargs.get('b_init', "zeros")
        
        # Basis decomposition parameters
        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        
        # Optional root transformation
        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if b_init is not None:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        
        if self.w_gain:
            gain = nn.init.calculate_gain('relu')
        else:
            gain = 1.0

        if self.w_init == 'schlichtkrull-normal':
            schlichtkrull_normal_(self.basis, (self.num_bases, self.in_channels, self.out_channels), gain)
            schlichtkrull_normal_(self.att, (self.num_relations, self.num_bases), gain)
        elif self.w_init == 'schlichtkrull-uniform':
            schlichtkrull_uniform_(self.basis, gain)
            schlichtkrull_uniform_(self.att, gain)
        elif self.w_init == "uniform":
            torch.nn.init.uniform_(self.basis, -gain, gain)
            torch.nn.init.uniform_(self.att, -gain, gain)
        elif self.w_init == "normal":
            torch.nn.init.normal_(self.basis, 0.0, gain)
            torch.nn.init.normal_(self.att, 0.0, gain)
        else:
            torch.nn.init.xavier_uniform_(self.basis, gain=gain)
            torch.nn.init.xavier_uniform_(self.att, gain=gain)
      
        # Initialize root weight
        if self.root is not None:
            bound_root = math.sqrt(6.0 / (self.in_channels + self.out_channels))
            torch.nn.init.uniform_(self.root, -bound_root, bound_root)
        
        # Initialize bias       
        if self.b_init == 'zeros':
            torch.nn.init.zeros_(self.bias)
        elif self.b_init == 'ones':
            torch.nn.init.ones_(self.bias)
        elif self.b_init == 'uniform':
            torch.nn.init.uniform_(self.bias, -0.1, 0.1)
        elif self.b_init == 'normal':
            torch.nn.init.normal_(self.bias, 0.0, 0.1)
        else:
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type, size=None):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        
        weight = (self.att @ self.basis.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels, self.out_channels)
        
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)

            if not torch.is_floating_point(x):
                out = out + self.propagate(
                    tmp,
                    x=weight[i, x] ,
                    edge_type_ptr=None,
                    size=size,
                )
            else:
                h = self.propagate(tmp, x=x, edge_type_ptr=None,
                                   size=size)
                out = out + (h @ weight[i])
        root = self.root
        if root is not None:
            if not torch.is_floating_point(x):
                out = out + root[x]
            else:
                out = out + x @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: torch.Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if (torch_geometric.typing.WITH_SEGMM and not is_compiling()
                and edge_type_ptr is not None):
            # TODO Re-weight according to edge type degree for `aggr=mean`.
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)
        
    
    def __repr__(self):
        return '{}({}, {}, num_relations={}, num_bases={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations, self.num_bases)
