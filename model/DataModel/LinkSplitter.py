from torch_geometric.data import Data
from torch import Tensor
import torch

class LinkSplitter:

    def __init__(self, data: Data, disjoint_train_ratio: float) :
        self.data = data
        self.disjoint_train_ratio = disjoint_train_ratio
        self.train_data = Data()
        self.val_data = Data()
        self.test_data = Data()

        self.split()

        
    def split(
        self,
    ):
        self.train_data.edge_index = self.data.edge_index[:,self.data.train_mask]
        self.train_data.edge_type = self.data.edge_type[self.data.train_mask]
        self.train_data.num_relations = self.data.num_relations
        self.train_data.num_nodes = self.data.num_nodes
        self.val_data.edge_index = self.data.edge_index[:,self.data.val_mask]
        self.val_data.edge_type = self.data.edge_type[self.data.val_mask]
        self.val_data.num_relations = self.data.num_relations
        self.val_data.num_nodes = self.data.num_nodes

        self.test_data.edge_index = self.data.edge_index[:,self.data.test_mask]
        self.test_data.edge_type = self.data.edge_type[self.data.test_mask]
        self.test_data.num_relations = self.data.num_relations
        self.test_data.num_nodes = self.data.num_nodes

        self.train_data = self._split_mp_and_supervision(self.train_data)
        self.val_data_data = self._split_mp_and_supervision(self.val_data)
        self.test_data = self._split_mp_and_supervision(self.test_data)

        self.train_data = self._add_reverse_edges(self.train_data)
        self.val_data = self._add_reverse_edges(self.val_data)
        self.test_data = self._add_reverse_edges(self.test_data)

        self.train_data = self._self_loop_edges(self.train_data)
        self.val_data = self._self_loop_edges(self.val_data)
        self.test_data = self._self_loop_edges(self.test_data)

        return self.train_data, self.val_data, self.test_data


    def _add_reverse_edges(
        self,
        data: Data
    ):
        edge_index = data.edge_index 
        rev_edge_index = torch.flip(edge_index,[0])
        data.edge_index = torch.concat([edge_index,rev_edge_index],dim=1)
        rev_edge_type = data.edge_type + data.num_relations
        data.edge_type = torch.concat([data.edge_type,rev_edge_type],dim=0)

        data.edge_label_index = torch.concat([data.edge_label_index,torch.flip(data.edge_label_index,[0])],dim=1)
        data.edge_label_type = torch.concat([data.edge_label_type,data.edge_label_type + data.num_relations],dim=0)

        data.num_relations = len(data.edge_type.unique())
       
        return data

    def _self_loop_edges(
        self,
        data: Data
    ):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        loop_edges = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
        data.edge_index = torch.cat([edge_index, loop_edges], dim=1)
        self_loop_type = torch.full((num_nodes,), data.num_relations, dtype=torch.long)
        data.edge_type = torch.cat([data.edge_type, self_loop_type], dim=0)
        data.num_relations += 1
        return data


    def _split_mp_and_supervision(
        self, data: Data
    ):
        """Splits the edges into message passing and supervision sets"""

        edge_index = data.edge_index
        edge_type = data.edge_type
        num_edges = edge_index.size(1)
        indices = torch.randperm(num_edges)
        mask = indices < self.disjoint_train_ratio * num_edges

        data.edge_index = edge_index[:, mask]
        data.edge_type = edge_type[mask]

        data.edge_label_index = edge_index[:, ~mask]
        data.edge_label_type = edge_type[~mask]

        return data

