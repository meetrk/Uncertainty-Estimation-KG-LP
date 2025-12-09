from torch_geometric.data import Data
import torch
from utils.utils import edge_neighborhood

class LinkSplitter:

    def __init__(self, data: Data, disjoint_train_ratio: float) :
        self.data = data
        self.disjoint_train_ratio = disjoint_train_ratio
        self.train_data = Data()
        self.val_data = Data()
        self.test_data = Data()

        if self.data.get('train_mask') is None or self.data.get('val_mask') is None or self.data.get('test_mask') is None:
            size = self.data.edge_index.size(1)
            indices = torch.randperm(size)
            train_cutoff = int(0.8 * size)
            val_cutoff = int(0.9 * size)
            train_mask = indices < train_cutoff
            val_mask = (indices >= train_cutoff) & (indices < val_cutoff)
            test_mask = indices >= val_cutoff
            self.data.train_mask = train_mask
            self.data.val_mask = val_mask
            self.data.test_mask = test_mask

        self.train_data.edge_index = self.data.edge_index[:,self.data.train_mask]
        self.train_data.edge_type = self.data.edge_type[self.data.train_mask]
        self.train_data.num_relations = self.data.num_relations
        self.train_data.num_nodes = self.train_data.num_nodes

        self.val_data.edge_index = self.data.edge_index[:,self.data.val_mask]
        self.val_data.edge_type = self.data.edge_type[self.data.val_mask]
        self.val_data.num_relations = self.data.num_relations
        self.val_data.num_nodes = self.val_data.num_nodes

        self.test_data.edge_index = self.data.edge_index[:,self.data.test_mask]
        self.test_data.edge_type = self.data.edge_type[self.data.test_mask]
        self.test_data.num_relations = self.data.num_relations
        self.test_data.num_nodes = self.test_data.num_nodes
        

    def add_edges(
        self,
        data
    ):

        data = self._split_mp_and_supervision(data)
        data = self._add_reverse_edges(data)
        data = self._self_loop_edges(data)
        return data
    
    def _add_reverse_edges(
        self,
        data: Data,
        mode: str = "train"
    ):
        edge_index = data.edge_index 
        rev_edge_index = torch.flip(edge_index,[0])
        data.edge_index = torch.concat([edge_index,rev_edge_index],dim=1)
        rev_edge_type = data.edge_type + self.data.num_relations
        data.edge_type = torch.concat([data.edge_type,rev_edge_type],dim=0)

        if mode == "train":
            data.edge_label_index = torch.concat([data.edge_label_index,torch.flip(data.edge_label_index,[0])],dim=1)
            data.edge_label_type = torch.concat([data.edge_label_type,data.edge_label_type + self.data.num_relations],dim=0)

        data.num_relations = len(data.edge_type.unique())
       
        return data

    def _self_loop_edges(
        self,
        data: Data
    ):
        edge_index = data.edge_index
        nodes = torch.unique(edge_index)
        self_mask = edge_index[0] == edge_index[1]
        if self_mask.any():
            existing_self_nodes = torch.unique(edge_index[0, self_mask])
        else:
            existing_self_nodes = torch.tensor([], dtype=nodes.dtype, device=nodes.device)

        # nodes that need a self-loop
        if existing_self_nodes.numel() > 0:
            nodes_to_add = nodes[~torch.isin(nodes, existing_self_nodes)]
        else:
            nodes_to_add = nodes

        if nodes_to_add.numel() == 0:
            return data

        loop_edges = torch.stack([nodes_to_add, nodes_to_add], dim=0)
        data.edge_index = torch.cat([edge_index, loop_edges], dim=1)

        device = data.edge_type.device if hasattr(data, "edge_type") else data.edge_index.device
        self_loop_type = torch.full((nodes_to_add.size(0),), data.num_relations, dtype=torch.long, device=device)
        data.edge_type = torch.cat([data.edge_type, self_loop_type], dim=0)
        data.num_relations += 1
        return data


    def _split_mp_and_supervision(
            self, data: Data
        ):
            """
            Splits the edges into message passing and supervision sets following 
            link prediction dropout strategy.
            
            Message Passing: Uses a subset of edges (defined by the ratio).
            Supervision: Uses ALL edges (reconstructing the full graph).
            """

            edge_index = data.edge_index
            edge_type = data.edge_type
            num_edges = edge_index.size(1)

            # 1. Supervision Set: MUST include ALL edges (Positives)
            # We preserve the full graph structure here to train the model 
            # to reconstruct even the edges that are dropped from MP.
            data.edge_label_index = edge_index.clone()
            data.edge_label_type = edge_type.clone()

            # 2. Message Passing Set: Apply Edge Dropout
            # We select a random subset of edges to act as the "context" graph.
            indices = torch.randperm(num_edges)
            
            # Assuming self.disjoint_train_ratio acts as the 'keep_prob' (e.g., 0.5 to keep 50%)
            # If your ratio represents the 'dropout' amount, change '<' to '>'.
            mask = indices < int(self.disjoint_train_ratio * num_edges)

            data.edge_index = edge_index[:, mask]
            data.edge_type = edge_type[mask]

            return data
    

    def generate_batch_triples(
        self,
        num_nodes: int,
        config: dict,
        mode: str = "train",
        sampling: str = "batch",
    ):
        """ Generate batch triples from data """
        
        # 1. Determine source data based on mode
        if mode == "train":
            edge_index = self.train_data.edge_index
            edge_type = self.train_data.edge_type
            batch_size = config['sampling']['batch_size']
        elif mode == "val":
            edge_index = self.val_data.edge_index
            edge_type = self.val_data.edge_type
            sampling = "full"
        elif mode == "test":
            edge_index = self.test_data.edge_index
            edge_type = self.test_data.edge_type
            sampling = "full"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 2. CRITICAL: Evaluation Mode
        if mode == "val":
            batch = Data()
            batch.edge_label_index = edge_index
            batch.edge_label_type = edge_type
            
            # Context: Training Graph
            # Must augment this with inverses/self-loops so it matches training topology
            context_data = Data(
                edge_index=self.train_data.edge_index, 
                edge_type=self.train_data.edge_type,
                num_nodes=num_nodes # Pass num_nodes to ensure self-loops are correct
            )
            # Temporarily set num_relations for the helper functions
            context_data.num_relations = self.data.num_relations

            context_data = self._add_reverse_edges(context_data, mode="val")
            context_data = self._self_loop_edges(context_data)
            
            batch.edge_index = context_data.edge_index
            batch.edge_type = context_data.edge_type
            return batch

        if mode == "test":
            batch = Data()
            batch.edge_label_index = edge_index
            batch.edge_label_type = edge_type
            
            # Context: Training + Validation Graph
            # FIX: Concatenate along dim=1 for edge_index
            combined_edge_index = torch.cat([self.train_data.edge_index, self.val_data.edge_index], dim=1)
            combined_edge_type = torch.cat([self.train_data.edge_type, self.val_data.edge_type], dim=0)
            
            context_data = Data(
                edge_index=combined_edge_index,
                edge_type=combined_edge_type,
                num_nodes=num_nodes
            )
            context_data.num_relations = self.data.num_relations

            context_data = self._add_reverse_edges(context_data, mode="test")
            context_data = self._self_loop_edges(context_data)

            batch.edge_index = context_data.edge_index
            batch.edge_type = context_data.edge_type
            return batch

        # 3. Training Mode
        triples = torch.stack([edge_index[0], edge_type, edge_index[1]], dim=1)

        if sampling == "edge-neighborhood":
            if batch_size > triples.size(0):
                batch_size = triples.size(0)
                
            indices = edge_neighborhood(
                edge_index=edge_index, 
                edge_type=edge_type, 
                sample_size=batch_size, 
                num_nodes=num_nodes, 
                return_indices=True
            )
            
            batch = Data()
            batch.edge_index = torch.stack([triples[indices][:, 0], triples[indices][:, 2]], dim=0)
            batch.edge_type = triples[indices][:, 1]
            batch = self.add_edges(batch)

        elif sampling == "sample":
            sample_size = config['sampling'].get('sample_size', batch_size)
            indices = torch.randperm(triples.size(0))[:sample_size]
            batch = Data()
            batch.edge_index = torch.stack([triples[indices][:, 0], triples[indices][:, 2]], dim=0)
            batch.edge_type = triples[indices][:, 1]
            batch = self.add_edges(batch)

        elif sampling == "full":
            batch = Data()
            batch.edge_index = edge_index
            batch.edge_type = edge_type
            batch = self.add_edges(batch)

        else:
            raise ValueError(f"Invalid sampling method: {sampling}")

        return batch