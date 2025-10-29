import os
import numpy as np
from torch_geometric.data import Data
import torch
from typing import Dict, Tuple, Any
from pathlib import Path



def read_triplets(file_path: str, entity2id: Dict[str, int], relation2id: Dict[str, int]) -> np.ndarray:
    """
    Read triplets from a file and convert to numerical format.
    
    Args:
        file_path: Path to the triplets file
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        
    Returns:
        NumPy array of triplets in (head_id, relation_id, tail_id) format
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If entity or relation not found in mappings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Triplets file not found: {file_path}")
    
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f"Warning: Line {line_num} in {file_path} has {len(parts)} parts, expected 3")
                    continue
                    
                head, relation, tail = parts
                triplets.append((
                    entity2id[head], 
                    relation2id[relation], 
                    entity2id[tail]
                ))
            except KeyError as e:
                print(f"Warning: Unknown entity/relation at line {line_num}: {e}")
                continue

    return np.array(triplets)


def load_dataset(dataset_config: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load knowledge graph dataset from files.
    
    Args:
        dataset_config: Configuration dictionary containing dataset path
        
    Returns:
        Tuple of (entity2id, relation2id, train_triplets, valid_triplets, test_triplets)
        
    Raises:
        FileNotFoundError: If dataset files are missing
        ValueError: If dataset configuration is invalid
    """
    if 'path' not in dataset_config:
        raise ValueError("Dataset configuration must contain 'path' key")
    
    file_path = Path(dataset_config['path'])
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    print(f"Loading data from {file_path}")

    # Load entities
    entities_file = file_path / 'entities.dict'
    if not entities_file.exists():
        raise FileNotFoundError(f"Entities file not found: {entities_file}")
        
    entity2id = {}
    with open(entities_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                eid, entity = parts
                entity2id[entity] = int(eid)

    # Load relations
    relations_file = file_path / 'relations.dict'
    if not relations_file.exists():
        raise FileNotFoundError(f"Relations file not found: {relations_file}")
        
    relation2id = {}
    with open(relations_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                rid, relation = parts
                relation2id[relation] = int(rid)

    # Load triplets
    train_triplets = read_triplets(str(file_path / 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(str(file_path / 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(str(file_path / 'test.txt'), entity2id, relation2id)

    # Print statistics
    print(f'Number of entities: {len(entity2id)}')
    print(f'Number of relations: {len(relation2id)}')
    print(f'Number of train triples: {len(train_triplets)}')
    print(f'Number of valid triples: {len(valid_triplets)}')
    print(f'Number of test triples: {len(test_triplets)}')

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets



def generate_data(entity2id: Dict[str, int], relation2id: Dict[str, int], 
                 train_triplets: np.ndarray, valid_triplets: np.ndarray, 
                 test_triplets: np.ndarray) -> Data:
    """
    Generate PyTorch Geometric Data object from triplets.
    
    Args:
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        train_triplets: Training triplets
        valid_triplets: Validation triplets
        test_triplets: Test triplets
        
    Returns:
        PyTorch Geometric Data object
    """
    num_entities = len(entity2id)
    num_relations = len(relation2id)

    # Create edge_index and edge_type from training triplets
    edge_index = torch.tensor(train_triplets[:, [0, 2]].T, dtype=torch.long)
    edge_type = torch.tensor(train_triplets[:, 1], dtype=torch.long)

    data = Data(
        num_nodes=num_entities, 
        edge_index=edge_index, 
        edge_type=edge_type,
        num_relations=num_relations,
        train_triplets=torch.tensor(train_triplets, dtype=torch.long),
        valid_triplets=torch.tensor(valid_triplets, dtype=torch.long),
        test_triplets=torch.tensor(test_triplets, dtype=torch.long)
    )

    return data