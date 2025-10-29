import os
from collections import defaultdict

# Define paths
dataset_dir = "/Users/meet/Documents/Documents/thesis/untitled folder/CODE/dataset/wn18rr"
output_dir = dataset_dir

# Initialize dictionaries to store unique entities and relations
entities = set()
relations = set()

# Process train, valid, and test files
for split in ['train', 'valid', 'test']:
    file_path = os.path.join(dataset_dir, f"{split}.txt")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head, relation, tail = parts
                    entities.add(head)
                    entities.add(tail)
                    relations.add(relation)

# Sort and create mappings
sorted_entities = sorted(list(entities))
sorted_relations = sorted(list(relations))

# Write entities.dict
with open(os.path.join(output_dir, 'entities.dict'), 'w') as f:
    for idx, entity in enumerate(sorted_entities):
        f.write(f"{idx}\t{entity}\n")

# Write relations.dict
with open(os.path.join(output_dir, 'relations.dict'), 'w') as f:
    for idx, relation in enumerate(sorted_relations):
        f.write(f"{idx}\t{relation}\n")

print(f"Created entities.dict with {len(sorted_entities)} entities")
print(f"Created relations.dict with {len(sorted_relations)} relations")