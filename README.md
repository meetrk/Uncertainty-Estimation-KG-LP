# Uncertainty Estimation for Knowledge Graph Link Prediction

A PyTorch-based implementation for link prediction on knowledge graphs with uncertainty estimation methods. This project is part of thesis work aimed at developing robust uncertainty quantification techniques for knowledge graph completion tasks.

## Overview

Link prediction in knowledge graphs is a fundamental task that predicts missing relations between entities. However, understanding the confidence and uncertainty of these predictions is crucial for real-world applications. This project implements:

- **Encoder**: Relational Graph Convolutional Network (RGCN) for learning entity representations
- **Decoders**: DistMult and TransE for scoring candidate triples
- **Uncertainty Methods** (planned): Monte Carlo Dropout and Deep Ensemble techniques

## Features

- PyTorch and PyTorch Geometric based implementation
- Support for standard benchmark datasets (WN18RR, FB15k-237)
- Modular architecture with configurable encoders and decoders
- TensorBoard integration for training visualization
- Automatic checkpoint saving during training
- Comprehensive evaluation metrics (Link Prediction + AUC/Loss)
- YAML-based configuration system

## Project Structure

```
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── config/                      # Configuration files
│   ├── lp_fb15k237.yaml        # FB15k-237 dataset config
│   └── lp_wn18rr.yaml          # WN18RR dataset config
├── dataset/                     # Dataset files
│   ├── fb15k237/               # FB15k-237 dataset
│   └── wn18rr/                 # WN18RR dataset
├── model/                       # Model implementations
│   ├── encoder/                # Graph encoders (RGCN)
│   │   ├── model.py           # RGCN encoder implementation
│   │   └── layer.py           # RGCN layer implementation
│   ├── decoder/                # Link prediction decoders
│   │   ├── distmult.py        # DistMult decoder
│   │   ├── transe.py          # TransE decoder
│   │   └── kgemodel.py        # Base KGE model
│   └── trainer/                # Training pipeline
│       └── pipeline.py         # Training and evaluation pipeline
├── utils/                       # Utility functions
│   ├── config_loader.py        # Configuration loading
│   ├── dataset_loader.py       # Dataset loading and processing
│   ├── evaluation.py           # Evaluation metrics
│   ├── initialiser.py          # Weight initialization
│   └── utils.py                # General utilities
├── checkpoints/                 # Saved model checkpoints
└── runs/                        # TensorBoard logs
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/meetrk/Uncertainty-Estimation-KG-LP.git
cd Uncertainty-Estimation-KG-LP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run training with a configuration file:

```bash
# Train on WN18RR dataset
python main.py --config config/lp_wn18rr.yaml

# Train on FB15k-237 dataset
python main.py --config config/lp_fb15k237.yaml
```

### Configuration

Configuration files are in YAML format and control all aspects of training:

**Example configuration (`config/lp_wn18rr.yaml`):**
```yaml
dataset:
  name: WN18RR
  path: ./dataset/wn18rr

model:
  encoder:
    type: RGCN
    hidden_layer_size: 200
    embedding_dim: 200
    num_layers: 2
    dropout: 0.2
    num_bases: 4
    w_init: schlichtkrull-normal
    
  decoder:
    type: DistMult  # or TransE
    l2_penalty_type: schlichtkrull-l2
    l2_penalty: 0
  
training:
  epochs: 10
  sampling:
    batch_size: 16384
    method: edge-neighborhood
    negative_sampling_ratio: 1
    head_corrupt_prob: 0.5
```

### Logging Options

Set logging verbosity:
```bash
python main.py --config config/lp_wn18rr.yaml --log-level DEBUG
```

Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`

### Monitoring Training

Training metrics are logged to TensorBoard:
```bash
tensorboard --logdir runs/
```

Open http://localhost:6006 in your browser to view training curves, losses, and evaluation metrics.

## Pipeline

The training pipeline consists of the following stages:

1. **Data Loading**: Load knowledge graph datasets (WN18RR or FB15k-237)
2. **Model Initialization**: 
   - RGCN encoder for learning entity embeddings
   - DistMult/TransE decoder for scoring triples
3. **Training Loop**:
   - Batch sampling with negative sampling
   - Forward pass through encoder and decoder
   - Loss computation and backpropagation
   - Checkpoint saving at specified intervals
4. **Evaluation**:
   - **Link Prediction Metrics**: MRR, Hits@1, Hits@3, Hits@10
   - **Classification Metrics**: Evaluation loss and AUC score
5. **Logging**: 
   - Console logging with configurable verbosity
   - TensorBoard visualization
   - Training history saved to file

## Datasets

### WN18RR
- **Entities**: 40,943
- **Relations**: 11
- **Training triples**: 86,835
- **Validation triples**: 3,034
- **Test triples**: 3,134

### FB15k-237
- **Entities**: 14,541
- **Relations**: 237
- **Training triples**: 272,115
- **Validation triples**: 17,535
- **Test triples**: 20,466

Both datasets are subsets of larger knowledge graphs with inverse relations removed to prevent trivial predictions.

## Model Architecture

### RGCN Encoder
- Relational Graph Convolutional Network with basis decomposition
- 2-layer architecture with configurable hidden dimensions
- Dropout for regularization
- Xavier initialization with optional custom initialization schemes

### Decoders

**DistMult**: 
- Bilinear scoring function
- Simple and effective for symmetric relations

**TransE**: 
- Translation-based scoring
- Effective for hierarchical relations

## Evaluation Metrics

The project implements two types of evaluation:

### Link Prediction Metrics
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks
- **Hits@K**: Percentage of correct entities in top K predictions
  - Hits@1, Hits@3, Hits@10

### Classification Metrics
- **AUC (Area Under ROC Curve)**: Binary classification performance
- **Evaluation Loss**: Cross-entropy loss on validation set

## Known Issues

**Current Challenge**: The model achieves good AUC scores but shows poor performance on link prediction metrics (MRR, Hits@K). This discrepancy suggests:
- The model performs well at binary classification (edge exists vs. not)
- But struggles with ranking correct entities among all candidates
- Potential issues with negative sampling strategy or training pipeline calibration

## Future Work

The following features and improvements are planned:

- [ ] **Investigate and resolve link prediction issues**
  - Analyze ranking behavior
  - Experiment with different negative sampling strategies
  - Tune hyperparameters for better ranking performance

- [ ] **Implement Uncertainty Estimation Methods**:
  - [ ] **Monte Carlo Dropout**: Use dropout at inference time to estimate epistemic uncertainty
  - [ ] **Deep Ensembles**: Train multiple models with different initializations to quantify prediction uncertainty

- [ ] **Additional Evaluation**:
  - Calibration metrics (ECE, MCE)
  - Uncertainty quality metrics
  - Out-of-distribution detection


## Checkpoints

Model checkpoints are automatically saved during training in the `checkpoints/` directory. Checkpoints include:
- Model state dictionary
- Optimizer state
- Training epoch
- Configuration

Load a checkpoint for inference or resume training.

## Logging

Logs are written to:
- **Console**: Real-time training progress
- **training.log**: Persistent file logging
- **TensorBoard**: Visual metrics and training curves in `runs/` directory

## Contributing

This is thesis work in active development. Suggestions and discussions are welcome via emails.

## License

See [LICENSE](LICENSE) file for details.


## References

- [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (RGCN)
- [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575) (DistMult)
- [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) (TransE)

## Contact

Email - meet.kachhadiya@tum.de