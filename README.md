# Uncertainty Estimation for Knowledge Graph Link Prediction

A PyTorch-based implementation for link prediction on knowledge graphs with uncertainty estimation and calibration methods. This project implements robust uncertainty quantification techniques for knowledge graph completion tasks, including Monte Carlo Dropout, Deep Ensembles, and post-hoc calibration.

## Overview

Link prediction in knowledge graphs is a fundamental task that predicts missing relations between entities. However, understanding the confidence and uncertainty of these predictions is crucial for real-world applications. This project implements:

- **Encoder**: Relational Graph Convolutional Network (RGCN) for learning entity representations
- **Decoders**: DistMult and TransE for scoring candidate triples
- **Uncertainty Methods**: 
  - **Monte Carlo Dropout**: Epistemic uncertainty via stochastic forward passes
  - **Deep Ensembles**: Uncertainty quantification through model diversity
- **Calibration Methods**: Temperature Scaling, Isotonic Regression, and Platt Scaling

## Features

- PyTorch and PyTorch Geometric based implementation
- Support for standard benchmark datasets (WN18RR, FB15k-237)
- Modular architecture with configurable encoders and decoders
- **Uncertainty Estimation Methods**:
  - Monte Carlo Dropout with configurable sampling
  - Deep Ensemble with multiple independent models
- **Calibration Methods**:
  - Temperature Scaling (scalar and MC-Dropout variants)
  - Isotonic Regression
  - Platt Scaling (input-dependent neural network)
- **Comprehensive Evaluation Metrics**:
  - Link Prediction: MRR, Mean Rank, Hits@1/3/10
  - Calibration: ECE, ACE, Brier Score
  - Uncertainty: Predictive variance, reliability diagrams
- **Training Features**:
  - Early stopping with patience
  - Edge dropout regularization
  - Label smoothing
  - Gradient clipping
  - TensorBoard integration
  - Automatic checkpoint saving
- YAML-based configuration system

## Project Structure

```
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── config/                      # Configuration files
│   ├── lp_fb15k237.yaml        # FB15k-237 dataset config
│   ├── lp_wn18rr.yaml          # WN18RR dataset config
│   ├── lp_fb15k237_ensemble.yaml  # FB15k-237 ensemble config
│   └── lp_wn18rr_ensemble.yaml    # WN18RR ensemble config
├── data/                        # Dataset storage (auto-downloaded)
│   └── RLPD/                   # RelLinkPredDataset cache
├── model/                       # Model implementations
│   ├── encoder/                # Graph encoders
│   │   ├── model.py           # RGCN encoder implementation
│   │   └── layer.py           # RGCN layer implementation
│   ├── decoder/                # Link prediction decoders
│   │   ├── distmult.py        # DistMult decoder
│   │   ├── transe.py          # TransE decoder
│   │   └── kgemodel.py        # Base KGE model
│   ├── ensemble/               # Uncertainty estimation
│   │   └── deep_ensemble.py   # Deep Ensemble implementation
│   ├── calibrator/             # Calibration methods
│   │   ├── tempscaling.py     # Temperature Scaling
│   │   ├── isotonic.py        # Isotonic Regression
│   │   └── plattscaling.py    # Platt Scaling
│   └── trainer/                # Training pipeline
│       ├── pipeline.py         # Single model training
│       ├── ensemble_pipeline.py # Ensemble training
│       └── basepipeline.py     # Base pipeline class
├── utils/                       # Utility functions
│   ├── config_loader.py        # Configuration loading
│   ├── evaluation.py           # Evaluation metrics & uncertainty
│   ├── initialiser.py          # Weight initialization
│   └── utils.py                # General utilities
├── misc/                        # Miscellaneous utilities
│   └── rel_link_pred_dataset.py # Dataset wrapper
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
# Train single model on WN18RR dataset
python main.py --config config/lp_wn18rr.yaml

# Train single model on FB15k-237 dataset
python main.py --config config/lp_fb15k237.yaml

# Train Deep Ensemble on WN18RR
python main.py --config config/lp_wn18rr_ensemble.yaml

# Train Deep Ensemble on FB15k-237
python main.py --config config/lp_fb15k237_ensemble.yaml
```

### Uncertainty Evaluation

To evaluate uncertainty on a trained model, set `load_model: true` in the config and specify the checkpoint path:

```yaml
training:
  load_model: true
  checkpoint_path: ./checkpoints/model_epoch_1000.pth
  test: true
```

Then run:
```bash
python main.py --config config/lp_wn18rr.yaml
```

This will load the model, apply calibration (if enabled), and compute uncertainty metrics on the test set.

### Configuration

Configuration files are in YAML format and control all aspects of training, uncertainty estimation, and calibration.

**Example configuration for single model (`config/lp_wn18rr.yaml`):**
```yaml
dataset:
  name: WN18RR
  path: ./dataset/wn18rr

model:
  encoder:
    type: RGCN
    hidden_layer_size: 500
    embedding_dim: 500
    dropout: 0.2
    num_bases: 5
    bases_enabled: true
    
  decoder:
    type: DistMult  # or TransE
    l2_penalty: 0.001
    w_gain: false
    b_init: false

ensemble:
  enabled: false  # Set to true for Deep Ensemble
  num_models: 5   # Number of ensemble members
  
training:
  epochs: 10000
  sampling:
    negative_sampling_ratio: 3
    edge_dropout: 0.2
  optimiser:
    learning_rate: 0.01
    weight_decay: 0
  evaluation_frequency: 100
  early_stopping:
    enabled: true
    patience: 10
    delta: 0.001
  load_model: false  # Set to true to load checkpoint
  save_model: true
  checkpoint_path: ./checkpoints/model.pth
  test: true
  label_smoothing:
    positive: 0.9  # Smooth positive labels
    negative: 0.05 # Smooth negative labels

calibration:
  enabled: true
  type: mc_dropout  # or 'ensemble' for Deep Ensemble
  mc_samples: 5     # Number of MC forward passes
  method: isotonic_regression  # or 'temperature_scaling', 'platt_scaling'
  max_iters: 1000
  learning_rate: 0.01
```

**Key Configuration Options:**

- **Ensemble**: Enable Deep Ensemble by setting `ensemble.enabled: true`
- **MC Dropout**: Use with single model by setting `calibration.type: mc_dropout`
- **Calibration Method**: Choose from `temperature_scaling`, `isotonic_regression`, or `platt_scaling`
- **Edge Dropout**: Regularization during training via `training.sampling.edge_dropout`
- **Label Smoothing**: Prevents overconfident predictions

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

The training pipeline supports both single model and ensemble training:

### Single Model Training Pipeline

1. **Data Loading**: Load knowledge graph datasets (WN18RR or FB15k-237)
2. **Model Initialization**: 
   - RGCN encoder for learning entity embeddings
   - DistMult/TransE decoder for scoring triples
3. **Training Loop**:
   - Negative sampling with configurable ratio
   - Edge dropout for regularization
   - Label smoothing for calibration
   - Forward pass through encoder and decoder
   - Loss computation and backpropagation with gradient clipping
   - Early stopping with patience mechanism
   - Checkpoint saving at specified intervals
4. **Evaluation**:
   - **Link Prediction Metrics**: MRR, Mean Rank, Hits@1/3/10
   - **Calibration Metrics**: ECE, ACE, Brier Score
5. **Uncertainty Estimation** (optional):
   - Monte Carlo Dropout: Multiple stochastic forward passes
   - Predictive variance computation
6. **Calibration** (optional):
   - Post-hoc calibration using validation set
   - Temperature Scaling, Isotonic Regression, or Platt Scaling
7. **Logging**: 
   - Console logging with configurable verbosity
   - TensorBoard visualization
   - Training history saved to file

### Deep Ensemble Training Pipeline

1. **Ensemble Initialization**: Create N independent models with different random seeds
2. **Parallel Training**: Train each model independently
3. **Diversity Tracking**: Monitor prediction variance across ensemble members
4. **Aggregation**: 
   - Mean prediction across ensemble
   - Uncertainty via variance of predictions
5. **Ensemble Evaluation**: Compute metrics using aggregated predictions
6. **Calibration** (optional): Calibrate ensemble predictions

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
- Dropout for regularization (can be enabled at inference for MC Dropout)
- Support for basis decomposition to reduce parameters
- Xavier initialization with optional custom initialization schemes

### Decoders

**DistMult**: 
- Bilinear scoring function: `score(h, r, t) = h^T diag(r) t`
- Simple and effective for symmetric relations
- Optional L2 regularization

**TransE**: 
- Translation-based scoring: `score(h, r, t) = -||h + r - t||`
- Effective for hierarchical relations

Both decoders support optional calibration layers for improved confidence estimates.

### Uncertainty Methods

**Monte Carlo Dropout**:
- Enable dropout during inference
- Perform multiple stochastic forward passes (default: 5)
- Compute mean and variance of predictions
- Epistemic uncertainty estimation

**Deep Ensemble**:
- Train N independent models (default: 5)
- Different random initializations for diversity
- Aggregate predictions via mean and variance
- Captures both epistemic and aleatoric uncertainty

### Calibration Methods

Applied post-training to improve probability estimates:

**Temperature Scaling**:
- Single scalar parameter to rescale logits
- Preserves ranking while improving calibration
- Variants: Standard and MC-Dropout aware

**Isotonic Regression**:
- Non-parametric calibration
- Learns monotonic mapping from scores to probabilities
- Requires validation data

**Platt Scaling**:
- Learns input-dependent temperature
- Neural network-based approach
- More flexible than temperature scaling

## Evaluation Metrics

The project implements comprehensive evaluation across multiple dimensions:

### Link Prediction Metrics
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of correct entities
- **Mean Rank**: Average rank of correct entities
- **Hits@K**: Percentage of correct entities in top K predictions
  - Hits@1, Hits@3, Hits@10

### Calibration Metrics
- **ECE (Expected Calibration Error)**: Measures calibration quality using binning
- **ACE (Adaptive Calibration Error)**: Adaptive binning version of ECE
- **Brier Score**: Mean squared difference between predicted probabilities and true labels
- **Reliability Diagrams**: Visual calibration assessment

### Uncertainty Metrics
- **Predictive Variance**: Variance of predictions across MC samples or ensemble members
- **Epistemic Uncertainty**: Captured via MC Dropout or ensemble disagreement
- **Calibration Curves**: Relationship between predicted confidence and actual accuracy

## Known Issues and Limitations

**Link Prediction Performance**: While the model achieves good calibration metrics, there is ongoing work to improve link prediction metrics (MRR, Hits@K). This discrepancy suggests:
- The model performs well at probability calibration
- But there's room for improvement in entity ranking
- Ongoing investigation into negative sampling strategies and training dynamics

**Computational Cost**: Deep Ensemble and MC Dropout require multiple forward passes, increasing inference time proportionally to the number of samples/models.

## Future Work

The following enhancements are planned:

- [ ] **Performance Improvements**:
  - Further optimize link prediction metrics
  - Experiment with alternative negative sampling strategies
  - Hyperparameter tuning for better ranking performance

- [ ] **Additional Uncertainty Methods**:
  - Evidential deep learning
  - Bayesian neural networks
  - Test-time augmentation

- [ ] **Evaluation Extensions**:
  - Out-of-distribution detection experiments
  - Uncertainty-aware metrics
  - Selective prediction based on uncertainty
  - Adversarial robustness evaluation

- [ ] **Scalability**:
  - Multi-GPU training for large ensembles
  - Efficient uncertainty estimation for large graphs
  - Incremental learning support


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

This is active research work. Suggestions, discussions, and contributions are welcome via issues and pull requests.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.


## References

### Knowledge Graph Embedding Models
- [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) - Schlichtkrull et al., 2018 (RGCN)
- [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575) - Yang et al., 2015 (DistMult)
- [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) - Bordes et al., 2013 (TransE)

### Uncertainty Estimation
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) - Gal & Ghahramani, 2016
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474) - Lakshminarayanan et al., 2017

### Calibration
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) - Guo et al., 2017 (Temperature Scaling)
- [Obtaining Well Calibrated Probabilities Using Bayesian Binning](https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf) - Naeini et al., 2015

## Contact

Email - meet.kachhadiya@tum.de