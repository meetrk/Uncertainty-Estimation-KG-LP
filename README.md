# Uncertainty Estimation for Knowledge Graph Link Prediction

A PyTorch implementation for link prediction on knowledge graphs with uncertainty estimation (Monte Carlo Dropout, Deep Ensembles) and calibration methods (Temperature Scaling, Isotonic Regression, Platt Scaling).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/meetrk/Uncertainty-Estimation-KG-LP.git
cd Uncertainty-Estimation-KG-LP
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run training with a configuration file:
```bash
python main.py --config config/lp_wn18rr.yaml
python main.py --config config/lp_wn18rr_ensemble.yaml  # For ensemble
```

## Configuration

All hyperparameters are configured via YAML files in `config/`. Edit these to customize:

**Key Options:**
- **Dataset**: `WN18RR` or `FB15k-237`
- **Model**: RGCN encoder + DistMult/TransE decoder
- **Uncertainty**: 
  - Single model: Set `calibration.type: mc_dropout`
  - Ensemble: Set `ensemble.enabled: true`
- **Calibration**: Choose `temperature_scaling`, `isotonic_regression`, or `platt_scaling`
- **Training**: Learning rate, batch size, epochs, early stopping, edge dropout, label smoothing

**Full Example Config with Options:**
```yaml
dataset:
  name: WN18RR                    # Dataset: WN18RR or FB15k-237
  path: ./dataset/wn18rr          # Dataset path

model:
  encoder:
    type: RGCN                    # Encoder: RGCN
    hidden_layer_size: 500        # Hidden layer dimension
    embedding_dim: 500            # Entity embedding dimension
    dropout: 0.2                  # Dropout probability (0-1)
    num_bases: 5                  # Number of basis functions for decomposition
    bases_enabled: true           # Use basis decomposition (true/false)
  
  decoder:
    type: DistMult                # Decoder: DistMult or TransE
    l2_penalty: 0.001             # L2 regularization strength
    w_gain: false                 # Weight gain initialization
    b_init: false                 # Bias initialization

ensemble:
  enabled: false                  # Enable Deep Ensemble (true/false)
  num_models: 5                   # Number of ensemble members (if enabled)

training:
  epochs: 10000                   # Number of training epochs
  
  sampling:
    negative_sampling_ratio: 3    # Negative samples per positive
    edge_dropout: 0.2             # Edge dropout during training (0-1)
  
  optimiser:
    learning_rate: 0.01           # Adam learning rate
    weight_decay: 0               # L2 weight decay
  
  evaluation_frequency: 100       # Evaluate every N epochs
  
  early_stopping:
    enabled: true                 # Enable early stopping
    patience: 10                  # Stop if no improvement for N evaluations
    delta: 0.001                  # Minimum improvement threshold
  
  load_model: false               # Load checkpoint instead of training
  save_model: true                # Save checkpoints during training
  checkpoint_path: ./checkpoints/model.pth  # Path to save/load checkpoint
  test: true                      # Evaluate on test set after training
  
  label_smoothing:
    positive: 0.9                 # Label smoothing for positive samples
    negative: 0.05                # Label smoothing for negative samples

calibration:
  enabled: true                   # Apply post-hoc calibration
  type: standard                  # Calibration type: standard or mc_dropout or ensemble
  mc_samples: 5                   # MC forward passes (for mc_dropout)
  method: isotonic_regression     # temperature_scaling, isotonic_regression, or platt_scaling
  max_iters: 1000                 # Max calibration iterations
  learning_rate: 0.01             # Calibration learning rate
```

## Logging

Set log level and view training progress:
```bash
python main.py --config config/lp_wn18rr.yaml --log-level DEBUG

# Monitor with TensorBoard
tensorboard --logdir runs/
```

## Project Structure

```
├── main.py                 # Entry point
├── config/                 # Configuration files (YAML)
├── model/
│   ├── encoder/           # RGCN encoder
│   ├── decoder/           # DistMult, TransE decoders
│   ├── ensemble/          # Deep Ensemble
│   ├── calibrator/        # Calibration methods
│   └── trainer/           # Training pipelines
├── utils/                 # Config loader, evaluation, utilities
└── misc/                  # Dataset handling
```

## Features

- **Encoder**: RGCN with basis decomposition
- **Decoders**: DistMult, TransE
- **Uncertainty**: Monte Carlo Dropout + Deep Ensembles
- **Calibration**: Temperature Scaling, Isotonic Regression, Platt Scaling
- **Metrics**: MRR, Hits@K, ECE, ACE, Brier Score
- **Training**: Early stopping, edge dropout, label smoothing, gradient clipping, TensorBoard logging

## Datasets

- **WN18RR**: 40K entities, 11 relations, 86K training triples
- **FB15k-237**: 14K entities, 237 relations, 272K training triples

## References

- **RGCN**: Schlichtkrull et al., [Modeling Relational Data with GCNs](https://arxiv.org/abs/1703.06103)
- **Uncertainty**: Gal & Ghahramani, [Dropout as Bayesian Approximation](https://arxiv.org/abs/1506.02142); Lakshminarayanan et al., [Deep Ensembles](https://arxiv.org/abs/1612.01474)


## License

Apache 2.0 - See [LICENSE](LICENSE)