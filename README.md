# Knowledge Graph Link Prediction

A PyTorch-based implementation for link prediction on knowledge graphs using graph neural networks.

## Project Structure

```
├── main.py                 # Main entry point
├── config/                 # Configuration files
│   ├── lp_fb15k237.yaml   # FB15k-237 dataset config
│   └── lp_wn18rr.yaml     # WN18RR dataset config
├── dataset/               # Dataset files
│   ├── fb15k237/         # FB15k-237 dataset
│   └── wn18rr/           # WN18RR dataset
├── model/                # Model implementations
│   ├── encoder/          # Graph encoders
│   └── decoder/          # Link prediction decoders
├── utils/                # Utility functions
│   ├── config_loader.py  # Configuration loading
│   └── dataset_loader.py # Dataset loading and processing
└── misc/                 # Miscellaneous scripts
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py --config config/lp_fb15k237.yaml
```

## Datasets

This project supports two standard knowledge graph datasets:
- **FB15k-237**: A subset of Freebase with inverse relations removed
- **WN18RR**: A subset of WordNet with inverse relations removed

## Configuration

Configuration files are in YAML format. See `config/` directory for examples.

## Development

- Use `black` for code formatting
- Use `mypy` for type checking
- Run tests with `pytest`