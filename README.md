# vision2time: Cross-Modal Time Series Classification Benchmark

A benchmarking framework for evaluating cross-modal transfer learning approaches on multivariate time series classification (MTSC) tasks.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/xxx/vision2time.git 
cd vision2time

# Setup environment and install dependencies
make setup

# Check installation status
make status
```

### Usage

```bash
# Run benchmark
make benchmark

# Check results
cat data/results/benchmark_results.csv
```

## Features

- **Multiple Models**: CNN, BiLSTM, Transformer, Vision Transformer
- **Channel Selection**: First-3, Random, PCA strategies for ViT
- **Multi-format Support**: Both `.npy` and `.ts` dataset formats
- **Comprehensive Evaluation**: Accuracy, F1-score, training time metrics
- **Error Handling**: Robust training with gradient clipping and fallbacks

## Project Structure

```
vision2time/
├── main.py                 # Main entry point
├── Makefile               # Build and run commands
├── environment.yaml       # Conda environment
├── src/
│   ├── data/
│   │   └── loader.py      # Data loading utilities
│   ├── models/
│   │   ├── cnn.py         # 1D CNN model
│   │   ├── lstm.py        # Bidirectional LSTM
│   │   ├── transformer.py # Transformer encoder
│   │   └── vit.py         # Vision Transformer adapter
│   ├── training/
│   │   └── trainer.py     # Training utilities
│   └── utils/
│       └── benchmark.py   # Benchmark runner
├── data/
│   ├── raw/               # Input datasets
│   └── results/           # Benchmark results
```

## Make Commands

```bash
# Setup and Installation
make setup          # Create conda environment and install dependencies
make install        # Install package in development mode
make install-pip    # Alternative pip-only installation

# Data and Testing
make status         # Check environment and data status

# Benchmarking
make benchmark      # Run full benchmark

# Utilities
make clean          # Clean build artifacts and results
make clean-env      # Remove conda environment
```

## Models

| Model | Description | Parameters |
|-------|-------------|------------|
| CNN-1D | 1D Convolutional Neural Network | ~250K |
| BiLSTM | Bidirectional LSTM | ~300K |
| Transformer | Transformer Encoder | ~400K |
| ViT | Vision Transformer Adapter | ~1M-80M |

## Channel Selection Strategies

For multivariate time series with >3 channels, ViT uses:

- **First**: Use first 3 channels
- **Random**: Random channel selection (seeded)
- **PCA**: PCA-based dimensionality reduction

## Data Format

### NPY Format
```python
{
    'train': {'X': np.array(shape=(n_train, n_channels, seq_len)), 'y': np.array(shape=(n_train,))},
    'test': {'X': np.array(shape=(n_test, n_channels, seq_len)), 'y': np.array(shape=(n_test,))}
}
```

### TS Format
Standard `.ts` files as used in UCR/UEA archives.

## Results

Results are saved as CSV with columns:
- Dataset info (name, channels, length, classes)
- Model config (name, strategy, parameters)
- Performance (accuracy, F1-score, training time)
- Metadata (timestamp, hardware info)

### Custom Datasets
```bash
# Place your datasets in data/raw/Multivariate_ts/
# Then run benchmark
make benchmark
```
