# spe
Stochastic Positional Encoding for PyTorch.

## Installation

```bash
pip install -e .
```

## Usage

Create an instance of either `SineSPE` or `ConvSPE`, and an instance of `SPEFilter`:
```python
spe_encoder = spe.SineSPE(num_heads=8,          # Number of attention heads
                          in_features=64,       # Dimension of keys and queries
                          num_realizations=64,  # New dimension of keys and queries
                          num_sines=5)          # Number of sinusoidal components
spe_filter = spe.SPEFilter(gated=True, code_shape=spe_encoder.code_shape)
```
`SineSPE` and `ConvSPE` take care of generating the positional codes  Q̅ and K̅, and `SPEFilter` combines these with queries Q and keys K to form new queries Q̂ and keys K̂:
```python
pos_codes = spe_encoder(queries.shape[:2])  # pos_codes is a tuple (qbar, kbar)
queries, keys = spe_filter(queries, keys, pos_codes)
```
