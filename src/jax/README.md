# jax-spe
Stochastic Positional Encoding for JAX/Flax.

## Installation

```bash
pip install -e .
```

# Usage
The `SineSPE` and `ConvSPE` modules generate positional codes Q̅ and K̅, the `SPEGate` applies the optional gating, and the `apply_spe` functions combines Q̅ and K̅ with queries Q and keys K to form new queries Q̂ and keys K̂.

See the [example notebook](./examples/test_spe.ipynb).
