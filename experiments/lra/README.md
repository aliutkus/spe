# Long-Range Arena experiments

Original LRA repository: [google-research/long-range-arena](https://github.com/google-research/long-range-arena)

## Setup

Install JAX (adjust the `jaxlib` version according to your CUDA version):
```bash
pip install jax==0.2.6 jaxlib==0.1.57+cuda102 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Install the rest of the requirements:
```bash
pip install -r requirements.txt
pip install -e ./fast_attention ./long-range-arena ../../src/flax
```
