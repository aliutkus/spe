# Stochastic Positional Encoding (SPE)

This is the source code repository for the ICML 2021 paper [*Relative Positional Encoding for Transformers with Linear Complexity*](https://arxiv.org/abs/2105.08399) by Antoine Liutkus, Ondřej Cífka, Shih-Lun Wu, Umut Şimşekli, Yi-Hsuan Yang and Gaël Richard.

In this paper, we propose **Stochastic Positional Encoding** (SPE), which provably behaves like relative PE while being compatible with linear-complexity Transformers. We do this by drawing a connection between positional encoding and cross-covariance structures of correlated Gaussian processes.

![image](https://user-images.githubusercontent.com/8046580/119335679-fcf09280-bc8c-11eb-9525-bec9372bf6fb.png)

Check out also the [companion website](https://cifkao.github.io/spe/) with music examples.

## SPE implementation

We have implemented SPE in PyTorch and JAX/Flax. Each implementation is available as a separate Python package under [`src`](./src).

## Experiments

Each of the 3 experiments (LRA, pop piano generation, groove continuation) has a dedicated directory under [`experiments`](./experiments). See the README files there for how to set up the environment and prepare the datasets. To make sure you have the custom dependencies for each experiment, clone this repository with `--recurse-submodules` or run `git submodule init && git submodule update` after cloning.
