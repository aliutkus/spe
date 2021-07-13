# Stochastic Positional Encoding (SPE)

This is the source code repository for the ICML 2021 paper [*Relative Positional Encoding for Transformers with Linear Complexity*](http://proceedings.mlr.press/v139/liutkus21a.html) by Antoine Liutkus, Ondřej Cífka, Shih-Lun Wu, Umut Şimşekli, Yi-Hsuan Yang and Gaël Richard.

In this paper, we propose **Stochastic Positional Encoding** (SPE), which provably behaves like relative PE while being compatible with linear-complexity Transformers. We do this by drawing a connection between positional encoding and cross-covariance structures of correlated Gaussian processes.

![image](https://user-images.githubusercontent.com/8046580/119335679-fcf09280-bc8c-11eb-9525-bec9372bf6fb.png)

Check out also the [companion website](https://cifkao.github.io/spe/) with music examples.

Citation:
```bibtex
@inproceedings{pmlr-v139-liutkus21a,
  title = 	 {Relative Positional Encoding for {Transformers} with Linear Complexity},
  author =       {Liutkus, Antoine and C{\'i}fka, Ond{\v r}ej and Wu, Shih-Lun and {\c S}im{\c s}ekli, Umut and Yang, Yi-Hsuan and Richard, Ga{\"e}l},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {7067--7079},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/liutkus21a/liutkus21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/liutkus21a.html}
}
```

## SPE implementation

We have implemented SPE in PyTorch and JAX/Flax. Each implementation is available as a separate Python package under [`src`](./src).

## Experiments

Each of the 3 experiments (LRA, pop piano generation, groove continuation) has a dedicated directory under [`experiments`](./experiments). See the README files there for how to set up the environment and prepare the datasets. To make sure you have the custom dependencies for each experiment, clone this repository with `--recurse-submodules` or run `git submodule init && git submodule update` after cloning.
