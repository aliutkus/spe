# Stochastic Positional Encoding (SPE)

This is the source code repository for the ICML 2021 paper [*Relative Positional Encoding for Transformers with Linear Complexity*](https://arxiv.org/abs/2105.08399) by Antoine Liutkus, Ondřej Cífka, Shih-Lun Wu, Umut Şimşekli, Yi-Hsuan Yang and Gaël Richard.

In this paper, we propose **Stochastic Positional Encoding** (SPE), which provably behaves like relative PE while being compatible with linear-complexity Transformers. We do this by drawing a connection between positional encoding and cross-covariance structures of correlated Gaussian processes.

![image](https://user-images.githubusercontent.com/8046580/119335679-fcf09280-bc8c-11eb-9525-bec9372bf6fb.png)
