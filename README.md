# pyhms

`pyhms` is a Python implementation of Hierarchic Memetic Strategy (HMS).

The Hierarchic Memetic Strategy is a stochastic global optimizer designed to tackle highly multimodal problems. It is a composite global optimization strategy consisting of a multi-population evolutionary strategy and some auxiliary methods. The HMS makes use of a dynamically-evolving data structure that provides an organization among the component populations. It is a tree with a fixed maximal height and variable internal node degree. Each component population is governed by a particular optimization engine. This package provides a simple python implementation.

### Installation
Installation can be done using `pypi`:
```
pip install pyhms
```
It's also possible to install the current master branch:
```
pip install git+https://github.com/maciejsmolka/pyhms.git@master
```

### Usage

TODO

### Relevant literature

- J. Sawicki, M. Łoś, M. Smołka, R. Schaefer. Understanding measure-driven algorithms solving irreversibly ill-conditioned problems. Natural Computing 21:289-315, 2022. doi: [10.1007/s11047-020-09836-w](https://doi.org/10.1007/s11047-020-09836-w)
- J. Sawicki, M. Łoś, M. Smołka, J. Alvarez-Aramberri. Using Covariance Matrix Adaptation Evolutionary Strategy to boost the search accuracy in hierarchic memetic computations. Journal of computational science, 34, 48-54, 2019. doi: [https://doi.org/10.1016/j.jocs.2019.04.005](https://doi.org/10.1016/j.jocs.2019.04.005)
