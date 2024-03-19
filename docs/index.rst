.. include:: ../README.rst

Welcome to pyhms's documentation!
===================================

**pyhms** is a Python implementation of Hierarchic Memetic Strategy (HMS).

The Hierarchic Memetic Strategy is a stochastic global optimizer designed to tackle highly multimodal problems. It is a composite global optimization strategy consisting of a multi-population evolutionary strategy and some auxiliary methods. The HMS makes use of a dynamically-evolving data structure that provides an organization among the component populations. It is a tree with a fixed maximal height and variable internal node degree. Each component population is governed by a particular optimization engine. This package provides a simple python implementation.

Check out the :doc:`usage` section for further information, including
how to install the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   Home <self>
   algorithm
   usage
   inspecting
   problem
