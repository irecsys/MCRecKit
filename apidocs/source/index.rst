.. MCRecKit documentation master file, created by
   sphinx-quickstart on Sat Nov 16 20:42:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MCRecKit's documentation!
====================================

MCRecKit: Introduction
=======================

.. image:: https://img.shields.io/badge/License-MIT-orange.svg
   :target: ./LICENSE
   :alt: License

.. image:: https://badges.aleen42.com/src/python.svg
   :target: https://badges.aleen42.com/src/python.svg
   :alt: Python

MCRecKit is a Python-based open-source library for multi-criteria
recommender systems (MCRSs). It was built upon RecBole v1.0.1. 

MCRecKit implements multi-stage MCRS algorithms, where usually two stages are involved:
1. Predicting multi-criteria ratings by given a user and an item.
2. Aggregating the predicted multi-criteria ratings for either rating prediction or top-N recommendation tasks.

In terms of the second step above, two strategies (i.e., MLP and MC-ranking) 
were provided. The multi-layer perceptron (MLP) can be used to estimate the overall rating
from the predicted multi-criteria ratings. The multi-criteria ranking (MC-ranking) can provide
top-N item recommendations directly by utilizing ranking methods (such as Pareto ranking) based on predicted multi-criteria
ratings. CUDA programming was used to accelerate the process of Pareto rankings.

Running Environments
--------------------

The library was tested and run on Python 3.9.20. You can create a conda environment by using:

.. code-block:: bash

    conda create -n mcreckit python=3.9.20
    conda activate mcreckit

Required libraries can be installed using:

.. code-block:: bash

    pip install -r requirements.txt

The command above will install CPU-only PyTorch. If you do need GPU-supported
Pytorch, refer to https://pytorch.org/get-started/locally/ for installation.

In addition, for algorithms using multi-criteria rankings (e.g., CriteriaSort), we utilized
CUDA programming for accelerations. Users need to further install cudatoolkit. An example in
conda environment can be shown below:

.. code-block:: bash

    conda install cudatoolkit

The library was tested by using the following environments:

- ``python==3.9.20``
- ``recbole==1.0.1``
- ``numpy==1.20.0``
- ``lightgbm==4.5.0``
- ``xgboost==2.1.1``
- ``numba==0.58.1``
- ``pymoo==0.6.1.2``

Note that you may need to downgrade numpy to 1.20.0 for compatibility concerns.

Quick-Start
-----------

With the source code, you can use the provided script for initial usage of our library:

.. code-block:: bash

    python -m run_config run/CombRP_Independent.yaml

Data Sets
---------

Three data sets, including ITM-Rec, OpenTable, and Yahoo!Movies, are provided in the ``dataset`` folder.

Major Releases
--------------

.. list-table:: 
   :header-rows: 1

   * - Releases
     - Date
   * - v1.0.0
     - 11/24/2024

Contributors
------------

- Dr. Yong Zheng, Illinois Institute of Technology, USA
- Dr. David Xuejun Wang, Morningstar, Inc., USA
- Dr. Qin Ruan, University College Dublin, Ireland

References
----------

- Yong Zheng, David Xuejun Wang, Qin Ruan. 
  "MCRecKit: An Open-Source Library for Multi-Criteria Recommendations"，2024 IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT). IEEE, 2024.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   
   mcreckit/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
