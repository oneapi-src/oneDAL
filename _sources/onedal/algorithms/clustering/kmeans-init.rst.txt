.. ******************************************************************************
.. * Copyright 2020-2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. highlight:: cpp
.. default-domain:: cpp

.. _alg_kmeans_init:

======================
K-Means initialization
======================

.. include:: ../../../includes/clustering/kmeans-init-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _kmeans_init_c_math:

Computing
---------

Given the training set :math:`X = \{ x_1, \ldots, x_n \}` of
:math:`p`-dimensional feature vectors and a positive integer :math:`k`, the
problem is to find a set :math:`C = \{ c_1, \ldots, c_k \}` of
:math:`p`-dimensional initial centroids.

.. _kmeans_init_c_math_dense:

Computing method: *dense*
-------------------------

The method chooses first :math:`k` feature vectors from the training set
:math:`X`.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: K-Means initialization <api_kmeans_init>`.

-------------
Usage example
-------------

.. include:: ../../../includes/clustering/kmeans-init-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/clustering/kmeans-init-examples.rst
