.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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

.. _alg_kmeans:

=======
K-Means
=======
The K-Means algorithm solves :capterm:`clustering` problem by partitioning
:math:`n` feature vectors into :math:`k` clusters minimizing some criterion.
Each cluster is characterized by a representative point, called *a centroid*.

.. |t_math| replace:: `Training <kmeans_t_math_>`_
.. |t_lloyd| replace:: `Lloyd's <kmeans_t_math_lloyd_>`_
.. |t_input| replace:: `train_input <kmeans_t_api_input_>`_
.. |t_result| replace:: `train_result <kmeans_t_api_result_>`_
.. |t_op| replace:: `train(...) <kmeans_t_api_>`_

.. |i_math| replace:: `Inference <kmeans_i_math_>`_
.. |i_lloyd| replace:: `Lloyd's <kmeans_i_math_lloyd_>`_
.. |i_input| replace:: `infer_input <kmeans_i_api_input_>`_
.. |i_result| replace:: `infer_result <kmeans_i_api_result_>`_
.. |i_op| replace:: `infer(...) <kmeans_i_api_>`_

=============== =========================== ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|             |t_lloyd|            |t_op|   |t_input|   |t_result|
   |i_math|             |i_lloyd|            |i_op|   |i_input|   |i_result|
=============== =========================== ======== =========== ============

------------------------
Mathematical formulation
------------------------

.. _kmeans_t_math:

Training
--------
Given the training set :math:`X = \{ x_1, \ldots, x_n \}` of
:math:`p`-dimensional feature vectors and a positive integer :math:`k`, the
problem is to find a set :math:`C = \{ c_1, \ldots, c_k \}` of
:math:`p`-dimensional centroids that minimize the objective function

.. math::
   \Phi_{X}(C) = \sum_{i = 1}^n d^2(x_i, C),

where :math:`d^2(x_i, C)` is the squared Euclidean distance from :math:`x_i` to
the closest centroid in :math:`C`,

.. math::
   d^2(x_i, C) = \min_{1 \leq j \leq k} \| x_i - c_j \|^2, \quad 1 \leq i \leq n.

Expression :math:`\|\cdot\|` denotes :math:`L_2` `norm
<https://mathworld.wolfram.com/L2-Norm.html>`_.

.. note::
   In the general case, :math:`d` may be an arbitrary distance function. Current
   version of the oneDAL spec defines only Euclidean distance case.


.. _kmeans_t_math_lloyd:

Training method: *Lloyd's*
~~~~~~~~~~~~~~~~~~~~~~~~~~
The Lloyd's method [Lloyd82]_ consists in iterative updates of centroids by
applying the alternating *Assignment* and *Update* steps, where :math:`t`
denotes a index of the current iteration, e.g., :math:`C^{(t)} = \{ c_1^{(t)},
\ldots, c_k^{(t)} \}` is the set of centroids at the :math:`t`-th iteration. The
method requires the initial centroids :math:`C^{(1)}` to be specified at the
beginning of the algorithm (:math:`t = 1`).

**(1) Assignment step:** Assign each feature vector :math:`x_i` to the nearest
centroid. :math:`y_i^{(t)}` denotes the assigned label (cluster index) to the
feature vector :math:`x_i`.

.. math::
   y_i^{(t)} = \mathrm{arg}\min_{1 \leq j \leq k} \| x_i - c_j^{(t)} \|^2, \quad 1 \leq i \leq n.

Each feature vector from the training set :math:`X` is assigned to exactly one
centroid so that :math:`X` is partitioned to :math:`k` disjoint sets (clusters)

.. math::
   S_j^{(t)} = \big\{ \; x_i \in X : \; y_i^{(t)} = j \; \big\}, \quad 1 \leq j \leq k.

**(2) Update step:** Recalculate centroids by averaging feature vectors assigned
to each cluster.

.. math::
   c_j^{(t + 1)} = \frac{1}{|S_j^{(t)}|} \sum_{x \in S_j^{(t)}} x, \quad 1 \leq j \leq k.

The steps (1) and (2) are performed until the following **stop condition**,

.. math::
   \sum_{j=1}^k \big\| c_j^{(t)} - c_j^{(t+1)} \big\|^2 < \varepsilon,

is satisfied or number of iterations exceeds the maximal value :math:`T` defined
by the user.


.. _kmeans_i_math:

Inference
---------
Given the inference set :math:`X' = \{ x_1', \ldots, x_m' \}` of
:math:`p`-dimensional feature vectors and the set :math:`C = \{ c_1, \ldots, c_k
\}` of centroids produced at the training stage, the problem is to predict the
index :math:`y_j' \in \{ 0, \ldots, k-1 \}`, :math:`1 \leq j \leq m`, of the
centroid in accordance with a method-defined rule.

.. _kmeans_i_math_lloyd:

Inference method: *Lloyd's*
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lloyd's inference method computes the :math:`y_j'` as an index of the centroid
closest to the feature vector :math:`x_j'`,

.. math::
   y_j' = \mathrm{arg}\min_{1 \leq l \leq k} \| x_j' - c_l \|^2, \quad 1 \leq j \leq m.


-------------
Usage example
-------------
Training
--------
.. .. onedal_code:: oneapi::dal::kmeans::example::run_training

Inference
---------
.. .. onedal_code:: oneapi::dal::kmeans::example::run_inference

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::kmeans`` namespace and be available via inclusion of the
``oneapi/dal/algo/kmeans.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::kmeans::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans::task::v1

Model
-----
.. onedal_class:: oneapi::dal::kmeans::v1::model


.. _kmeans_t_api:

Training :expr:`train(...)`
--------------------------------
.. _kmeans_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::kmeans::v1::train_input


.. _kmeans_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::kmeans::v1::train_result

Operation
~~~~~~~~~
.. .. onedal_func:: oneapi::dal::kmeans::v1::train


.. _kmeans_i_api:

Inference :expr:`infer(...)`
----------------------------
.. _kmeans_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::kmeans::v1::infer_input


.. _kmeans_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::kmeans::v1::infer_result

Operation
~~~~~~~~~
.. .. onedal_func:: oneapi::dal::kmeans::v1::infer
