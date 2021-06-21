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

.. _alg_knn:

=================================
k-Nearest Neighbors Search (k-NN)
=================================

.. include:: ../../../includes/nearest-neighbors/knn-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _knn_t_math:

Training
--------

Let :math:`X = \{ x_1, \ldots, x_n \}` be the training set of
:math:`p`-dimensional feature vectors. Given :math:`X` and the number of nearest neighbors
:math:`k`, the problem is to build a model that allows distance computation
between the feature vectors in training and inference sets at the inference
stage.

.. _knn_t_math_brute_force:

Training method: *brute-force*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: knn.rst
   :start-after: brute_force_method_begin:
   :end-before: brute_force_method_end:

.. _knn_t_math_kd_tree:

Training method: *k-d tree*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: knn.rst
   :start-after: kdtree_method_begin:
   :end-before: kdtree_method_end:

.. _knn_i_math:

Inference
---------
Let :math:`X' = \{ x_1', \ldots, x_m' \}` be the inference set of
:math:`p`-dimensional feature vectors. Given :math:`X'`, the model produced at
the training stage and the number of nearest neighbors :math:`k`

.. include:: knn.rst
   :start-after: inference_one_step_begin:
   :end-before: inference_one_step_end:

.. _knn_i_math_brute_force:

Inference method: *brute-force*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. include:: knn.rst
   :start-after: inference_brute_force_begin:
   :end-before: inference_brute_force_end:

.. _knn_i_math_kd_tree:

Inference method: *k-d tree*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_inference_kdtree_begin

.. include:: knn.rst
   :start-after: inference_kdtree_begin:
   :end-before: inference_kdtree_end:

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: k-Nearest Neighbors Search <api_knn>`.

--------
Examples
--------

.. include:: ../../../includes/nearest-neighbors/knn-examples.rst
