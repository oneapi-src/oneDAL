.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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

.. _brute_force_method_begin:

The training operation produces the model that stores all the feature vectors
from the initial training set :math:`X`.

.. _brute_force_method_end:

.. _kdtree_method_begin:

The training operation builds a :math:`k`-:math:`d` tree that partitions the
training set :math:`X` (for more details, see :txtref:`k-d Tree <kd_tree>`).

.. _kdtree_method_end:

.. _inference_one_step_begin:

#. Identify the set :math:`N(x_j') \subseteq X` of the :math:`k` feature vectors
   in the training set that are nearest to :math:`x_j'` with respect to the
   Euclidean distance, which is chosen by default. The distance can be customized
   with the predefined set of pairwise distances: :ref:`Minkowski distances
   <alg_minkowski_distance>` with fractional degree (including Euclidean distance),
   :ref:`Chebyshev distance <alg_chebyshev_distance>` and
   :ref:`Cosine distance <alg_cosine_distance>`.

.. _inference_one_step_end:

.. _inference_brute_force_begin:

Brute-force inference method determines the set :math:`N(x_j')` of the
nearest feature vectors by iterating over all the pairs :math:`(x_j', x_i)` in
the implementation defined order, :math:`1 \leq i \leq n`, :math:`1 \leq j \leq
m`.
.. _inference_brute_force_end:

.. _inference_kdtree_begin:

K-d tree inference method traverses the :math:`k`-:math:`d` tree to find feature
vectors associated with a leaf node that are closest to :math:`x_j'`, :math:`1
\leq j \leq m`. The set :math:`\tilde{n}(x_j')` of the currently-known nearest
:math:`k`-th neighbors is progressively updated during tree traversal. The
search algorithm limits exploration of the nodes for which the distance between
the :math:`x_j'` and respective part of the feature space is not less than the
distance between :math:`x_j'` and the most distant feature vector from
:math:`\tilde{n}(x_j')`. Once tree traversal is finished, :math:`\tilde{n}(x_j')
\equiv N(x_j')`.

.. _inference_kdtree_end:
