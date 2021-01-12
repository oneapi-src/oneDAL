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

.. _alg_knn:

=========================================
k-Nearest Neighbors Classification (k-NN)
=========================================
:math:`k`-NN :capterm:`classification` algorithm infers the class for the new
feature vector by computing majority vote of the :math:`k` nearest observations
from the training set.


.. |t_math| replace:: `Training <knn_t_math_>`_
.. |t_brute_f| replace:: `Brute-force <knn_t_math_brute_force_>`_
.. |t_kd_tree| replace:: `k-d tree <knn_t_math_kd_tree_>`_
.. |t_input| replace:: `train_input <knn_t_api_input_>`_
.. |t_result| replace:: `train_result <knn_t_api_result_>`_
.. |t_op| replace:: `train(...) <knn_t_api_>`_

.. |i_math| replace:: `Inference <knn_i_math_>`_
.. |i_brute_f| replace:: `Brute-force <knn_i_math_brute_force_>`_
.. |i_kd_tree| replace:: `k-d tree <knn_i_math_kd_tree_>`_
.. |i_input| replace:: `infer_input <knn_i_api_input_>`_
.. |i_result| replace:: `infer_result <knn_i_api_result_>`_
.. |i_op| replace:: `infer(...) <knn_i_api_>`_

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|      |t_brute_f|   |t_kd_tree|   |t_op|   |t_input|   |t_result|
   |i_math|      |i_brute_f|   |i_kd_tree|   |i_op|   |i_input|   |i_result|
=============== ============= ============= ======== =========== ============

------------------------
Mathematical formulation
------------------------

.. _knn_t_math:

Training
--------
Let :math:`X = \{ x_1, \ldots, x_n \}` be the training set of
:math:`p`-dimensional feature vectors, let :math:`Y = \{ y_1, \ldots, y_n \}` be
the set of class labels, where :math:`y_i \in \{ 0, \ldots, c-1 \}`, :math:`1
\leq i \leq n`. Given :math:`X`, :math:`Y` and the number of nearest neighbors
:math:`k`, the problem is to build a model that allows distance computation
between the feature vectors in training and inference sets at the inference
stage.

.. _knn_t_math_brute_force:

Training method: *brute-force*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The training operation produces the model that stores all the feature vectors
from the initial training set :math:`X`.


.. _knn_t_math_kd_tree:

Training method: *k-d tree*
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The training operation builds a :math:`k`-:math:`d` tree that partitions the
training set :math:`X` (for more details, see :txtref:`k-d Tree <kd_tree>`).


.. _knn_i_math:

Inference
---------
Let :math:`X' = \{ x_1', \ldots, x_m' \}` be the inference set of
:math:`p`-dimensional feature vectors. Given :math:`X'`, the model produced at
the training stage and the number of nearest neighbors :math:`k`, the problem is
to predict the label :math:`y_j'` for each :math:`x_j'`, :math:`1 \leq j \leq
m`, by performing the following steps:

#. Identify the set :math:`N(x_j') \subseteq X` of the :math:`k` feature vectors
   in the training set that are nearest to :math:`x_j'` with respect to the
   Euclidean distance.

#. Estimate the conditional probability for the :math:`l`-th class as the
   fraction of vectors in :math:`N(x_j')` whose labels :math:`y_j` are equal to
   :math:`l`:

   .. math::
      :label: p_predict

      P_{jl} = \frac{1}{| N(x_j') |} \Big| \big\{ x_r \in N(x_j') : y_r = l
      \big\} \Big|, \quad 1 \leq j \leq m, \; 0 \leq l < c.


#. Predict the class that has the highest probability for the feature vector
   :math:`x_j'`:

   .. math::
      :label: y_predict

      y_j' = \mathrm{arg}\max_{0 \leq l < c} P_{jl},
      \quad 1 \leq j \leq m.


.. _knn_i_math_brute_force:

Inference method: *brute-force*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Brute-force inference method determines the set :math:`N(x_j')` of the
nearest feature vectors by iterating over all the pairs :math:`(x_j', x_i)` in
the implementation defined order, :math:`1 \leq i \leq n`, :math:`1 \leq j \leq
m`. The final prediction is computed according to the equations :eq:`p_predict`
and :eq:`y_predict`.


.. _knn_i_math_kd_tree:

Inference method: *k-d tree*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
K-d tree inference method traverses the :math:`k`-:math:`d` tree to find feature
vectors associated with a leaf node that are closest to :math:`x_j'`, :math:`1
\leq j \leq m`. The set :math:`\tilde{n}(x_j')` of the currently-known nearest
:math:`k`-th neighbors is progressively updated during tree traversal. The
search algorithm limits exploration of the nodes for which the distance between
the :math:`x_j'` and respective part of the feature space is not less than the
distance between :math:`x_j'` and the most distant feature vector from
:math:`\tilde{n}(x_j')`. Once tree traversal is finished, :math:`\tilde{n}(x_j')
\equiv N(x_j')`. The final prediction is computed according to the equations
:eq:`p_predict` and :eq:`y_predict`.

-------------
Usage example
-------------

Training
--------

::

   knn::model<> run_training(const table& data,
                           const table& labels) {
      const std::int64_t class_count = 10;
      const std::int64_t neighbor_count = 5;
      const auto knn_desc = knn::descriptor<float>{class_count, neighbor_count};

      const auto result = train(knn_desc, data, labels);

      return result.get_model();
   }

Inference
---------

::

   table run_inference(const knn::model<>& model,
                     const table& new_data) {
      const std::int64_t class_count = 10;
      const std::int64_t neighbor_count = 5;
      const auto knn_desc = knn::descriptor<float>{class_count, neighbor_count};

      const auto result = infer(knn_desc, model, new_data);

      print_table("labels", result.get_labels());
   }

--------
Examples
--------

.. include:: ./includes/knn-examples.rst

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::knn`` namespace and be available via inclusion of the
``oneapi/dal/algo/knn.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::knn::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::knn::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::knn::task::v1

Model
-----
.. onedal_class:: oneapi::dal::knn::v1::model


.. _knn_t_api:

Training :cpp:expr:`train(...)`
--------------------------------
.. _knn_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::knn::v1::train_input


.. _knn_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::knn::v1::train_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              knn::train_result train(const Descriptor& desc, \
                                         const knn::train_input& input)

   :tparam desc: k-NN algorithm descriptor :expr:`knn::desc`
   :tparam input: Input data for the training operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.labels.has_data == true`
      | :expr:`input.data.row_count == input.labels.row_count`
      | :expr:`input.labels.column_count == 1`
      | :expr:`input.labels[i] >= 0`
      | :expr:`input.labels[i] < desc.class_count`

.. _knn_i_api:

Inference :cpp:expr:`infer(...)`
---------------------------------
.. _knn_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::knn::v1::infer_input


.. _knn_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::knn::v1::infer_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              knn::infer_result infer(const Descriptor& desc, \
                                         const knn::infer_input& input)

   :tparam desc: k-NN algorithm descriptor :expr:`knn::desc`
   :tparam input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.has_data == true`
   Postconditions
     | :expr:`result.labels.row_count == input.data.row_count`
     | :expr:`result.labels.column_count == 1`
     | :expr:`result.labels[i] >= 0`
     | :expr:`result.labels[i] < desc.class_count`
