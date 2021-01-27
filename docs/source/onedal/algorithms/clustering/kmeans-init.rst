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

The K-Means initialization algorithm receives :math:`n` feature vectors as input
and chooses :math:`k` initial centroids. After initialization, K-Means algorithm
uses the initialization result to partition input data into :math:`k` clusters.

.. |c_math| replace:: `Computing <kmeans_init_c_math_>`_
.. |c_dense| replace:: `Dense <kmeans_init_c_math_dense_>`_
.. |c_input| replace:: `compute_input <kmeans_init_c_api_input_>`_
.. |c_result| replace:: `compute_result <kmeans_init_c_api_result_>`_
.. |c_op| replace:: `compute(...) <kmeans_init_c_api_>`_

=============== =========================== ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |c_math|             |c_dense|            |c_op|   |c_input|   |c_result|
=============== =========================== ======== =========== ============

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

-------------
Usage example
-------------

Computing
---------

::

   table run_compute(const table& data) {
      const auto kmeans_desc = kmeans_init::descriptor<float,
                                                      kmeans_init::method::dense>{}
         .set_cluster_count(10)

      const auto result = compute(kmeans_desc, data);

      print_table("centroids", result.get_centroids());

      return result.get_centroids();
   }

--------
Examples
--------

.. include:: ./includes/kmeans-init-examples.rst

---------------------
Programming Interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::kmeans_init`` namespace and be available via inclusion of the
``oneapi/dal/algo/kmeans_init.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::kmeans_init::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans_init::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans_init::task::v1

Computing :cpp:expr:`compute(...)`
--------------------------------------

.. _kmeans_init_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::kmeans_init::v1::compute_input

.. _kmeans_init_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::kmeans_init::v1::compute_result

.. _kmeans_init_c_api:

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              kmeans_init::compute_result compute(const Descriptor& desc, \
                                         const kmeans_init::compute_input& input)

   :tparam desc: K-Means algorithm descriptor :expr:`kmeans_init::desc`
   :tparam input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.data.row_count == desc.cluster_count`
   Postconditions
      | :expr:`result.centroids.has_data == true`
      | :expr:`result.centroids.row_count == desc.cluster_count`
      | :expr:`result.centroids.column_count == input.data.column_count`
