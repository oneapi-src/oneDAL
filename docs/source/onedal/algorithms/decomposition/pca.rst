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

.. _alg_pca:

===================================
Principal Components Analysis (PCA)
===================================
Principal Component Analysis (PCA) is an algorithm for exploratory data analysis
and :capterm:`dimensionality reduction`. PCA transforms a set of feature vectors
of possibly correlated features to a new set of uncorrelated features, called
principal components. Principal components are the directions of the largest
variance, that is, the directions where the data is mostly spread out.

.. |t_math| replace:: `Training <pca_t_math_>`_
.. |t_cov| replace:: `Covariance <pca_t_math_cov_>`_
.. |t_svd| replace:: `SVD <pca_t_math_svd_>`_
.. |t_input| replace:: `train_input <pca_t_api_input_>`_
.. |t_result| replace:: `train_result <pca_t_api_result_>`_
.. |t_op| replace:: `train(...) <pca_t_api_>`_

.. |i_math| replace:: `Inference <pca_i_math_>`_
.. |i_cov| replace:: `Covariance <pca_i_math_cov_>`_
.. |i_svd| replace:: `SVD <pca_i_math_svd_>`_
.. |i_input| replace:: `infer_input <pca_i_api_input_>`_
.. |i_result| replace:: `infer_result <pca_i_api_result_>`_
.. |i_op| replace:: `infer(...) <pca_i_api_>`_

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|        |t_cov|       |t_svd|     |t_op|   |t_input|   |t_result|
   |i_math|        |i_cov|       |i_svd|     |i_op|   |i_input|   |i_result|
=============== ============= ============= ======== =========== ============

------------------------
Mathematical formulation
------------------------

.. _pca_t_math:

Training
--------
Given the training set :math:`X = \{ x_1, \ldots, x_n \}` of
:math:`p`-dimensional feature vectors and the number of principal components
:math:`r`, the problem is to compute :math:`r` principal directions
(:math:`p`-dimensional eigenvectors [Lang87]_) for the training set. The
eigenvectors can be grouped into the :math:`r \times p` matrix :math:`T` that
contains one eigenvector in each row.

.. _pca_t_math_cov:

Training method: *Covariance*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method uses eigenvalue decomposition of the covariance matrix to compute
the principal components of the datasets. The method relies on the following
steps:

#. Computation of the covariance matrix
#. Computation of the eigenvectors and eigenvalues
#. Formation of the matrices storing the results

Covariance matrix computation is performed in the following way:

#. Compute the vector-column of sums :math:`s_i = \sum_{j=1}^n x_{i,j}, \quad 1 \leq i \leq p`.

#. Compute the cross-product :math:`P = X^TX - s^Ts`.

#. Compute the covariance matrix :math:`\Sigma = \frac{1}{n - 1} P`.

To compute eigenvalues :math:`\lambda_i` and eigenvectors :math:`\upsilon_i`, the implementer can choose an arbitrary
method such as [Ping14]_.

The final step is to sort the set of pairs :math:`(\lambda_i, \upsilon_i)` in
the descending order by :math:`\lambda_i` and form the resulting matrix :math:`T
= (\upsilon_{i,1}, \cdots, \upsilon_{i,r}), \quad 1 \leq i \leq p`.
Additionally, the means and variances of the initial dataset are returned.

.. _pca_t_math_svd:

Training method: *SVD*
~~~~~~~~~~~~~~~~~~~~~~

This method uses singular value decomposition of the dataset to compute its
principal components. The method relies on the following steps:

#. Computation of the singular values and singular vectors
#. Formation of the matrices storing the results

To compute singular values :math:`\lambda_i` and singular vectors :math:`u_i`
and :math:`v_i`, the implementer can choose an arbitrary method such as
[Demmel90]_.

The final step is to sort the set of pairs :math:`(\lambda_i, v_i)` in the
descending order by :math:`\lambda_i` and form the resulting matrix :math:`T =
(v_{i,1}, \cdots, v_{i,r}), \quad 1 \leq i \leq p`. Additionally, the means and
variances of the initial dataset are returned.

Sign-flip technique
~~~~~~~~~~~~~~~~~~~
Eigenvectors computed by some eigenvalue solvers are not uniquely defined due to
sign ambiguity. To get the deterministic result, a sign-flip technique should be
applied. One of the sign-flip techniques proposed in [Bro07]_ requires the
following modification of matrix :math:`T`:

.. math::
   \hat{T}_i = T_i \cdot \mathrm{sgn}(\max_{1 \leq j \leq p } |{T}_{ij}|), \quad 1 \leq i \leq r,

where :math:`T_i` is :math:`i`-th row, :math:`T_{ij}` is the element in the
:math:`i`-th row and :math:`j`-th column, :math:`\mathrm{sgn}(\cdot)` is the
signum function,

.. math::
   \mathrm{sgn}(x) =
      \begin{cases}
         -1, & x < 0, \\
          0, & x = 0, \\
          1, & x > 0.
      \end{cases}

.. _pca_i_math:

Inference
---------
Given the inference set :math:`X' = \{ x_1', \ldots, x_m' \}` of
:math:`p`-dimensional feature vectors and the :math:`r \times p` matrix
:math:`T` produced at the training stage, the problem is to transform :math:`X'`
to the set :math:`X'' = \{ x_1'', \ldots, x_m'' \}`, where :math:`x_{j}''` is an
:math:`r`-dimensional feature vector, :math:`1 \leq j \leq m`.

The feature vector :math:`x_{j}''` is computed through applying linear
transformation [Lang87]_ defined by the matrix :math:`T` to the feature vector
:math:`x_{j}'`,

.. math::
   :label: x_transform

   x_{j}'' = T x_{j}', \quad 1 \leq j \leq m.


.. _pca_i_math_cov:
.. _pca_i_math_svd:

Inference methods: *Covariance* and *SVD*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Covariance and SVD inference methods compute :math:`x_{j}''` according to
:eq:`x_transform`.


-------------
Usage example
-------------

Training
--------

::

   pca::model<> run_training(const table& data) {
      const auto pca_desc = pca::descriptor<float>{}
         .set_component_count(5)
         .set_deterministic(true);

      const auto result = train(pca_desc, data);

      print_table("means", result.get_means());
      print_table("variances", result.get_variances());
      print_table("eigenvalues", result.get_eigenvalues());
      print_table("eigenvectors", result.get_eigenvectors());

      return result.get_model();
   }

Inference
---------

::

   table run_inference(const pca::model<>& model,
                     const table& new_data) {
      const auto pca_desc = pca::descriptor<float>{}
         .set_component_count(model.get_component_count());

      const auto result = infer(pca_desc, model, new_data);

      print_table("labels", result.get_transformed_data());
   }

--------
Examples
--------

.. include:: ./includes/pca-examples.rst

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::pca`` namespace and be available via inclusion of the
``oneapi/dal/algo/pca.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::pca::v1::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::pca::method::v1

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::pca::task::v1

Model
-----
.. onedal_class:: oneapi::dal::pca::v1::model


.. _pca_t_api:

Training :cpp:expr:`train(...)`
--------------------------------
.. _pca_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::pca::v1::train_input


.. _pca_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::pca::v1::train_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              pca::train_result train(const Descriptor& desc, \
                                         const pca::train_input& input)

   :tparam desc: PCA algorithm descriptor :expr:`pca::desc`
   :tparam input: Input data for the training operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.data.column_count >= desc.component_count`
   Postconditions
      | :expr:`result.means.row_count == 1`
      | :expr:`result.means.column_count == desc.component_count`
      | :expr:`result.variances.row_count == 1`
      | :expr:`result.variances.column_count == desc.component_count`
      | :expr:`result.variances[i] >= 0.0`
      | :expr:`result.eigenvalues.row_count == 1`
      | :expr:`result.eigenvalues.column_count == desc.component_count`
      | :expr:`result.model.eigenvectors.row_count == 1`
      | :expr:`result.model.eigenvectors.column_count == desc.component_count`

.. _pca_i_api:

Inference :cpp:expr:`infer(...)`
---------------------------------
.. _pca_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::pca::v1::infer_input


.. _pca_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::pca::v1::infer_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              pca::infer_result infer(const Descriptor& desc, \
                                         const pca::infer_input& input)

   :tparam desc: PCA algorithm descriptor :expr:`pca::desc`
   :tparam input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.model.eigenvectors.row_count == desc.component_count`
      | :expr:`input.model.eigenvectors.column_count == input.data.column_count`
   Postconditions
      | :expr:`result.transformed_data.row_count == input.data.row_count`
      | :expr:`result.transformed_data.column_count == desc.component_count`
