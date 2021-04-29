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

.. _alg_pca:

===================================
Principal Components Analysis (PCA)
===================================

.. include:: ../../../includes/decomposition/pca-introduction.rst

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

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Principal Components Analysis <api_pca>`.

-------------
Usage example
-------------

.. include:: ../../../includes/decomposition/pca-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/decomposition/pca-examples.rst
