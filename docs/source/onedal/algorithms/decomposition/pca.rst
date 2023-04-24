.. ******************************************************************************
.. * Copyright 2023 Intel Corporation
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
Given a training data set :math:`X_{n \times p}` with :math:`n`observations and
:math:`p` features, the problem is to compute :math:`r, 1 \leq r \leq p` 
principal directions (:math:`p`-dimensional eigenvectors [Lang87]_) of the 
training date set. The eigenvectors can be grouped into an :math:`r \times p` 
matrix :math:`T` that contains one eigenvector in each row.

The principal components can be computed with any of the following two methods:

#. Covariance (or Correlation)
#. Singular Value Decomposition(SVD)

.. _pca_t_math_cov:

Training method: *Covariance*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Principal components can be computed by both covariance matrix and correlation matrix.
The choice of covariance matrix or correlation matrix is application dependent. 
More specifically, if scaling of the features is important for a problem, 
which is often the case, using the correlation matrix  to compute principal components is more appropriate.
By default, oneDAL uses correlation matrix to compute the principal components. But it is possible 
to use the covariance matrix by calling the ``precomputed`` method and feeding a covariance matrix as input
to the PCA algorithm. To compute the covariance matrix the :ref:`Covariance <alg_covariance>` algorithm can be used.

The eigenvector associated with the :math:`k`-th largest eigenvalue of the covariance (or correlation) matrix 
is also the :math:`k`-th principal component of the training data set.  Based on this principle,
the principal components can be computed in three steps:

#. Computation of the covariance (or correlation) matrix
#. Computation of the eigenvectors and eigenvalues of the covariance (or correlation) matrix
#. Processing (sorting and storing) the results


Covariance matrix can be computed in the following way:

#. Compute the column means :math:`M = (M_j)`,:math:`j = \overline{1,p}`, :math:`M_j = \frac{1}{n}\sum _{i} X_{ij}`.
#. Compute the sample covariance matrix :math:`S = (S_{ij})`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`, :math:`S_{ij} = \frac{1}{n-1}\sum_{k=1}^{n}(X_{ki} - M_i)(X_{kj}-M_j)`.

Corelation matrix can be computed from covariance matrix in the following way:

#. Compute the correlation matrix :math:`C = C_{ij}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`,:math:`C_{ij} = \frac{S_{ij}}{\sqrt{S_{ii}\cdot S_{jj}}}`.


The eigenvalues :math:`\lambda_k` and eigenvectors :math:`\upsilon_k` can be computed by an arbitrary
method such as [Ping14]_.

In the final step, the eigenvalues (:math:`\lambda_k`) are sorted in descending order 
to determine the order of the principal components. Each principal component is 
stored as a row of the final resulting matrix, :math:`T = (\upsilon_{1j}, \cdots, \upsilon_{rj}), \quad 1 \leq j \leq p`.
Additionally, the means and variances of the input dataset are returned.

.. _pca_t_math_svd:

Training method: *SVD*
~~~~~~~~~~~~~~~~~~~~~~

The singular value decomposition (SVD) is a matrix factorization technique that
decomposes an observation matrix :math:`X_{n \times p}` into three matrices as :math:`X = U\SigmaV^*`.
Here, the columns of :math:`U` are the left-singular vectors, the
columns of  :math:`V` are the right-singular vectors, :math:`V^*` is the 
conjugate transpose of the matrix :math:`V` and the diagonal entries of :math:`\Sigma` 
are the singular values (:math:`\sigma`) of :math:`X`. The right-singular vectors are 
the principal components of :math:`X`. The steps of computing principal components using SVD technique
are as follows:

#. Mean centering the input data
#. Decomposing the mean-centered input data to compute the singular values and the singular vectors
#. Processing(sorting and storing) the results

First step is to mean center the input data :math:`M = M_{ij}`, where :math:`M_{ij} = X_{ij} - \frac{\sum_{i=1}^n X_{ij}}{n}`.

Singular values :math:`\sigma_k`, left-singular vectors :math:`U_k` and right-singular vectors :math:`V_k` of matrix :math:`M` can be computed with an arbitrary method like the one described in [Demmel90]_.

The final step is to find a permutation matrix :math:`Q_{p \times p}` such that the diagonal entries of :math:`\SigmaQ`are sorted in a descending order i.e :math:`\sigma_k \geq \sigma_{k+1}, for all k < p assuming n > p`.
The rows of the resulting matrix :math:`T = V^*Q` are the principal components of :math:`X`. Note that the rows
of :math:`T` are also the eigenvectors of the covariance matrix of :math:`X`. Additionally, the means and
variances of the initial dataset are returned.

Sign-flip technique
~~~~~~~~~~~~~~~~~~~
The eigenvectors (or the right-singular vectors) are not uniquely defined because the negative of any eigenvector is also an eigenvector of the input matrix. The signs of the eigenvectors or the singular vectors
often depend on the solver used. A sign-flip technique like the one proposed in [Bro07]_ helps remove the ambiguity.
The sign-flip function modifies the matrix :math:`T` as follows:

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
Given the inference data set :math:`X^'_{m \times p}` with :math:`m` observations, :math:`p`features, 
and the :math:`r \times p` transformation matrix :math:`T` produced at the training stage, 
the problem is to transform :math:`X^'_{m \times p}` to :math:`X^{''}_{m \times r}`, where :math:`X^{''}_{i}` is an
:math:`r`-dimensional transformed observation.

Each individual observation :math:`X^'_{i}` can be transformed by applying the following linear
transformation [Lang87]_ defined by the matrix :math:`T`,

.. math::
   :label: x_transform

   X^{''}_{i} = X^'_{i}T^T, \quad 1 \leq i \leq m.


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

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SMPD mode (only on GPU).

-------------
Usage example
-------------

.. include:: ../../../includes/decomposition/pca-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/decomposition/pca-examples.rst
