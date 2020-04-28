.. ******************************************************************************
.. * Copyright 2014-2020 Intel Corporation
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

.. _pca:

Principal Component Analysis
============================

.. toctree::
   :glob:
   :maxdepth: 4


Principal Component Analysis (PCA) is a method for exploratory data
analysis. PCA transforms a set of observations of possibly correlated
variables to a new set of uncorrelated variables, called principal
components. Principal components are the directions of the largest
variance, that is, the directions where the data is mostly spread
out.

Because all principal components are orthogonal to each other, there
is no redundant information. This is a way of replacing a group of
variables with a smaller set of new variables. PCA is one of powerful
techniques for dimension reduction.

Details
*******

Given a set :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})\}` of :math:`p`-dimensional
feature vectors or a :math:`p \times p` correlation matrix and the number of principal components :math:`p_r`, 
the problem is to compute :math:`p_r` principal directions
(eigenvectors) for the data set. The library returns the
transformation matrix :math:`T` of size :math:`p_r \times p`, which contains
eigenvectors in the row-major order and a vector of respective
eigenvalues in descending order.

|product| provides two methods for running PCA:

-  SVD
-  Correlation

Eigenvectors computed by PCA are not uniquely defined due to sign
ambiguity. PCA supports fast ad-hoc "sign flip" technique described
in the paper [Bro07]_. It modifies the signs of eigenvectors shown below:

   .. math::
	\hat{T_i} = T_i \cdot sgn(\max_{1 \leq j \leq p } |{T}_{i,j}|), i=1, \ldots ,p_r

where :math:`T`-transformation matrix is computed by PCA, :math:`T_i` - :math:`i`-th
row in the matrix, :math:`j` - column number, :math:`sgn` - signum function:

   .. math::
	   sgn(x) = \lbrace \begin{array}{c}
         -1, & x < 0,\\
		 0, & x = 0, \\
         1, & x > 0.\end{array}

You can provide these types of input data to the PCA algorithms of
the library:

-  Original, non-normalized data set

-  Normalized data set, where each feature has the zero mean and unit variance

-  Correlation matrix

Batch Processing
****************

Algorithm Input
---------------
The PCA algorithm accepts the input described below. Pass the
Input ID as a parameter to the methods that provide input for your algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - data
     - Use when the input data is a normalized or non-normalized data set.
       Pointer to the :math:`n \times p` numeric table that contains the input data set.
       This input can be an object of any class derived from NumericTable.
   * - correlation
     - Use when the input data is a correlation matrix. Pointer to the :math:`p \times p`
       numeric table that contains the correlation matrix. This input can be an
       object of any class derived from NumericTable except
       PackedTriangularMatrix.

Algorithm Parameters
--------------------

The PCA algorithm has the following parameters, depending on the
computation method parameter method:

.. list-table::
   :widths: 10 10 15 25
   :header-rows: 1
   :align: left

   * - Parameter
     - method
     - Default Value
     - Description
   * - algorithmFPType
     - defaultDense or svdDense
     - float
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be float or double.
   * - method
     - Not applicable
     - defaultDense
     - Available methods for PCA computation:

       - defaultDense - the correlation method
       - svdDense - the SVD method

   * - covariance
     - defaultDense
     - SharedPtr<covariance::Batch<algorithmFPType, covariance::defaultDense> >
     - The correlation and variance-covariance matrices algorithm to be used
       for PCA computations with the correlation method.
   * - normalization
     - svdDense
     - SharedPtr<normalization::zscore::Batch<algorithmFPType,
       normalization::zscore::defaultDense>>
     - The data normalization algorithm to be used for PCA computations with
       the SVD method. 
   * - nComponents
     - defaultDense, svdDense
     - 0
     - Number of principal components :math:`p_r`. If it is zero, the algorithm
       will compute the result for :math:`p_r = p`.
   * - isDeterministic
     - defaultDense, svdDense
     - false
     - If true, the algorithm applies the "sign flip" technique to the results.
   * - resultsToCompute
     - defaultDense, svdDense
     - none
     - The 64-bit integer flag that specifies which optional result to compute.

       Provide one of the following values to request a single characteristic
       or use bitwise OR to request a combination of the characteristics:

       -  mean
       -  variance
       -  eigenvalue

Algorithm Output
----------------

The PCA algorithm calculates the results described below. Pass the
Result ID as a parameter to the methods that access the results of
your algorithm. 

.. list-table::
   :widths: 25 25
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - eigenvalues
     - Pointer to the :math:`1 \times p_r` numeric table that contains eigenvalues
       in the descending order. By default, this result is an object of the
       HomogenNumericTable class, but you can define the result as an object of
       any class derived from NumericTable except PackedSymmetricMatrix,
       PackedTriangularMatrix, and CSRNumericTable.
   * - eigenvectors
     - Pointer to the :math:`p_r \times p` numeric table that contains eigenvectors
       in the row-major order. By default, this result is an object of the
       HomogenNumericTable class, but you can define the result as an object of
       any class derived from NumericTable except PackedSymmetricMatrix,
       PackedTriangularMatrix, and CSRNumericTable.
   * - means
     - Pointer to the :math:`1 \times p_r` numeric table that contains mean values
       for each feature.
       Optional.
       If correlation is provided then the vector is filed with zeroes.
   * - variances
     - Pointer to the :math:`1 \times p_r` numeric table that contains mean values
       for each feature.
       Optional.
       If correlation is provided then the vector is filed with zeroes.
   * - dataForTransform
     - Pointer to key value data collection containing the aggregated data for
       normalization and whitening with the following key value pairs:

       -  mean - mean
       -  variance - variance
       -  eigenvalue - eigenvalue

       If resultsToCompute does not contain mean, the dataForTransform means
       table is NULL. If resultsToCompute does not contain variances, the
       dataForTransform variances table is NULL. If resultsToCompute does not
       contain eigenvalues, the dataForTransform eigenvalues table is NULL.

Please note the following:

.. note::

   -  If the function result is not requested through the
      resultsToCompute parameter, the respective element of the result
      contains a NULL pointer.
   -  By default, each numeric table specified by the collection
      elements is an object of the HomogenNumericTable class, but you
      can define the result as an object of any class derived from
      NumericTable, except for PackedSymmetricMatrix,
      PackedTriangularMatrix, and CSRNumericTable.
   -  For the svdDense method n should not be less than :math:`p`. If :math:`n > p`,
      svdDense returns an error.


Online Processing
*****************

At this moment, the description of 
`online processing for Principal Component Analysis <https://software.intel.com/en-us/daal-programming-guide-online-processing-2>`_
is only available in Developer Guide for Intel(R) DAAL.

Distributed Processing
**********************

At this moment, the description of
`distributed processing for Principal Component Analysis <https://software.intel.com/en-us/daal-programming-guide-distributed-processing-4>`_
is only available in Developer Guide for Intel(R) DAAL.

Examples
********

.. tabs::

  .. tab:: DPC++

    Batch Processing:

    - :ref:`pca_cor_dense_batch.cpp`

  .. tab:: C++

    Batch Processing:

    -  :cpp_example:`pca_cor_dense_batch.cpp <pca/pca_cor_dense_batch.cpp>`
    -  :cpp_example:`pca_cor_csr_batch.cpp <pca/pca_cor_csr_batch.cpp>`
    -  :cpp_example:`pca_svd_dense_batch.cpp <pca/pca_svd_dense_batch.cpp>`

  .. tab:: Java*

    Batch Processing:

    -  :java_example:`PCACorDenseBatch.java <pca/PCACorDenseBatch.java>`
    -  :java_example:`PCACorCSRBatch.java <pca/PCACorCSRBatch.java>`
    -  :java_example:`PCASVDDenseBatch.java <pca/PCASVDDenseBatch.java>`

.. Python*:

.. -  pca_cor_dense_batch.py
.. -  pca_cor_csr_batch.py
.. -  pca_svd_dense_batch.py


Performance Considerations
**************************

To get the best overall performance of the PCA algorithm:

-  If input data is homogeneous, provide the input data and store
   results in homogeneous numeric tables of the same type as
   specified in the algorithmFPType class template parameter.
-  If input data is non-homogeneous, use AOS layout rather than SOA
   layout.

PCA computation using the correlation method involves the correlation
and variance-covariance matrices algorithm. Depending on the method
of this algorithm, the performance of PCA computations may vary. For
sparse data sets, use the methods of this algorithm for sparse data.

Batch Processing
----------------

Because the PCA in the batch processing mode performs
normalization for data passed as Input ID, to achieve the best
performance, normalize the input data set. To inform the algorithm
that the data is normalized, set the normalization flag for the
input numeric table that represents your data set by calling the
setNormalizationFlag() method of the NumericTableIface class.

Because the PCA with the correlation method (defaultDense) in the
batch processing mode is based on the computation of the
correlation matrix, to achieve the best performance, precompute
the correlation matrix. To pass the precomputed correlation matrix
to the algorithm, use correlation as Input ID.

Online Processing
-----------------

PCA with the SVD method (svdDense) in the online processing mode
is at least as computationally complex as in the batch processing
mode and has high memory requirements for storing auxiliary data
between calls to compute(). On the other hand, the online version
of the PCA with the SVD method may enable you to hide the latency
of reading data from a slow data source. To do this, implement
load prefetching of the next data block in parallel with the
compute() method for the current block.

Distributed Processing
----------------------

PCA with the SVD method (svdDense) in the distributed processing
mode requires gathering local-node p x p numeric tables on the
master node. When the amount of local-node work is small, that is,
when the local-node data set is small, the network data transfer
may become a bottleneck. To avoid this situation, ensure that
local nodes have a sufficient amount of work. For example,
distribute the input data set across a smaller number of nodes.

.. include:: ../../../opt-notice.rst