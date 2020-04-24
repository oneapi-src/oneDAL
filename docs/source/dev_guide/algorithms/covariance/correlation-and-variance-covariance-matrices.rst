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

Correlation and Variance-Covariance Matrices
********************************************

.. toctree::
   :glob:
   :maxdepth: 4

Variance-covariance and correlation matrices are among the most
important quantitative measures of a data set that characterize
statistical relationships involving dependence.

Specifically, the covariance measures the extent to which variables
"fluctuate together" (that is, co-vary). The correlation is the
covariance normalized to be between -1 and +1. A positive correlation
indicates the extent to which variables increase or decrease
simultaneously. A negative correlation indicates the extent to which
one variable increases while the other one decreases. Values close to
+1 and -1 indicate a high degree of linear dependence between
variables.


Details
=======

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of
dimension :math:`p`, the problem is to compute the sample means and
variance-covariance matrix or correlation matrix:


.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m(1), \ldots , m(p))`, where :math:`m\left(j\right)=\frac{1}{n}\sum _{i}{x}_{ij}`
   * - Variance-covariance matrix
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij}=\frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m(i))(x_{kj}-m(j))`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij}=\frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`



Batch Processing
================

Algorithm Input
---------------

The correlation and variance-covariance matrices algorithm accepts
the input described below. Pass the Input ID as a parameter to the
methods that provide input for your algorithm.

.. list-table::
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - data
     - Pointer to the :math:`n \times p` numeric table for which the variance-covariance or
       correlation matrix :math:`C` is computed. While the input for defaultDense,
       singlePassDense, or sumDense method can be an object of any class
       derived from NumericTable, the input for fastCSR, singlePassCSR, or
       sumCSR method can only be an object of the CSRNumericTable class.

Algorithm Parameters
--------------------

The correlation and variance-covariance matrices algorithm has the
following parameters:

.. list-table::
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - algorithmFPType
     - float
     - The floating-point type that the algorithm uses for intermediate computations. Can be float or double.
   * - method
     - defaultDense
     - Available methods for computation of correlation and variance-covariance matrices:

       + defaultDense - default performance-oriented method
       + singlePassDense - implementation of the single-pass algorithm proposed by D.H.D. West
       + sumDense - implementation of the algorithm in the cases where the
         basic statistics associated with the numeric table are pre-computed
         sums; returns an error if pre-computed sums are not defined
       + fastCSR - performance-oriented method for CSR numeric tables
       + singlePassCSR - implementation of the single-pass algorithm proposed by D.H.D. West; optimized for CSR numeric tables
       + sumCSR - implementation of the algorithm in the cases where the basic
         statistics associated with the numeric table are pre-computed sums;
         optimized for CSR numeric tables; returns an error if pre-computed
         sums are not defined

   * - outputMatrixType
     - covarianceMatrix
     - The type of the output matrix. Can be:

       + covarianceMatrix - variance-covariance matrix
       + correlationMatrix - correlation matrix

Algorithm Output
----------------

The correlation and variance-covariance matrices algorithm calculates
the result described below. Pass the Result ID as a parameter to the
methods that access the results of your algorithm.

.. list-table::
   :header-rows: 1
   :align: left


   * - Result ID
     - Result
   * - covariance
     - Use when outputMatrixType=covarianceMatrix. Pointer to the numeric table
       with the :math:`p \times p` variance-covariance matrix. By default, this result is an
       object of the HomogenNumericTable class, but you can define the result
       as an object of any class derived from NumericTable except
       PackedTriangularMatrix and CSRNumericTable.
   * - correlation
     - Use when outputMatrixType=correlationMatrix. Pointer to the numeric
       table with the :math:`p \times p` correlation matrix. By default, this result is an
       object of the HomogenNumericTable class, but you can define the result
       as an object of any class derived from NumericTable except
       PackedTriangularMatrix and CSRNumericTable.
   * - mean
     - Pointer to the :math:`1 \times p` numeric table with means. By default, this result
       is an object of the HomogenNumericTable class, but you can define the
       result as an object of any class derived from NumericTable except
       PackedTriangularMatrix, PackedSymmetricMatrix, and CSRNumericTable.


Online Processing
=================

At this moment, the description of
`online processing for Correlation and Variance-Covariance Matrices <https://software.intel.com/en-us/daal-programming-guide-online-processing-1>`_
is only available in Developer Guide for Intel(R) DAAL.

Distributed Processing
======================

At this moment, the description of
`distributed processing for Correlation and Variance-Covariance Matrices <https://software.intel.com/en-us/daal-programming-guide-distributed-processing-1>`_
is only available in Developer Guide for Intel(R) DAAL.

Examples
========

.. tabs::

  .. tab:: DPC++

    Batch Processing:

    - :ref:`cor_dense_batch.cpp`

    Online Processing:

    - :ref:`cor_dense_online.cpp`

  .. tab:: C++

    Batch Processing:

    -  :cpp_example:`cov_dense_batch.cpp <covariance/cov_dense_batch.cpp>`
    -  :cpp_example:`cov_csr_batch.cpp <covariance/cov_csr_batch.cpp>`


  .. tab:: Java*

    Batch Processing:

    -  :java_example:`CovDenseBatch.java <covariance/CovDenseBatch.java>`
    -  :java_example:`CovCSRBatch.java <covariance/CovCSRBatch.java>`

.. Python*:

.. -  cov_dense_batch.py
.. -  cov_csr_batch.py


Performance Considerations
==========================

To get the best overall performance when computing correlation or
variance-covariance matrices:

-  If input data is homogeneous, provide the input data and store
   results in homogeneous numeric tables of the same type as
   specified in the algorithmFPType class template parameter.
-  If input data is non-homogeneous, use AOS layout rather than SOA
   layout.


.. include:: ../../../opt-notice.rst
